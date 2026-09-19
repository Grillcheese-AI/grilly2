"""A Llama-family decoder: Llama, Qwen2 and models built the same way
(RMSNorm, rotary position embeddings, grouped-query attention, a SwiGLU
MLP). It is the machinery's own model, so a safetensors checkpoint loads
and generates with no model-zoo adapter installed (CLAUDE.md §11).

The parameter names are Hugging Face's (``model.layers.0.self_attn.q_proj``
...), so a checkpoint of this family loads with
``model.load_state_dict(load_file(path), assign=True)`` and no remapping.
Reading a checkpoint's ``config.json`` into a :class:`LlamaConfig` is the
adapter's job.

The computation follows transformers' Qwen2 and Llama code, op for op:
- ``weight * (x * rsqrt(mean(x^2) + eps))``;
- rotary tables from ``1 / theta^(2i/d)``, with ``rotate_half``;
- ``down(silu(gate(x)) * up(x))``.
Decoding runs against a :class:`~grilly.infer.cache.KVCache`. The prompt
uses causal ``scaled_dot_product_attention``, then each new token runs
against the cache on the GPU.

**Two paths.** Without autograd (what ``generate`` runs), a step uses the
fused kernels: add-then-normalize, the rotary embedding straight from a
projection's layout into the cache, SwiGLU, and one int8 linear for q/k/v
and another for gate/up once the model is quantized. That is about 11
dispatches per layer against 47. Each kernel computes what the ops it
replaces compute, bit for bit, except the norm, whose sum runs in another
order. With autograd the same layer runs as plain grilly ops, which carry
gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import Tensor, _C, _not_yet, cat, from_numpy, is_grad_enabled
from ... import nn
from ...nn import functional as F
from ..cache import KVCache
from ..generation import CausalLMOutput, GenerationMixin
from ..quantization import Int8Tensor

__all__ = ["LlamaConfig", "LlamaForCausalLM"]


@dataclass
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    qkv_bias: bool = False  # Qwen2: True
    o_bias: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_act != "silu":
            raise _not_yet(f"LlamaConfig(hidden_act={self.hidden_act!r}) (grilly has silu)")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")


class _Rotary:
    """cos and sin tables of rotary position embeddings, on the device,
    grown as positions are needed. Computed as transformers does, in
    float32: ``inv_freq = 1 / theta^(arange(0, d, 2) / d)``, then
    ``freqs = positions * inv_freq``, duplicated."""

    def __init__(self, head_dim: int, theta: float) -> None:
        exponent = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(head_dim)
        self.inv_freq = (np.float32(1.0) / (np.float32(theta) ** exponent)).astype(np.float32)
        self.length = 0
        self.cos = self.sin = None

    def tables(self, length: int):
        if length > self.length:
            size = max(length, 2 * self.length, 256)
            freqs = np.arange(size, dtype=np.float32)[:, None] * self.inv_freq[None, :]
            emb = np.concatenate([freqs, freqs], axis=-1)
            self.cos, self.sin = from_numpy(np.cos(emb)), from_numpy(np.sin(emb))
            self.length = size
        return self.cos, self.sin


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """``x * cos + rotate_half(x) * sin``, rotate_half as transformers':
    ``cat(-x2, x1)`` of the two halves of the last dimension."""
    half = x.shape[-1] // 2
    return x * cos + cat((-x[..., half:], x[..., :half]), -1) * sin


def _fused_weight(cache: dict, modules) -> tuple:
    """One int8 weight and bias holding the modules' rows, built once, or
    (None, None) while their weights are float32. Rebuilt if the weights
    are replaced (a load, or quantization)."""
    weights = [m.weight for m in modules]
    key = tuple(id(w) for w in weights)
    if cache.get("key") != key:
        cache.clear()
        cache["key"] = key
        cache["weight"] = cache["bias"] = None
        if all(isinstance(w, Int8Tensor) for w in weights):
            cache["weight"] = Int8Tensor._cat(weights)
            biases = [m.bias for m in modules]
            if all(b is not None for b in biases):
                cache["bias"] = cat([b.detach() for b in biases], 0)
    return cache["weight"], cache["bias"]


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        c = config
        self.layer_idx = layer_idx
        self.heads, self.kv_heads, self.head_dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, bias=c.qkv_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=c.qkv_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=c.qkv_bias)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=c.o_bias)
        self._qkv: dict = {}  # a plain dict: not a registered parameter

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None, pos0: int, fused: bool) -> Tensor:
        b, t, _ = x.shape
        heads, kv_heads, dim = self.heads, self.kv_heads, self.head_dim
        filled = 0 if cache is None else cache.get_seq_length(self.layer_idx)
        if filled and t != 1:
            raise _not_yet("several new tokens against a non-empty KV cache (chunked prefill)")
        if not fused:
            return self._composed(x, cos, sin, cache, pos0, filled)

        weight, bias = _fused_weight(self._qkv, (self.q_proj, self.k_proj, self.v_proj))
        if weight is not None:
            qkv = F.linear(x, weight, bias)
            q_src, q_col = qkv, 0
            k_src, k_col = qkv, heads * dim
            v_src, v_col = qkv, (heads + kv_heads) * dim
        else:
            q_src, k_src, v_src = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q_col = k_col = v_col = 0
        q = Tensor._wrap(_C.rope(q_src._t, q_col, heads, dim, cos._t, sin._t, pos0))
        if filled:  # a decode step: rotate the keys into the cache, copy the values
            cache.append_rotary(self.layer_idx, k_src, k_col, v_src, v_col, cos, sin, pos0)
            out = cache.attend(self.layer_idx, q).reshape(b, 1, heads * dim)  # one token: already this layout
            return self.o_proj(out)
        k = Tensor._wrap(_C.rope(k_src._t, k_col, kv_heads, dim, cos._t, sin._t, pos0))
        v = v_src[..., v_col:v_col + kv_heads * dim].reshape(b, t, kv_heads, dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        if cache is not None:
            cache.append(self.layer_idx, k, v)
        return self.o_proj(attn.transpose(1, 2).reshape(b, t, heads * dim))

    def _composed(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None, pos0: int, filled: int) -> Tensor:
        """The same step as plain grilly ops, so gradients flow."""
        b, t, _ = x.shape
        heads, kv_heads, dim = self.heads, self.kv_heads, self.head_dim
        cos, sin = cos.narrow(0, pos0, t), sin.narrow(0, pos0, t)
        q = self.q_proj(x).reshape(b, t, heads, dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, kv_heads, dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, kv_heads, dim).transpose(1, 2)
        q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
        if filled:
            cache.append(self.layer_idx, k, v)
            out = cache.attend(self.layer_idx, q)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            if cache is not None:
                cache.append(self.layer_idx, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, heads * dim))


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.width = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._gate_up: dict = {}

    def forward(self, x: Tensor, fused: bool = False) -> Tensor:
        if not fused:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        weight, _ = _fused_weight(self._gate_up, (self.gate_proj, self.up_proj))
        if weight is not None:
            both = F.linear(x, weight)
            act = _C.swiglu(both._t, 0, both._t, self.width, self.width)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
            act = _C.swiglu(gate._t, 0, up._t, 0, self.width)
        return self.down_proj(Tensor._wrap(act))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None, pos0: int) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, pos0, False)
        return h + self.mlp(self.post_attention_layernorm(h))


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaForCausalLM(nn.Module, GenerationMixin):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._rotary = _Rotary(config.head_dim, config.rope_theta)
        self.tie_weights()

    def tie_weights(self) -> None:
        """With ``tie_word_embeddings``, the output projection is the
        embedding matrix, as in transformers."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def make_cache(self, batch_size: int, max_length: int) -> KVCache:
        c = self.config
        return KVCache(c.num_hidden_layers, batch_size, c.num_key_value_heads, c.head_dim, max_length)

    def _norm(self, x: Tensor, residual: Tensor | None, norm: nn.RMSNorm):
        """One fused kernel: (normalized, x + residual), or just the norm."""
        w, eps = norm.weight._t, self.config.rms_norm_eps
        if residual is None:
            return Tensor._wrap(_C.rms_norm(x._t, self.config.hidden_size, w, eps)), x
        y, h = _C.add_rms_norm(x._t, residual._t, self.config.hidden_size, w, eps)
        return Tensor._wrap(y), Tensor._wrap(h)

    def forward(self, input_ids: Tensor, past_key_values: KVCache | None = None, logits_to_keep: int = 0) -> CausalLMOutput:
        """Logits for ``input_ids`` (batch, T), continuing ``past_key_values``
        when given. ``logits_to_keep=n`` computes only the last ``n``
        positions' logits, as transformers does (0 keeps all)."""
        t = input_ids.shape[1]
        start = 0 if past_key_values is None else past_key_values.get_seq_length(0)
        cos, sin = self._rotary.tables(start + t)
        layers = self.model.layers
        x = self.model.embed_tokens(input_ids)
        if is_grad_enabled():  # gradients: plain ops, one layer at a time
            for layer in layers:
                x = layer(x, cos, sin, past_key_values, start)
            x = self.model.norm(x)
        else:
            # Each layer's residual add is folded into the norm that follows
            # it, so the next norm's weight comes one step ahead.
            y, residual = self._norm(x, None, layers[0].input_layernorm)
            for i, layer in enumerate(layers):
                a = layer.self_attn(y, cos, sin, past_key_values, start, True)
                y, residual = self._norm(a, residual, layer.post_attention_layernorm)
                m = layer.mlp(y, True)
                nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else self.model.norm
                y, residual = self._norm(m, residual, nxt)
            x = y
        if logits_to_keep:
            x = x.narrow(1, t - logits_to_keep, logits_to_keep)
        return CausalLMOutput(logits=self.lm_head(x), past_key_values=past_key_values)
