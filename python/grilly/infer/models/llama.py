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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import Tensor, _not_yet, cat, from_numpy
from ... import nn
from ...nn import functional as F
from ..cache import KVCache
from ..generation import CausalLMOutput, GenerationMixin

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

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
        filled = 0 if cache is None else cache.get_seq_length(self.layer_idx)
        if filled == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            if cache is not None:
                cache.append(self.layer_idx, k, v)
        elif t == 1:
            cache.append(self.layer_idx, k, v)
            out = cache.attend(self.layer_idx, q)
        else:
            raise _not_yet("several new tokens against a non-empty KV cache (chunked prefill)")
        return self.o_proj(out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim))


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)
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

    def forward(self, input_ids: Tensor, past_key_values: KVCache | None = None, logits_to_keep: int = 0) -> CausalLMOutput:
        """Logits for ``input_ids`` (batch, T), continuing ``past_key_values``
        when given. ``logits_to_keep=n`` computes only the last ``n``
        positions' logits, as transformers does (0 keeps all)."""
        b, t = input_ids.shape
        start = 0 if past_key_values is None else past_key_values.get_seq_length(0)
        cos, sin = self._rotary.tables(start + t)
        cos, sin = cos.narrow(0, start, t), sin.narrow(0, start, t)
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, cos, sin, past_key_values)
        if logits_to_keep:
            x = x.narrow(1, t - logits_to_keep, logits_to_keep)
        logits = self.lm_head(self.model.norm(x))
        return CausalLMOutput(logits=logits, past_key_values=past_key_values)
