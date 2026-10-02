"""LFM2 — Liquid AI's hybrid decoder: gated short convolutions, with
grouped-query attention every few layers.

At LFM2.5-2.6B that is **22 conv blocks and 8 attention blocks** out of 30.
The conv blocks are the interesting half: each carries a rolling state of
``conv_L_cache`` positions — three — instead of a KV cache, so 22 of the 30
layers cost a fixed 3 columns per channel however long the context gets,
and only 8 layers grow. A 4096-token cache is 134 MB here rather than the
503 MB an all-attention model of this shape would want.

The attention blocks are Llama's, with per-head q/k norms, under different
names: ``out_proj`` for ``o_proj``, ``q_layernorm`` for ``q_norm``. Rather
than copy that code — it carries the fused qkv, the rotary-into-cache path
and the int8 projections — :class:`Lfm2Attention` subclasses
:class:`~grilly.infer.models.llama.LlamaAttention` and renames its
registered submodules, exposing the old names as properties so the shared
forward still resolves them. The checkpoint therefore loads with LFM2's own
parameter names and no remapping, which is the property
``grilly.infer.models`` keeps so a model runs without the Hugging Face
adapter installed (CLAUDE.md §11).

**The short conv, and why it is a decode step's friend.** Per block::

    B, C, x = in_proj(h).chunk(3)        # two gates and a value
    y = C * causal_depthwise_conv(B * x) # kernel conv_L_cache, per channel
    out = out_proj(y)

At one token the convolution is a weighted sum over the last three
positions, so the state is a ``(batch, hidden, 3)`` buffer that shifts by
one and takes the new column — **fixed size, written in place**, which is
what ``docs/capture.md`` asks of anything inside a recorded step. There is
no ``nn.Conv1d`` here and none is needed: a depthwise kernel of length L is
L shifted multiplies and L-1 adds, on the one dispatch path (§5).

The computation follows ``transformers.models.lfm2.modeling_lfm2`` op for
op, including the chunk order ``(B, C, x)`` and the decode form
``sum(state * weight, -1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ... import Tensor, _C, _not_yet, cat, is_grad_enabled, zeros
from ... import nn
from ...nn import functional as F
from ..cache import KVCache, PaddedKVCache
from ..generation import CausalLMOutput, GenerationMixin
from .llama import LlamaAttention, LlamaConfig, LlamaMLP, _Rotary

__all__ = ["Lfm2Config", "Lfm2ForCausalLM", "Lfm2ConvState"]

ATTENTION, CONV = "full_attention", "conv"


@dataclass
class Lfm2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    #: one of ``"conv"`` / ``"full_attention"`` per layer, in order
    layer_types: tuple = ()
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    conv_L_cache: int = 3
    conv_bias: bool = False
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if not self.layer_types:
            raise ValueError("Lfm2Config needs layer_types, one entry per layer")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for "
                f"{self.num_hidden_layers} layers"
            )
        unknown = set(self.layer_types) - {ATTENTION, CONV}
        if unknown:
            raise _not_yet(f"layer_types {sorted(unknown)} (grilly has {CONV!r} and {ATTENTION!r})")
        if self.conv_bias:
            raise _not_yet("conv_bias=True (LFM2's released checkpoints have no conv bias)")

    @property
    def attention_layers(self) -> tuple:
        """The indices that carry attention, in order. Their position in
        this tuple is their slot in the KV cache — the cache holds eight
        layers here, not thirty."""
        return tuple(i for i, kind in enumerate(self.layer_types) if kind == ATTENTION)

    def as_llama(self) -> LlamaConfig:
        """The attention blocks' config. LFM2's attention *is* Llama's with
        per-head q/k norms and no biases anywhere."""
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.norm_eps,
            rope_theta=self.rope_theta,
            qkv_bias=False,
            o_bias=False,
            qk_norm=True,
            tie_word_embeddings=self.tie_word_embeddings,
        )


def _rename(module: nn.Module, pairs) -> None:
    """Re-register `module`'s submodules under new names, keeping order.

    The shared forward still reaches them through the properties below, so
    this changes what a ``state_dict`` is keyed by and nothing else."""
    renamed = type(module._modules)()
    mapping = dict(pairs)
    for name, child in module._modules.items():
        renamed[mapping.get(name, name)] = child
    module._modules.clear()
    module._modules.update(renamed)


class Lfm2Attention(LlamaAttention):
    """Llama's attention under LFM2's parameter names.

    ``cache_slot`` is the index into the KV cache, which counts only
    attention layers — layer 2 of the model is cache slot 0.
    """

    def __init__(self, config: Lfm2Config, layer_idx: int, cache_slot: int) -> None:
        super().__init__(config.as_llama(), cache_slot)
        self.model_layer_idx = layer_idx
        _rename(self, [("o_proj", "out_proj"), ("q_norm", "q_layernorm"), ("k_norm", "k_layernorm")])

    o_proj = property(lambda self: self._modules["out_proj"])
    q_norm = property(lambda self: self._modules["q_layernorm"])
    k_norm = property(lambda self: self._modules["k_layernorm"])


class Lfm2MLP(LlamaMLP):
    """Llama's SwiGLU under LFM2's names: w1 gates, w3 lifts, w2 projects
    back — ``w2(silu(w1(x)) * w3(x))``, which is Llama's
    ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__(config.as_llama())
        _rename(self, [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")])

    gate_proj = property(lambda self: self._modules["w1"])
    up_proj = property(lambda self: self._modules["w3"])
    down_proj = property(lambda self: self._modules["w2"])


class _Conv1dWeight(nn.Module):
    """Holds ``conv.weight`` of shape ``(hidden, 1, L)``.

    A module only so the checkpoint's ``conv.conv.weight`` lands somewhere;
    the convolution itself is shifted multiplies, not a Conv1d.
    """

    def __init__(self, channels: int, length: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(zeros((channels, 1, length)))


class Lfm2ConvState:
    """The rolling state of every conv block: one ``(batch, hidden, L)``
    buffer per block, allocated once and written in place.

    This is the conv half of what a KV cache is for attention, and it is
    the reason 22 of 30 layers cost nothing as context grows. Fixed size
    and in-place updates are also exactly what a recorded step needs
    (``docs/capture.md``), so a decode loop over these is graphable.
    """

    def __init__(self, blocks: int, batch: int, hidden: int, length: int) -> None:
        self.length = length
        self.states = [zeros((batch, hidden, length)) for _ in range(blocks)]
        self.filled = False

    def reset(self) -> None:
        for state in self.states:
            state.zero_()
        self.filled = False

    def select_rows(self, index) -> None:
        """Keep only batch rows ``index`` (an int64 tensor), for a batch
        that has dropped its finished streams."""
        self.states = [state.index_select(0, index) for state in self.states]


class Lfm2ShortConv(nn.Module):
    """The gated short convolution.

    ``conv_slot`` indexes this block's rolling state, which counts only
    conv layers.
    """

    def __init__(self, config: Lfm2Config, conv_slot: int) -> None:
        super().__init__()
        hidden, length = config.hidden_size, config.conv_L_cache
        self.hidden, self.length, self.conv_slot = hidden, length, conv_slot
        self.conv = _Conv1dWeight(hidden, length)
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=config.conv_bias)
        self.out_proj = nn.Linear(hidden, hidden, bias=config.conv_bias)

    def _taps(self) -> Tensor:
        """The kernel as ``(hidden, L)`` — the stored shape is Conv1d's
        ``(hidden, 1, L)`` because that is what the checkpoint holds."""
        return self.conv.weight.reshape(self.hidden, self.length)

    def forward(self, x: Tensor, state: Lfm2ConvState | None, pad: Tensor | None = None) -> Tensor:
        """``pad``, ``(b, 1, t)`` with 0 on left padding, zeroes the padding
        before the convolution, so a row's first tokens see zeros behind
        them as they would alone. transformers zeroes the same positions
        (of the block's input, which in_proj, having no bias, maps to
        zero)."""
        b, t, _ = x.shape
        gates = self.in_proj(x)                                  # (b, t, 3*hidden)
        h = self.hidden
        gate_b = gates[..., :h]
        gate_c = gates[..., h:2 * h]
        value = gates[..., 2 * h:]
        bx = (gate_b * value).transpose(1, 2)                    # (b, hidden, t)
        if pad is not None:
            bx = bx * pad

        taps = self._taps()
        if state is not None and state.filled and t == 1:
            rolled = self._advance(state, bx)                    # (b, hidden, L)
            conv = (rolled * taps).sum(-1).reshape(b, h, 1)
        else:
            conv = self._over_sequence(bx, taps, t)
            if state is not None:
                self._seed(state, bx, t)
        y = (gate_c.transpose(1, 2) * conv).transpose(1, 2)      # (b, t, hidden)
        return self.out_proj(y.contiguous())

    def _advance(self, state: Lfm2ConvState, bx: Tensor) -> Tensor:
        """Shift the state one position and take the new column, in place.

        A shift rather than a true ring: at L=3 moving two columns is
        cheaper than rotating the kernel to follow a ring's phase, and it
        keeps the weights aligned with the state at every step.
        """
        buffer = state.states[self.conv_slot]
        buffer.copy_(cat((buffer.narrow(-1, 1, self.length - 1), bx), -1))
        return buffer

    def _seed(self, state: Lfm2ConvState, bx: Tensor, t: int) -> None:
        """After a prompt, the state is its last ``L`` columns, left-padded
        with zeros when the prompt is shorter — transformers pads the same
        way, so a one-token prompt leaves two zero columns."""
        buffer = state.states[self.conv_slot]
        keep = min(t, self.length)
        tail = bx.narrow(-1, t - keep, keep)
        if keep < self.length:
            pad = zeros((bx.shape[0], self.hidden, self.length - keep))
            tail = cat((pad, tail), -1)
        buffer.copy_(tail)
        state.filled = True

    def _over_sequence(self, bx: Tensor, taps: Tensor, t: int) -> Tensor:
        """Causal depthwise convolution over a whole prompt.

        ``out[:, :, j] = sum_i taps[:, i] * bx[:, :, j - (L-1-i)]``, with
        positions before the start reading zero. L shifted multiplies and
        L-1 adds; no Conv1d, and every term is one dispatch.
        """
        out = None
        for i in range(self.length):
            back = self.length - 1 - i                            # how far behind
            weight = taps[:, i].reshape(1, self.hidden, 1)
            if back == 0:
                shifted = bx
            elif back >= t:
                continue                                          # entirely before the start
            else:
                pad = zeros((bx.shape[0], self.hidden, back))
                shifted = cat((pad, bx.narrow(-1, 0, t - back)), -1)
            term = shifted * weight
            out = term if out is None else out + term
        return out if out is not None else zeros((bx.shape[0], self.hidden, t))


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int, slot: int) -> None:
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] == ATTENTION
        if self.is_attention:
            self.self_attn = Lfm2Attention(config, layer_idx, slot)
        else:
            self.conv = Lfm2ShortConv(config, slot)
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def mixer(self, y: Tensor, cos, sin, cache, conv_state, pos0: int, fused: bool) -> Tensor:
        if self.is_attention:
            return self.self_attn(y, cos, sin, cache, pos0, fused)
        # a left-padded prompt: zero the padding before it is convolved
        pad = cache.pad_columns(y.shape[1]) if getattr(cache, "padded", False) and y.shape[1] > 1 else None
        return self.conv(y, conv_state, pad)

    def forward(self, x: Tensor, cos, sin, cache, conv_state, pos0: int) -> Tensor:
        h = x + self.mixer(self.operator_norm(x), cos, sin, cache, conv_state, pos0, False)
        return h + self.feed_forward(self.ffn_norm(h))


class Lfm2Model(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        slots = {ATTENTION: 0, CONV: 0}
        layers = []
        for i, kind in enumerate(config.layer_types):
            layers.append(Lfm2DecoderLayer(config, i, slots[kind]))
            slots[kind] += 1
        self.layers = nn.ModuleList(layers)
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)


class Lfm2ForCausalLM(nn.Module, GenerationMixin):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._rotary = _Rotary(config.head_dim, config.rope_theta)
        self._conv_state: Lfm2ConvState | None = None
        self.tie_weights()

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    #: ``generate`` may give it left-padded batches: ``make_cache(keep=...)``
    supports_padding = True

    def make_cache(self, batch_size: int, max_length: int, keep=None):
        """A cache for the **attention** layers only — eight of thirty at
        LFM2.5-2.6B, so it is a quarter the size a layer-per-layer cache
        would be. The conv blocks' state is allocated beside it. With
        ``keep`` (the prompt's attention mask) it is a
        :class:`PaddedKVCache`, for a left-padded batch."""
        c = self.config
        self._conv_state = Lfm2ConvState(
            sum(1 for k in c.layer_types if k == CONV), batch_size, c.hidden_size, c.conv_L_cache)
        if keep is not None:
            return PaddedKVCache(len(c.attention_layers), batch_size,
                                 c.num_key_value_heads, c.head_dim, max_length, keep)
        return KVCache(len(c.attention_layers), batch_size,
                       c.num_key_value_heads, c.head_dim, max_length)

    def _select_rows(self, cache, index, undo: list) -> None:
        """The conv blocks' rolling state holds a row per stream too."""
        super()._select_rows(cache, index, undo)
        if self._conv_state is not None:
            self._conv_state.select_rows(index)

    def conv_state(self, batch_size: int = 1) -> Lfm2ConvState:
        """The rolling conv state, allocated on first use. ``generate``
        goes through ``make_cache``; this is for a caller driving
        ``forward`` directly."""
        if self._conv_state is None:
            c = self.config
            self._conv_state = Lfm2ConvState(
                sum(1 for k in c.layer_types if k == CONV), batch_size,
                c.hidden_size, c.conv_L_cache)
        return self._conv_state

    def _norm(self, x: Tensor, residual: Tensor | None, norm: nn.RMSNorm):
        w, eps = norm.weight._t, self.config.norm_eps
        if residual is None:
            return Tensor._wrap(_C.rms_norm(x._t, self.config.hidden_size, w, eps)), x
        y, h = _C.add_rms_norm(x._t, residual._t, self.config.hidden_size, w, eps)
        return Tensor._wrap(y), Tensor._wrap(h)

    def forward(self, input_ids: Tensor, past_key_values: KVCache | None = None,
                logits_to_keep: int = 0) -> CausalLMOutput:
        t = input_ids.shape[1]
        start = 0 if past_key_values is None else past_key_values.get_seq_length(0)
        cos, sin = self._rotary.tables(start + t)
        state = self.conv_state(input_ids.shape[0]) if past_key_values is not None else None
        layers = self.model.layers
        x = self.model.embed_tokens(input_ids)
        if is_grad_enabled():
            for layer in layers:
                x = layer(x, cos, sin, past_key_values, state, start)
            x = self.model.embedding_norm(x)
        else:
            # every residual add folded into the norm that follows it, so
            # the next layer's norm weight comes one step ahead
            y, residual = self._norm(x, None, layers[0].operator_norm)
            for i, layer in enumerate(layers):
                a = layer.mixer(y, cos, sin, past_key_values, state, start, True)
                y, residual = self._norm(a, residual, layer.ffn_norm)
                m = layer.feed_forward(y, True)
                nxt = layers[i + 1].operator_norm if i + 1 < len(layers) else self.model.embedding_norm
                y, residual = self._norm(m, residual, nxt)
            x = y
        if logits_to_keep:
            x = x.narrow(1, t - logits_to_keep, logits_to_keep)
        return CausalLMOutput(logits=self.lm_head(x), past_key_values=past_key_values)
