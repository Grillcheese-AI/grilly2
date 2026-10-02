"""CubbyLLM's base: a MinGRU recurrence with sliding-window attention every
few layers, a context router, and a context-generated low-rank adapter
between the trunk and the head.

At the 450M base that is **32 layers: 11 attend over a window of 512
tokens, 21 carry one recurrent vector each.** Neither kind grows with the
context: a recurrent layer's state is ``(batch, hidden)``, an attention
layer's is a ring of the last ``window`` keys and values. So a decode
stream is a fixed size from its first token — 46 MB per stream at the
base's shape in float32, almost all of it the eleven rings, however long it
runs — where a transformer's cache grows by a row per token.

The computation follows CubbyLLM's own torch code op for op (its
``cubbyllm/model`` package: ``assembly.py``, ``backbone/hybrid.py``,
``backbone/mingru.py``, ``core/context.py``, ``core/generation.py``):

- **context**: the running mean of the token embeddings up to each
  position goes through a small router — ``softmax(W2 tanh(W1 m))`` over
  ``n_slots`` — and weights a table of slot vectors into ``ctx``.
- **trunk**: ``x += mixer(rms(x)); x += ffn(rms(x))`` per layer, where the
  mixer is windowed attention with rotary positions on layers
  ``i % attn_every == 0`` and MinGRU elsewhere. MinGRU is
  ``h_t = a_t * h_{t-1} + sigmoid(g_t) * tanh(v_t)`` with
  ``a_t = 0.001 + 0.998 * sigmoid(d_t)``; over a prompt it runs as the
  log-domain scan CubbyLLM trains with (Heinsen 2023), at one token as the
  recurrence itself. The FFN is SwiGLU at twice the width, no biases.
- **adapter**: ``h += tanh(h @ (B(c) A(c))^T)`` where ``A(c)``, ``B(c)`` mix
  ``gen_basis`` rank-``gen_rank`` bases by coefficients generated from
  ``ctx``. The ``d x d`` matrix is never built: both contractions run
  against the stacked bases.
- **head**: a plain linear projection to the (padded) vocabulary.

The parameter names are this file's, not CubbyLLM's pickle layout: a
checkpoint is exported to safetensors under these names on the CubbyLLM
side, so this model loads with ``load_state_dict(load_file(path),
assign=True)`` and nothing of CubbyLLM installed (CLAUDE.md §11).

Not yet: left-padded batches (the recurrence and the context mean would
have to skip the padding), several new tokens against a non-empty state
(chunked prefill), quantized projections and graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass

from ... import Tensor, _not_yet, arange, cat, float32, full, int64, tensor, zeros
from ... import nn
from ...nn import functional as F
from ..generation import CausalLMOutput, GenerationMixin
from .llama import _apply_rotary, _Rotary

__all__ = ["CubbyConfig", "CubbyForCausalLM", "CubbyState"]

#: the scan's floors, as CubbyLLM trains with them
_A_FLOOR, _A_SPAN, _LOG_FLOOR, _SCAN_EPS = 0.001, 0.998, 1e-8, 1e-30


@dataclass
class CubbyConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    window: int = 512
    attn_every: int = 3
    ctx_dim: int = 32
    n_slots: int = 8
    router_hidden: int = 64
    gen_hidden: int = 256
    gen_basis: int = 16
    gen_rank: int = 8
    #: rows of ``layer_emb`` in the generator; only row 0 is read (the
    #: adapter runs once, after the trunk), the rest are kept so a
    #: checkpoint round-trips
    gen_layers: int = 1
    ffn_mult: int = 2
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-5
    #: token ids below this are real; the head's rows above it are padding
    vocab_real: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_real is None:
            self.vocab_real = self.vocab_size
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(f"hidden_size {self.hidden_size} is not divisible by "
                             f"num_attention_heads {self.num_attention_heads}")
        if self.head_dim % 2:
            raise ValueError(f"head dim {self.head_dim} must be even for the rotary embedding")
        if self.attn_every < 1 or self.window < 1:
            raise ValueError("attn_every and window must be at least 1")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def is_attention(self, layer: int) -> bool:
        return layer % self.attn_every == 0

    @property
    def attention_layers(self) -> tuple:
        return tuple(i for i in range(self.num_hidden_layers) if self.is_attention(i))

    @property
    def recurrent_layers(self) -> tuple:
        return tuple(i for i in range(self.num_hidden_layers) if not self.is_attention(i))


class CubbyState:
    """Everything a batch of decode streams carries, allocated once.

    Per recurrent layer one ``(batch, hidden)`` vector; per attention
    layer a ring of the last ``window`` rotated keys and values,
    ``(batch, heads, window, head_dim)``, written at ``position % window``;
    and the running sum of the embeddings, which the context is the mean
    of. The size does not depend on how many tokens have gone through.

    The ring holds keys in slot order rather than time order. Attention is
    a softmax over keys and a sum over values weighted by it, so the order
    only changes the order the sum runs in. ``live`` is 0 on a written slot
    and -inf on one not yet written, one mask for every attention layer.
    """

    def __init__(self, config: CubbyConfig, batch_size: int) -> None:
        c = config
        self.window = c.window
        ring = (batch_size, c.num_attention_heads, c.window, c.head_dim)
        self.ctx_sum = zeros((batch_size, c.hidden_size))
        self.recurrent = [zeros((batch_size, c.hidden_size)) for _ in c.recurrent_layers]
        self.keys = [zeros(ring) for _ in c.attention_layers]
        self.values = [zeros(ring) for _ in c.attention_layers]
        self.live = full((1, 1, 1, c.window), float("-inf"))
        self.position = 0

    def get_seq_length(self, layer: int = 0) -> int:
        """Tokens this state has taken in (Hugging Face's name)."""
        return self.position

    def reset(self) -> None:
        """Forget every token; the buffers stay allocated."""
        for t in [self.ctx_sum, *self.recurrent, *self.keys, *self.values]:
            t.zero_()
        self.live.fill_(float("-inf"))
        self.position = 0

    def select_rows(self, index) -> None:
        """Keep only batch rows ``index`` (int64), for a batch that has
        dropped its finished streams."""
        self.ctx_sum = self.ctx_sum.index_select(0, index)
        self.recurrent = [h.index_select(0, index) for h in self.recurrent]
        self.keys = [k.index_select(0, index) for k in self.keys]
        self.values = [v.index_select(0, index) for v in self.values]

    @property
    def batch_size(self) -> int:
        return self.ctx_sum.shape[0]

    @property
    def nbytes(self) -> int:
        return 4 * sum(t.numel() for t in [self.ctx_sum, *self.recurrent, *self.keys, *self.values])


class CubbyRouter(nn.Module):
    """``ctx = softmax(out(tanh(proj(m)))) @ slots``, with ``m`` the
    running mean of the embeddings. Frozen in CubbyLLM (trained offline);
    here it is simply weights."""

    def __init__(self, config: CubbyConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.router_hidden)
        self.out = nn.Linear(config.router_hidden, config.n_slots)
        self.slots = nn.Parameter(zeros((config.n_slots, config.ctx_dim)))

    def forward(self, mean: Tensor) -> Tensor:
        p = F.softmax(self.out(self.proj(mean).tanh()), -1)
        return p.matmul(self.slots)


class CubbyAttention(nn.Module):
    """Causal attention over the last ``window`` tokens, rotary positions,
    one fused q/k/v projection, no biases."""

    def __init__(self, config: CubbyConfig, slot: int) -> None:
        super().__init__()
        self.slot = slot  # which ring in CubbyState
        self.heads, self.head_dim, self.window = config.num_attention_heads, config.head_dim, config.window
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _qkv(self, x: Tensor):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        return tuple(y.reshape(b, t, self.heads, self.head_dim).transpose(1, 2) for y in (q, k, v))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, state: CubbyState | None) -> Tensor:
        """A whole sequence from position 0. With ``state``, the last
        ``window`` keys and values go into its ring."""
        b, t, d = x.shape
        q, k, v = self._qkv(x)
        cos, sin = cos.narrow(0, 0, t), sin.narrow(0, 0, t)
        q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
        i = arange(t).unsqueeze(1)
        j = arange(t).unsqueeze(0)
        keep = (i >= j) & ((i - j) < self.window)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        if state is not None:
            m = min(t, self.window)
            slots = tensor([p % self.window for p in range(t - m, t)], dtype=int64)
            state.keys[self.slot].index_copy_(2, slots, k.narrow(2, t - m, m).contiguous())
            state.values[self.slot].index_copy_(2, slots, v.narrow(2, t - m, m).contiguous())
        return self.o_proj(out.transpose(1, 2).reshape(b, t, d))

    def step(self, x: Tensor, cos: Tensor, sin: Tensor, state: CubbyState, slot: Tensor) -> Tensor:
        """One token at ``state.position``, against the ring."""
        b, _, d = x.shape
        q, k, v = self._qkv(x)
        pos = state.position
        cos, sin = cos.narrow(0, pos, 1), sin.narrow(0, pos, 1)
        q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
        keys, values = state.keys[self.slot], state.values[self.slot]
        keys.index_copy_(2, slot, k.contiguous())
        values.index_copy_(2, slot, v.contiguous())
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=state.live)
        return self.o_proj(out.transpose(1, 2).reshape(b, 1, d))


class MinGRU(nn.Module):
    """``h_t = a_t * h_{t-1} + sigmoid(gate) * tanh(value)``,
    ``a_t = 0.001 + 0.998 * sigmoid(decay)``, ``h_0 = 0``."""

    def __init__(self, config: CubbyConfig, slot: int) -> None:
        super().__init__()
        self.slot = slot  # which vector in CubbyState.recurrent
        d = config.hidden_size
        self.gate = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.decay = nn.Linear(d, d)

    def _terms(self, x: Tensor):
        x_scan = self.gate(x).sigmoid() * self.value(x).tanh()
        a = self.decay(x).sigmoid() * _A_SPAN + _A_FLOOR
        return x_scan, a

    def forward(self, x: Tensor, state: CubbyState | None) -> Tensor:
        """A whole sequence from ``h_0 = 0``, by the log-domain scan:
        ``h = exp(a* + lcse(log x+ - a*)) - exp(a* + lcse(log x- - a*))``
        with ``a* = cumsum(log a)``, positive and negative parts apart."""
        x_scan, a = self._terms(x)
        a_star = a.clamp(min=_LOG_FLOOR).log().cumsum(1)
        pos = (x_scan.clamp(min=0) + _SCAN_EPS).log() - a_star
        neg = ((-x_scan).clamp(min=0) + _SCAN_EPS).log() - a_star
        h = (a_star + pos.logcumsumexp(1)).exp() - (a_star + neg.logcumsumexp(1)).exp()
        if state is not None:
            t = h.shape[1]
            state.recurrent[self.slot].copy_(h.narrow(1, t - 1, 1).reshape(h.shape[0], h.shape[2]))
        return h

    def step(self, x: Tensor, state: CubbyState) -> Tensor:
        """One token: the recurrence itself, in the state's buffer."""
        b, _, d = x.shape
        x_scan, a = self._terms(x.reshape(b, d))
        h = state.recurrent[self.slot]
        h.copy_(h * a + x_scan)
        return h.reshape(b, 1, d)


class CubbyMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))`` at ``ffn_mult`` times the width."""

    def __init__(self, config: CubbyConfig) -> None:
        super().__init__()
        d, w = config.hidden_size, config.ffn_mult * config.hidden_size
        self.gate_proj = nn.Linear(d, w, bias=False)
        self.up_proj = nn.Linear(d, w, bias=False)
        self.down_proj = nn.Linear(w, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CubbyLayer(nn.Module):
    def __init__(self, config: CubbyConfig, layer: int, slot: int) -> None:
        super().__init__()
        self.is_attention = config.is_attention(layer)
        self.mixer_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = CubbyAttention(config, slot) if self.is_attention else MinGRU(config, slot)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = CubbyMLP(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, state: CubbyState | None) -> Tensor:
        y = self.mixer_norm(x)
        x = x + (self.mixer(y, cos, sin, state) if self.is_attention else self.mixer(y, state))
        return x + self.ffn(self.ffn_norm(x))

    def step(self, x: Tensor, cos: Tensor, sin: Tensor, state: CubbyState, slot: Tensor) -> Tensor:
        y = self.mixer_norm(x)
        x = x + (self.mixer.step(y, cos, sin, state, slot) if self.is_attention else self.mixer.step(y, state))
        return x + self.ffn(self.ffn_norm(x))


class CubbyAdapter(nn.Module):
    """``h + tanh(h @ W(c)^T)``, ``W(c) = B(c) A(c)``, never materialised.

    ``coeffs = mix_out(gelu(mix_in([gelu(ln(ctx_proj(c))), layer_emb[0]])))``;
    ``A(c) = sum_n coeffs_n A_n``, ``B(c) = sum_n coeffs_n B_n``. Both are
    linear in the coefficients, so ``h @ A(c)^T`` is one matmul against the
    stacked ``A_n`` followed by a weighted sum over ``n``, and the same for
    ``B`` — ``(..., n*r)`` numbers per position rather than ``d*r``."""

    def __init__(self, config: CubbyConfig) -> None:
        super().__init__()
        c = config
        self.basis, self.rank, self.hidden = c.gen_basis, c.gen_rank, c.hidden_size
        self.ctx_proj = nn.Linear(c.ctx_dim, c.gen_hidden)
        self.ctx_norm = nn.LayerNorm(c.gen_hidden, eps=c.layer_norm_eps)
        self.layer_emb = nn.Parameter(zeros((c.gen_layers, c.gen_hidden)))
        self.mix_in = nn.Linear(2 * c.gen_hidden, c.gen_hidden)
        self.mix_out = nn.Linear(c.gen_hidden, c.gen_basis)
        self.A_basis = nn.Parameter(zeros((c.gen_basis, c.gen_rank, c.hidden_size)))
        self.B_basis = nn.Parameter(zeros((c.gen_basis, c.hidden_size, c.gen_rank)))

    def coefficients(self, ctx: Tensor) -> Tensor:
        h = F.gelu(self.ctx_norm(self.ctx_proj(ctx)))
        emb = self.layer_emb.narrow(0, 0, 1).reshape(h.shape[-1])
        h = cat((h, emb.expand(*h.shape[:-1], h.shape[-1])), -1)
        return self.mix_out(F.gelu(self.mix_in(h)))

    def forward(self, h: Tensor, ctx: Tensor) -> Tensor:
        n, r, d = self.basis, self.rank, self.hidden
        c = self.coefficients(ctx)                                     # (..., n)
        lead = tuple(h.shape[:-1])
        xa = F.linear(h, self.A_basis.reshape(n * r, d)).reshape(*lead, n, r)
        xa = (xa * c.unsqueeze(-1)).sum(-2)                            # (..., r)
        y = (c.unsqueeze(-1) * xa.unsqueeze(-2)).reshape(*lead, n * r)
        b_stack = self.B_basis.permute(0, 2, 1).reshape(n * r, d)      # [(n, r), d]
        return h + y.matmul(b_stack).tanh()


class CubbyModel(nn.Module):
    def __init__(self, config: CubbyConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.router = CubbyRouter(config)
        slots = {True: 0, False: 0}
        layers = []
        for i in range(config.num_hidden_layers):
            kind = config.is_attention(i)
            layers.append(CubbyLayer(config, i, slots[kind]))
            slots[kind] += 1
        self.layers = nn.ModuleList(layers)
        self.adapter = CubbyAdapter(config)


class CubbyForCausalLM(nn.Module, GenerationMixin):
    #: left padding would have to reach the recurrence and the context mean
    supports_padding = False

    def __init__(self, config: CubbyConfig) -> None:
        super().__init__()
        self.config = config
        self.model = CubbyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._rotary = _Rotary(config.head_dim, config.rope_theta)

    def make_cache(self, batch_size: int, max_length: int, keep=None) -> CubbyState:
        """A :class:`CubbyState`. ``max_length`` changes nothing: the state
        is the same size at any length."""
        if keep is not None:
            raise _not_yet("left-padded batches for CubbyForCausalLM")
        return CubbyState(self.config, batch_size)

    def context(self, embeddings: Tensor, state: CubbyState | None) -> Tensor:
        """``ctx`` at every position of ``embeddings`` (batch, T, hidden):
        the router over the running mean of the embeddings so far."""
        t = embeddings.shape[1]
        if state is not None and state.position:
            state.ctx_sum.copy_(state.ctx_sum + embeddings.reshape(embeddings.shape[0], -1))
            mean = (state.ctx_sum / float(state.position + 1)).unsqueeze(1)
        else:
            count = arange(1, t + 1, dtype=float32).unsqueeze(-1)
            mean = embeddings.cumsum(1) / count
            if state is not None:
                state.ctx_sum.copy_(embeddings.sum(1))
        return self.model.router(mean)

    def forward(self, input_ids: Tensor, past_key_values: CubbyState | None = None,
                logits_to_keep: int = 0) -> CausalLMOutput:
        """Logits for ``input_ids`` (batch, T). A fresh ``past_key_values``
        is filled from the whole prompt; a filled one takes one token."""
        state = past_key_values
        b, t = input_ids.shape
        start = 0 if state is None else state.position
        if start and t != 1:
            raise _not_yet("several new tokens against a non-empty CubbyState (chunked prefill)")
        cos, sin = self._rotary.tables(start + t)
        x = self.model.embed_tokens(input_ids)
        ctx = self.context(x, state)
        layers = self.model.layers
        if start:
            slot = tensor([start % self.config.window], dtype=int64)
            state.live.index_fill_(3, slot, 0.0)
            for layer in layers:
                x = layer.step(x, cos, sin, state, slot)
        else:
            if state is not None:
                m = min(t, self.config.window)
                state.live.index_fill_(3, tensor([p % self.config.window for p in range(t - m, t)],
                                                 dtype=int64), 0.0)
            for layer in layers:
                x = layer(x, cos, sin, state)
        if state is not None:
            state.position = start + t
        x = self.model.adapter(x, ctx)
        if logits_to_keep:
            x = x.narrow(1, t - logits_to_keep, logits_to_keep)
        return CausalLMOutput(logits=self.lm_head(x), past_key_values=state)
