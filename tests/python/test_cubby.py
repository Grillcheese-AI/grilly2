"""CubbyLLM's base against a torch transcription of CubbyLLM's own code.

Reference (case (b)): CubbyLLM's forward and decode step, transcribed op
for op into ``reference`` below from its ``cubbyllm/model`` package
(``assembly.py``, ``backbone/hybrid.py``, ``backbone/mingru.py``,
``core/context.py``, ``core/generation.py``), so the test needs nothing of
CubbyLLM installed. It runs on the same weights in float64 — the reference
— and in float32 — the yardstick for what float32 can reach.

The bounds, derived rather than picked:

- **A prompt** runs MinGRU as CubbyLLM's log-domain scan. Its float32 error
  is not a dot-product bound: ``a* = cumsum(log a)`` falls by up to 6.9 a
  token, and ``exp(a* + lcse(...))`` loses about ``eps * |a*|`` relative to
  that. ``test_scan_scatter.py`` holds the scan to **twice torch float32's
  own error against float64**, and the prompt path is held to the same:
  grilly's logits may be at most twice as far from the float64 reference as
  torch's float32 logits are. The worst ratio is printed.
- **A decode step** runs the recurrence ``h = a*h + x``, so its error is
  the ordinary kind, bounded as ``test_lfm2.py`` bounds LFM2: the deepest
  reduction in a layer is the FFN's ``2 * hidden`` dot product, so
  ``|err| <= 2*hidden * 2^-24 * max|logits|``. torch float32 is asserted to
  sit inside the same bound, which is what shows the bound is not loose.

What would break and not be caught by a shape test: which layers attend
(``i % attn_every == 0``), the window (a key exactly ``window`` back must be
out, ``window - 1`` back in), the ring wrapping, the context being a
running mean rather than a mean of the whole prompt, and the adapter's
order of contraction. The model here is tiny with a window of 8, so a
40-token prompt and 30 decode steps cross the window and wrap the ring
several times.

Every test runs twice: on the 450M's shape (no memory) and with the
episodic memory on every second layer (``recall/read.py``'s forward,
transcribed into the reference: cosine top-3 over the tokens *more than
the window* back, a scaled dot-product softmax over them). Its decode
path has a store of 16 rows, so the 36-token decode also wraps the store
and reads only the last 16 aged tokens — the reference models that as a
lower bound on the candidates (``capacity``). What the memory adds that a
shape test would miss: the token exactly ``window`` back is attended, not
remembered; the one ``window + 1`` back is remembered, not attended; a
position with nothing beyond the window reads exactly zero; and the read
at a step sees the store *before* that step's token ages into it.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.infer.models import CubbyConfig, CubbyForCausalLM

torch = pytest.importorskip("torch")
F = torch.nn.functional

EPS32 = 2.0 ** -24
HIDDEN, HEADS, LAYERS, VOCAB, WINDOW = 32, 4, 5, 97, 8
MEM_EVERY, MEM_KEY, MEM_TOPK, MEM_CAP = 2, 8, 3, 16


def config(memory=False):
    mem = dict(mem_every=MEM_EVERY, mem_key=MEM_KEY, mem_topk=MEM_TOPK, mem_capacity=MEM_CAP) if memory else {}
    return CubbyConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=LAYERS,
                       num_attention_heads=HEADS, window=WINDOW, attn_every=3, ctx_dim=8,
                       n_slots=4, router_hidden=16, gen_hidden=16, gen_basis=4, gen_rank=2,
                       gen_layers=LAYERS, vocab_real=VOCAB - 2, **mem)


def has_memory(weights) -> bool:
    return "model.layers.0.memory.q.weight" in weights


@pytest.fixture(scope="module", params=["plain", "memory"])
def pair(request):
    """A grilly model with random weights, and the same weights as float64
    numpy arrays for the reference. Every parameter is non-trivial: the
    adapter's B bases start at zero in CubbyLLM, which would hide it, and
    so does the memory's exit ``o`` at growth."""
    rng = np.random.default_rng(7)
    model = CubbyForCausalLM(config(memory=request.param == "memory"))
    weights = {}
    with grilly.no_grad():
        for name, param in model.named_parameters():
            shape = tuple(param.shape)
            fan_in = shape[-1] if len(shape) > 1 else 1
            if name.endswith("norm.weight"):
                w = 1.0 + 0.1 * rng.standard_normal(shape)
            elif name.endswith(".bias"):
                w = 0.1 * rng.standard_normal(shape)
            else:
                w = rng.standard_normal(shape) / np.sqrt(fan_in)
            if name.endswith("mixer.decay.bias"):
                w = w + 1.0                               # CubbyLLM's init: ~0.73 retention
            weights[name] = w.astype(np.float32).astype(np.float64)
            param.copy_(grilly.from_numpy(weights[name].astype(np.float32)))
    return model, weights


# ── the reference ─────────────────────────────────────────────────────────

def reference(weights, ids, dtype, recurrence, capacity=None):
    """CubbyLLM's forward over ``ids`` (batch, T) from position 0.

    ``recurrence='scan'`` is the log-domain scan its training forward runs;
    ``'loop'`` is ``h = a*h + x`` a position at a time, which is what its
    decode step computes. Attention is the windowed causal mask over the
    whole sequence, which the step's ring reproduces. The memory read is
    ``MemoryRead.forward`` over the tokens beyond the window; ``capacity``
    keeps only the last that many of them, which is what the decode step's
    store holds once it has wrapped (``EpisodicStore`` at that capacity)."""
    c = config(memory=has_memory(weights))
    W = {k: torch.tensor(v, dtype=dtype) for k, v in weights.items()}
    ids = torch.tensor(ids)
    x = W["model.embed_tokens.weight"][ids]
    b, t, d = x.shape
    n = torch.arange(1, t + 1, dtype=dtype).unsqueeze(-1)
    mean = x.cumsum(dim=1) / n                                        # causal running mean
    r = "model.router."
    p = F.softmax(F.linear(torch.tanh(F.linear(mean, W[r + "proj.weight"], W[r + "proj.bias"])),
                           W[r + "out.weight"], W[r + "out.bias"]), dim=-1)
    ctx = p @ W[r + "slots"]

    def rms(v, w):
        return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    dh = d // HEADS
    inv = 1.0 / (10000.0 ** (torch.arange(0, dh, 2).to(dtype) / dh))
    ang = torch.outer(torch.arange(t).to(dtype), inv)
    emb = torch.cat([ang, ang], dim=-1)
    cos, sin = emb.cos().view(1, 1, t, dh), emb.sin().view(1, 1, t, dh)

    def rope(v):
        half = v.shape[-1] // 2
        return v * cos + torch.cat([-v[..., half:], v[..., :half]], dim=-1) * sin

    for i in range(c.num_hidden_layers):
        pre = f"model.layers.{i}."
        y = rms(x, W[pre + "mixer_norm.weight"])
        if c.is_attention(i):
            q, k, v = F.linear(y, W[pre + "mixer.qkv.weight"]).chunk(3, -1)
            q, k, v = (z.view(b, t, HEADS, dh).transpose(1, 2) for z in (q, k, v))
            q, k = rope(q), rope(k)
            j = torch.arange(t)
            keep = (j[:, None] >= j[None, :]) & (j[:, None] - j[None, :] < c.window)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
            y = F.linear(o.transpose(1, 2).reshape(b, t, d), W[pre + "mixer.o_proj.weight"])
        else:
            m = pre + "mixer."
            x_scan = torch.sigmoid(F.linear(y, W[m + "gate.weight"], W[m + "gate.bias"])) * \
                torch.tanh(F.linear(y, W[m + "value.weight"], W[m + "value.bias"]))
            a = 0.001 + 0.998 * torch.sigmoid(F.linear(y, W[m + "decay.weight"], W[m + "decay.bias"]))
            if recurrence == "scan":
                log_a = a.clamp(min=1e-8).log()
                a_star = log_a.cumsum(dim=1)
                eps = 1e-30
                hp = (a_star + ((x_scan.clamp(min=0) + eps).log() - a_star).logcumsumexp(dim=1)).exp()
                hn = (a_star + (((-x_scan).clamp(min=0) + eps).log() - a_star).logcumsumexp(dim=1)).exp()
                y = hp - hn
            else:
                h = torch.zeros(b, d, dtype=dtype)
                out = []
                for s in range(t):
                    h = h * a[:, s] + x_scan[:, s]
                    out.append(h)
                y = torch.stack(out, dim=1)
        x = x + y
        if c.is_memory(i):
            m = pre + "memory."
            hm = rms(x, W[pre + "memory_norm.weight"])
            q, k, v = (F.linear(hm, W[m + f"{p}.weight"]) for p in "qkv")
            sim = torch.einsum("bik,bjk->bij", F.normalize(q, dim=-1), F.normalize(k, dim=-1))
            j = torch.arange(t)
            back = j[:, None] - j[None, :]
            allowed = back > c.window
            if capacity is not None:
                allowed &= back <= c.window + capacity
            sim = sim.masked_fill(~allowed[None], float("-inf"))
            kk = min(c.mem_topk, t)
            topv, topi = sim.topk(kk, dim=-1)
            dk = k.shape[-1]
            k_set = torch.gather(k.unsqueeze(1).expand(b, t, t, dk), 2, topi.unsqueeze(-1).expand(b, t, kk, dk))
            v_set = torch.gather(v.unsqueeze(1).expand(b, t, t, d), 2, topi.unsqueeze(-1).expand(b, t, kk, d))
            valid = ~torch.isinf(topv)
            s = ((q.unsqueeze(2) * k_set).sum(-1) * dk ** -0.5).masked_fill(~valid, -1e9)
            r = F.linear((torch.softmax(s, dim=-1).unsqueeze(-1) * v_set).sum(2), W[m + "o.weight"])
            x = x + torch.where(~valid.any(-1, keepdim=True), torch.zeros_like(r), r)
        z = rms(x, W[pre + "ffn_norm.weight"])
        x = x + F.linear(F.silu(F.linear(z, W[pre + "ffn.gate_proj.weight"])) *
                         F.linear(z, W[pre + "ffn.up_proj.weight"]), W[pre + "ffn.down_proj.weight"])

    g = "model.adapter."
    hc = F.gelu(F.layer_norm(F.linear(ctx, W[g + "ctx_proj.weight"], W[g + "ctx_proj.bias"]),
                             (c.gen_hidden,), W[g + "ctx_norm.weight"], W[g + "ctx_norm.bias"], eps=1e-5))
    le = W[g + "layer_emb"][0]
    hc = torch.cat([hc, le.expand(*hc.shape[:-1], -1)], dim=-1)
    coeff = F.linear(F.gelu(F.linear(hc, W[g + "mix_in.weight"], W[g + "mix_in.bias"])),
                     W[g + "mix_out.weight"], W[g + "mix_out.bias"])
    xa = torch.einsum("...d,nrd->...nr", x, W[g + "A_basis"])
    xa = (xa * coeff.unsqueeze(-1)).sum(dim=-2)
    yy = coeff.unsqueeze(-1) * xa.unsqueeze(-2)
    x = x + torch.tanh(torch.einsum("...nr,ndr->...d", yy, W[g + "B_basis"]))
    return F.linear(x, W["lm_head.weight"]).numpy().astype(np.float64)


def prompt_logits(model, ids):
    with grilly.no_grad():
        return model(grilly.from_numpy(np.asarray(ids, np.int64))).logits.numpy().astype(np.float64)


def max_err(got, exact):
    return float(np.abs(got - exact).max())


# ── tests ─────────────────────────────────────────────────────────────────

def test_the_parameter_names_are_the_export_contract(pair):
    """CubbyLLM exports to these names; renaming one here breaks every
    exported checkpoint, so the set is spelled out."""
    model, weights = pair
    names = {n for n, _ in model.named_parameters()}
    per_attention = {"mixer_norm.weight", "mixer.qkv.weight", "mixer.o_proj.weight"}
    per_recurrent = {"mixer_norm.weight"} | {f"mixer.{p}.{k}" for p in ("gate", "value", "decay")
                                             for k in ("weight", "bias")}
    ffn = {"ffn_norm.weight", "ffn.gate_proj.weight", "ffn.up_proj.weight", "ffn.down_proj.weight"}
    expected = {"model.embed_tokens.weight", "lm_head.weight", "model.router.proj.weight",
                "model.router.proj.bias", "model.router.out.weight", "model.router.out.bias",
                "model.router.slots"}
    expected |= {f"model.adapter.{p}" for p in (
        "ctx_proj.weight", "ctx_proj.bias", "ctx_norm.weight", "ctx_norm.bias", "layer_emb",
        "mix_in.weight", "mix_in.bias", "mix_out.weight", "mix_out.bias", "A_basis", "B_basis")}
    memory = {"memory_norm.weight"} | {f"memory.{p}.weight" for p in "qkvo"}
    for i in range(LAYERS):
        local = (per_attention if i % 3 == 0 else per_recurrent) | ffn
        if has_memory(weights) and i % MEM_EVERY == 0:
            local |= memory
        expected |= {f"model.layers.{i}.{p}" for p in local}
    assert names == expected, sorted(names ^ expected)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 10, 40])
def test_a_prompt_matches_cubbyllm(pair, length):
    """Lengths either side of the window (8): below it the mask is plain
    causal, at 9 the first key drops out (and the memory still has nothing:
    it reads zero), at 10 the memory's first candidate appears, at 40 most
    keys are out and every memory read picks 3 of up to 31. Batch of two,
    so a row's context mean cannot borrow from the other row."""
    model, weights = pair
    rng = np.random.default_rng(length)
    ids = rng.integers(0, VOCAB, size=(2, length)).tolist()
    mine = prompt_logits(model, ids)
    exact = reference(weights, ids, torch.float64, "scan")
    theirs = reference(weights, ids, torch.float32, "scan")
    ratio = max_err(mine, exact) / max(max_err(theirs, exact), EPS32 * np.abs(exact).max())
    print(f"length {length}: grilly {max_err(mine, exact):.2e}, torch float32 "
          f"{max_err(theirs, exact):.2e}, ratio {ratio:.2f}")
    assert ratio <= 2.0, f"grilly is {ratio:.2f}x torch float32's error at length {length}"


def decode(model, prompt, count):
    """Prefill, then ``count`` tokens one at a time against the state,
    teacher-forced from ``prompt`` so every step's input is known."""
    with grilly.no_grad():
        state = model.make_cache(len(prompt), 0)
        head = [row[:-count] for row in prompt]
        out = [model(grilly.from_numpy(np.asarray(head, np.int64)), past_key_values=state)
               .logits.numpy()[:, -1]]
        for s in range(len(prompt[0]) - count, len(prompt[0]) - 1):
            step = grilly.from_numpy(np.asarray([[row[s]] for row in prompt], np.int64))
            out.append(model(step, past_key_values=state).logits.numpy()[:, -1])
    return np.stack(out, 1).astype(np.float64), state


@pytest.mark.parametrize("count", [31, 12])
def test_decode_matches_cubbyllms_step(pair, count):
    """36 tokens: five from the prompt and 31 steps, so the ring (8 slots)
    wraps three times, every recurrent vector carries 35 tokens, and the
    memory's store (16 rows) fills at step 24 and wraps for the last 12;
    or 24 from the prompt and 12 steps, so the prompt itself has to leave
    the store full (tokens 0-15, the next step overwrites row 0) and the
    ring holding the last 8. Each step's logits against the float64
    recurrence at that position.

    Not tested, because it is not the step's behaviour: a prompt longer
    than ``window + capacity`` reads its whole beyond-window past during
    the prefill, as CubbyLLM's training forward does, and only the steps
    after it see a store capped at the last ``capacity`` aged tokens."""
    model, weights = pair
    rng = np.random.default_rng(11)
    ids = rng.integers(0, VOCAB, size=(2, 36)).tolist()
    mine, _ = decode(model, ids, count)
    tail = slice(len(ids[0]) - count - 1, len(ids[0]) - 1)
    exact = reference(weights, ids, torch.float64, "loop", capacity=MEM_CAP)[:, tail]
    theirs = reference(weights, ids, torch.float32, "loop", capacity=MEM_CAP)[:, tail]
    bound = 2 * HIDDEN * EPS32 * np.abs(exact).max()
    print(f"decode: grilly {max_err(mine, exact) / bound:.2f} of the bound, "
          f"torch float32 {max_err(theirs, exact) / bound:.2f}")
    assert max_err(theirs, exact) <= bound, "torch float32 itself is outside the bound"
    assert max_err(mine, exact) <= bound, f"grilly is {max_err(mine, exact) / bound:.2f} of the bound"
    if has_memory(weights) and count == 31:
        # the wrap is real: the whole beyond-window past gives different logits at the end
        unbounded = reference(weights, ids, torch.float64, "loop")[:, tail]
        assert max_err(unbounded, exact) > bound


def test_greedy_tokens_are_identical_to_cubbyllm(pair):
    """The bar: argmax is a decision, so ``==``, through ``generate``, long
    enough for the ring to wrap four times."""
    model, weights = pair
    prompt, count = [5, 9, 14, 2], 32
    got = model.generate(grilly.tensor([prompt]), max_new_tokens=count).tolist()[0][len(prompt):]
    ids = list(prompt)
    for _ in range(count):
        ids.append(int(np.argmax(reference(weights, [ids], torch.float64, "loop", capacity=MEM_CAP)[0, -1])))
    want = ids[len(prompt):]
    if got != want:
        at = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        raise AssertionError(f"diverged at token {at}: {got[at]} vs {want[at]}")


def test_the_state_is_the_same_size_at_any_length(pair):
    """The inference claim: a stream costs the same after 3 tokens as after
    40. Buffers are written in place, never replaced. With memory the
    store is sized by the stream's known length (``max_length - window``
    rows, ``mem_capacity`` at most) and then never grows either."""
    model, _ = pair
    c = model.config
    with grilly.no_grad():
        state = model.make_cache(1, 0)
        model(grilly.tensor([[1, 2, 3]]), past_key_values=state)
        size = state.nbytes
        buffers = [id(t) for t in state.buffers()]
        for token in range(37):
            model(grilly.tensor([[token % VOCAB]]), past_key_values=state)
    assert state.position == 40
    assert state.nbytes == size
    assert [id(t) for t in state.buffers()] == buffers
    per_stream = 4 * (HIDDEN * (1 + len(c.recurrent_layers)) + 2 * len(c.attention_layers) * WINDOW * HIDDEN)
    per_stream += 4 * len(c.memory_layers) * ((2 * MEM_KEY + HIDDEN) * MEM_CAP + (MEM_KEY + HIDDEN) * WINDOW)
    assert size == per_stream
    if c.memory_layers:
        assert model.make_cache(1, 20).capacity == 20 - WINDOW
        assert model.make_cache(1, 9).capacity == MEM_TOPK          # never fewer rows than the read gathers
        assert model.make_cache(1, 100).capacity == MEM_CAP


def test_selected_rows_continue_as_they_would_have(pair):
    """Dropping a finished stream must keep every buffer's rows aligned: the
    recurrent vectors, both rings, the context sum and the memory's stores
    and rings (four tokens are in the store twelve tokens in). A misaligned
    row continues someone else's stream, an O(1) error; so the kept rows
    are held to the decode bound against the float64 reference of their own
    streams, rows 2 and 0 of three, twelve tokens in (past the window).

    Not compared to the unselected batch at rtol 1e-7: the same step at a
    batch of two and a batch of three differs by float32's error through
    five layers — measured 3.1e-6 on logits of magnitude ~5, about the
    decode error each has against float64 — not by one reassociation, so
    case (a)'s 1e-7 does not describe it."""
    model, weights = pair
    rng = np.random.default_rng(5)
    ids = rng.integers(0, VOCAB, size=(3, 12)).tolist()
    nxt = [4, 8, 15]
    with grilly.no_grad():
        state = model.make_cache(3, 0)
        model(grilly.tensor(ids), past_key_values=state)
        state.select_rows(grilly.tensor([2, 0]))
        kept = model(grilly.tensor([[nxt[2]], [nxt[0]]]), past_key_values=state).logits.numpy()[:, -1]
    streams = [ids[2] + [nxt[2]], ids[0] + [nxt[0]]]
    exact = reference(weights, streams, torch.float64, "loop")[:, -1]
    bound = 2 * HIDDEN * EPS32 * np.abs(exact).max()
    print(f"selected rows: grilly {max_err(kept, exact) / bound:.2f} of the bound")
    assert max_err(kept.astype(np.float64), exact) <= bound


def test_chunked_prefill_and_padding_say_not_yet(pair):
    model, _ = pair
    with grilly.no_grad():
        state = model.make_cache(1, 0)
        model(grilly.tensor([[1, 2]]), past_key_values=state)
        with pytest.raises(NotImplementedError):
            model(grilly.tensor([[3, 4]]), past_key_values=state)
    with pytest.raises(NotImplementedError):
        model.generate(grilly.tensor([[0, 5], [6, 7]]), attention_mask=grilly.tensor([[0, 1], [1, 1]]),
                       max_new_tokens=2)
