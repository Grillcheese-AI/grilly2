"""LFM2 against ``transformers``, op for op.

Reference: **transformers' own `Lfm2ForCausalLM`** (case (b)), on a small
random model whose weights are copied into both sides. A different
implementation and a different summation order, so the bound is derived
rather than picked: the deepest reduction in a layer is the MLP's
``intermediate_size``-long dot product, so logits are compared against a
float64 reference under ``n * eps * (|x| @ |W|)`` and the worst ratio is
printed.

A random model is the right size for this and the right *shape*: it
exercises every op the architecture has — the gated short convolution and
its rolling state, per-head q/k norms, grouped-query attention, rotary at
LFM2's theta, SwiGLU, a tied head — at a size that fits anywhere. The real
2.6B checkpoint is 10.8 GB in float32, past this card, so it is checked in
int8 for throughput rather than for bits.

What would break and not be caught by a shape test: the conv's chunk order
(``B, C, x``, not ``x, B, C``), which layers convolve and which attend, and
the state's padding convention when a prompt is shorter than the kernel.
All three are compared here against transformers running the same weights.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.infer.models import Lfm2Config, Lfm2ForCausalLM

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
lfm2 = pytest.importorskip("transformers.models.lfm2.modeling_lfm2")

HIDDEN, INTER, HEADS, KV_HEADS, VOCAB = 32, 64, 4, 2, 97
LAYER_TYPES = ("conv", "conv", "full_attention", "conv")


def configs():
    """The same model, described to both sides."""
    mine = Lfm2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
        num_hidden_layers=len(LAYER_TYPES), num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, layer_types=LAYER_TYPES,
        conv_L_cache=3, norm_eps=1e-5, rope_theta=1_000_000.0,
        tie_word_embeddings=True,
    )
    theirs = transformers.AutoConfig.for_model(
        "lfm2",
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
        num_hidden_layers=len(LAYER_TYPES), num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, layer_types=list(LAYER_TYPES),
        conv_L_cache=3, norm_eps=1e-5, rope_theta=1_000_000.0,
        tie_word_embeddings=True, block_auto_adjust_ff_dim=False,
        block_multiple_of=1, block_ffn_dim_multiplier=1.0,
    )
    return mine, theirs


@pytest.fixture(scope="module")
def pair():
    """A random transformers model and a grilly one holding its weights."""
    torch.manual_seed(0)
    mine_config, their_config = configs()
    reference = lfm2.Lfm2ForCausalLM(their_config).eval().float()
    model = Lfm2ForCausalLM(mine_config)

    have = dict(model.named_parameters())
    theirs = dict(reference.named_parameters())
    theirs.update(dict(reference.named_buffers()))
    missing = sorted(set(have) - set(theirs))
    assert not missing, f"grilly has parameters transformers does not: {missing}"
    with grilly.no_grad():
        for name, param in have.items():
            param.copy_(grilly.from_numpy(theirs[name].detach().numpy().copy()))
    model.tie_weights()
    return model, reference


def bound_ratio(got, reference, magnitude, terms):
    bound = terms * 2.0 ** -24 * np.maximum(magnitude, 1e-30)
    return float((np.abs(np.asarray(got, np.float64) - reference) / bound).max())


def logits_of(model, reference, ids):
    """Both sides' logits for one prompt, float64 reference from
    transformers run in double."""
    with grilly.no_grad():
        mine = model(grilly.from_numpy(np.asarray(ids, np.int64)[None, :])).logits.numpy()[0]
    with torch.no_grad():
        theirs = reference(torch.tensor([ids])).logits[0].numpy()
        exact = reference.double()(torch.tensor([ids])).logits[0].numpy().astype(np.float64)
        reference.float()
    return mine, theirs, exact


def test_the_parameter_names_are_the_checkpoints(pair):
    """No remapping: a checkpoint loads into this model as it is, which is
    what lets a model run with the Hugging Face adapter uninstalled."""
    model, reference = pair
    mine = {n for n, _ in model.named_parameters()}
    theirs = {n for n, _ in reference.named_parameters()}
    assert mine == theirs or mine == theirs - {"lm_head.weight"}, sorted(mine ^ theirs)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 11])
def test_a_prompt_matches_transformers(pair, length):
    """Lengths either side of the conv kernel (3): below it the state is
    zero-padded, at it exactly filled, above it rolling."""
    model, reference = pair
    ids = list(range(3, 3 + length))
    mine, theirs, exact = logits_of(model, reference, ids)
    scale = np.abs(exact).max()
    ratio = bound_ratio(mine, exact, scale, INTER)
    assert ratio <= 1.0, f"grilly is {ratio:.2f} of the bound at length {length}"
    assert bound_ratio(theirs, exact, scale, INTER) <= 1.0, "transformers itself is outside the bound"


def test_greedy_tokens_are_identical_to_transformers(pair):
    """The bar. Argmax is a decision, not a number, so this is ``==`` —
    and it runs long enough for the conv state to roll and the KV cache to
    fill several times."""
    model, reference = pair
    prompt = [5, 9, 14, 2]
    count = 40
    # a random model lands on the configured EOS almost at once, and a
    # one-token comparison would prove nothing; stop only on the count
    reference.generation_config.eos_token_id = None
    reference.generation_config.pad_token_id = 0
    mine = generate_greedy(model, prompt, count)
    theirs = reference.generate(torch.tensor([prompt]), max_new_tokens=count,
                                do_sample=False, use_cache=True)[0].tolist()[len(prompt):]
    assert len(mine) == len(theirs) == count, f"{len(mine)} vs {len(theirs)} tokens"
    if mine != theirs:
        at = next(i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b)
        raise AssertionError(f"diverged at token {at}: {mine[at]} vs {theirs[at]}")


def generate_greedy(model, prompt, count):
    """Prefill, then one token at a time against the caches — the decode
    path, not a re-run of the prompt."""
    out = []
    with grilly.no_grad():
        cache = model.make_cache(1, len(prompt) + count + 1)
        ids = grilly.from_numpy(np.asarray(prompt, np.int64)[None, :])
        logits = model(ids, past_key_values=cache).logits
        token = int(np.argmax(logits.numpy()[0, -1]))
        out.append(token)
        for _ in range(count - 1):
            step = grilly.from_numpy(np.asarray([[token]], np.int64))
            logits = model(step, past_key_values=cache).logits
            token = int(np.argmax(logits.numpy()[0, -1]))
            out.append(token)
    return out


def test_decode_equals_running_the_whole_prefix(pair):
    """The conv state and the KV cache have to carry exactly what a fresh
    forward would recompute; if the state's shift were off by one, this is
    where it shows."""
    model, _ = pair
    prompt = [7, 1, 19, 4, 22, 3]
    with grilly.no_grad():
        cache = model.make_cache(1, 32)
        model(grilly.from_numpy(np.asarray(prompt[:-1], np.int64)[None, :]), past_key_values=cache)
        stepped = model(grilly.from_numpy(np.asarray([[prompt[-1]]], np.int64)),
                        past_key_values=cache).logits.numpy()[0, -1]
        model._conv_state = None
        whole = model(grilly.from_numpy(np.asarray(prompt, np.int64)[None, :])).logits.numpy()[0, -1]
    rel = np.abs(stepped - whole).max() / max(np.abs(whole).max(), 1e-30)
    assert rel < 1e-5, f"decode drifted from a full forward by {rel:.2e}"


def test_the_cache_holds_only_the_attention_layers(pair):
    """22 of 30 layers convolve at the real shape, so a cache sized by
    layer count would be four times too big. This is the memory claim."""
    model, _ = pair
    cache = model.make_cache(1, 128)
    assert len(model.config.attention_layers) == LAYER_TYPES.count("full_attention")
    assert cache.num_layers == len(model.config.attention_layers)
    state = model.conv_state(1)
    assert len(state.states) == LAYER_TYPES.count("conv")
    assert tuple(state.states[0].shape) == (1, HIDDEN, 3)
    # the conv half costs a fixed three columns a channel however long the
    # context runs, where a KV layer grows with it
    conv_bytes = sum(s.numel() * 4 for s in state.states)
    assert conv_bytes < cache.nbytes, (conv_bytes, cache.nbytes)


def test_the_conv_state_is_fixed_size_and_written_in_place(pair):
    """What ``docs/capture.md`` needs from it: the buffers never change
    shape and never get replaced, so a recorded step keeps writing the
    same ones."""
    model, _ = pair
    with grilly.no_grad():
        cache = model.make_cache(1, 64)
        model(grilly.from_numpy(np.asarray([[5, 6, 7]], np.int64)), past_key_values=cache)
        state = model.conv_state(1)
        identity = [id(s) for s in state.states]
        shapes = [tuple(s.shape) for s in state.states]
        for token in (8, 9, 10, 11):
            model(grilly.from_numpy(np.asarray([[token]], np.int64)), past_key_values=cache)
        assert [id(s) for s in state.states] == identity, "a conv state buffer was replaced"
        assert [tuple(s.shape) for s in state.states] == shapes


# ── left-padded batches ──────────────────────────────────────────────────

UNEVEN = [[5, 9, 14, 2, 30, 8], [7, 1], [19, 4, 22, 3]]  # [7, 1] is shorter than the kernel


def left_padded(prompts):
    width = max(len(p) for p in prompts)
    return ([[0] * (width - len(p)) + p for p in prompts],
            [[0] * (width - len(p)) + [1] * len(p) for p in prompts])


def test_a_left_padded_batch_generates_what_each_row_generates_alone(pair):
    """The conv has no mask of its own: padding must be zeroed before it is
    convolved, or a row's first tokens read the pads' columns behind them.
    Long enough for the conv state to roll and the cache to fill."""
    model, _ = pair
    ids, mask = left_padded(UNEVEN)
    width, count = len(ids[0]), 24
    got = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=count).tolist()
    for r, p in enumerate(UNEVEN):
        mine, solo = got[r][width:], generate_greedy(model, p, count)
        if mine != solo:
            at = next(i for i, (a, b) in enumerate(zip(mine, solo)) if a != b)
            raise AssertionError(f"row {r} diverged from itself alone at token {at}: {mine[at]} vs {solo[at]}")


def test_a_left_padded_batch_generates_what_transformers_generates(pair):
    model, reference = pair
    reference.generation_config.eos_token_id = None
    reference.generation_config.pad_token_id = 0
    ids, mask = left_padded(UNEVEN)
    mine = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=24).tolist()
    with torch.no_grad():
        theirs = reference.generate(torch.tensor(ids), attention_mask=torch.tensor(mask), max_new_tokens=24,
                                    do_sample=False, use_cache=True).tolist()
    assert mine == theirs


def test_dropping_finished_rows_keeps_the_conv_state_aligned(pair, monkeypatch):
    """The conv state holds a row per stream as the KV cache does; when a
    row leaves, both have to lose the same row or the survivors convolve
    someone else's history."""
    model, _ = pair
    ids, mask = left_padded(UNEVEN)
    width = len(ids[0])
    free = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=16).tolist()
    eos = free[1][width + 2]
    drops = []
    original = model._select_rows

    def spy(cache, index, undo):
        drops.append(index.tolist())
        original(cache, index, undo)

    monkeypatch.setattr(model, "_select_rows", spy)
    got = model.generate(grilly.tensor(ids), attention_mask=grilly.tensor(mask), max_new_tokens=16,
                         eos_token_id=eos, pad_token_id=0).tolist()
    assert drops, "no row finished early, so this compared nothing"
    for r, p in enumerate(UNEVEN):
        solo = model.generate(grilly.tensor([p]), max_new_tokens=16, eos_token_id=eos).tolist()[0][len(p):]
        row = got[r][width:]
        assert row == solo + [0] * (len(row) - len(solo)), f"row {r}"
