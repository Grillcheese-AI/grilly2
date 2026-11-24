"""``generate`` with ``model.shortlist``: greedy decoding of one row picks
each token with a :class:`~grilly.infer.shortlist.ShortlistHead`, and
everything else runs the full head.

Tolerance: exact, ``==`` on tokens. With a route over every row the union is
the whole vocabulary, so the token is the full head's argmax (random
weights, where the top two logits are far apart next to fp32 rounding).
With a narrow route the expected tokens come from an independent loop over
``hidden`` and ``ShortlistHead.union``, scored in float64: that pins what
``generate`` feeds the route (the state the head reads, and the token just
fed in as ``prev``), not only that it calls it.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
from grilly.infer.generation import GenerationConfig, GenerationMixin
from grilly.infer.models import CubbyConfig, CubbyForCausalLM
from grilly.infer.shortlist import ShortlistHead

HIDDEN, VOCAB, CLUSTERS = 128, 300, 6        # the route needs dim a multiple of 128
PROMPT, COUNT = [5, 9, 14, 2, 77, 120], 24


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(11)
    m = CubbyForCausalLM(CubbyConfig(vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=3,
                                     num_attention_heads=4, window=8, attn_every=3, ctx_dim=8, n_slots=4,
                                     router_hidden=16, gen_hidden=16, gen_basis=4, gen_rank=2, gen_layers=3,
                                     vocab_real=VOCAB - 4))
    with grilly.no_grad():
        for name, param in m.named_parameters():
            shape = tuple(param.shape)
            if name.endswith("norm.weight"):
                w = 1.0 + 0.1 * rng.standard_normal(shape)
            elif name.endswith(".bias"):
                w = 0.1 * rng.standard_normal(shape)
            else:
                w = rng.standard_normal(shape) / np.sqrt(shape[-1] if len(shape) > 1 else 1)
            param.copy_(grilly.from_numpy(w.astype(np.float32)))
    return m


def _route(model, near, context, seed=2):
    """Clusters that split the rows at random, centroids at their means."""
    w = model.lm_head.weight.detach().numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(VOCAB)
    lists = np.array_split(order, CLUSTERS)
    cent = np.stack([w[c].mean(0) for c in lists]).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in lists])])
    succ = rng.integers(-1, VOCAB, (VOCAB, 3))
    return ShortlistHead(model.lm_head.weight, cent, np.concatenate(lists), offsets, succ,
                         near=near, context=context, ranks=CLUSTERS)


def _generate(model, head, prompt=None, **kw):
    model.shortlist = head
    try:
        return model.generate(grilly.tensor(prompt or [PROMPT]), max_new_tokens=COUNT, **kw).tolist()
    finally:
        model.shortlist = None


class _Streamer:
    def __init__(self):
        self.seen = []

    def put(self, t):
        self.seen.append(t.tolist())

    def end(self):
        self.seen.append("end")


def test_a_route_over_every_row_decodes_as_the_full_head(model):
    head = _route(model, near=VOCAB, context=VOCAB)
    full_stream, route_stream = _Streamer(), _Streamer()
    full = _generate(model, None, streamer=full_stream)
    routed = _generate(model, head, streamer=route_stream)
    assert routed == full
    assert route_stream.seen == full_stream.seen
    assert head.picks == COUNT and head.last.rows == VOCAB


def test_a_narrow_route_is_what_picks_the_tokens(model):
    head = _route(model, near=8, context=16)
    w = model.lm_head.weight.detach().numpy().astype(np.float64)
    want, prev = [], PROMPT[-1]
    with grilly.no_grad():
        cache = model.make_cache(1, len(PROMPT) + COUNT)
        x = model.hidden(grilly.tensor([PROMPT]), past_key_values=cache, logits_to_keep=1)
        for _ in range(COUNT):
            rows, _ = head.union(x, prev)
            tok = int(rows[np.argmax(w[rows] @ x.numpy().reshape(-1).astype(np.float64))])
            want.append(tok)
            prev = tok
            x = model.hidden(grilly.tensor([[tok]]), past_key_values=cache, logits_to_keep=1)
    routed = _generate(model, head)[0][len(PROMPT):]
    assert routed == want
    assert head.picks == COUNT
    assert routed != _generate(model, None)[0][len(PROMPT):]   # the route, not the full head, chose


def test_it_stops_at_eos(model):
    head = _route(model, near=VOCAB, context=VOCAB)
    full = _generate(model, None)[0][len(PROMPT):]
    eos = full[5]
    stop = full.index(eos) + 1
    routed = _generate(model, head, eos_token_id=eos)[0]
    assert routed == PROMPT + full[:stop]
    assert head.picks == stop


@pytest.mark.parametrize("kw", [
    dict(do_sample=True, temperature=0.8),
    dict(repetition_penalty=1.3),
    dict(logits_processor=lambda ids, scores: scores),
    dict(prompt=[PROMPT, [3, 1, 4, 1, 5, 9]]),
    dict(use_shortlist=False),
], ids=["sampling", "penalty", "processor", "batch", "switched-off"])
def test_the_full_head_runs_when_the_route_may_not(model, kw):
    head = _route(model, near=8, context=16)
    grilly.manual_seed(4)
    full = _generate(model, None, **kw)
    grilly.manual_seed(4)
    assert _generate(model, head, **kw) == full
    assert head.picks == 0


def test_a_model_without_hidden_says_so():
    class NoState(GenerationMixin):
        shortlist = object()
    with pytest.raises(TypeError, match="no hidden"):
        NoState()._shortlist_for(GenerationConfig(), 1, False, [])
