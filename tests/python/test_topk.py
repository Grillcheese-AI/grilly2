"""``topk`` for small k, against torch, exactly.

Reference: **torch** (case (b)) — but the bound is **0** for both values
and indices, and that is the point. grilly2 defines a total order for
sorting (descending puts NaN first, ties keep their original position),
and the small-k kernel has to give the same answer the bitonic sort gives,
not merely a similar one. A faster route to a different answer would be a
second dispatch path (CLAUDE.md §5) wearing a speedup as a disguise.

Why it exists: the bitonic path costs ``log2(n)*(log2(n)+1)/2`` dispatches
whatever k is — 108 for ``topk(8)`` over 16384 — because it sorts the whole
row to keep eight of it. A decode step asks for exactly that four times a
token, and a sampler asks for k<=64 once. The small-k kernel is one
dispatch, or two when a row is split across workgroups.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import grilly


def rand(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("n", [1, 63, 64, 65, 1024, 16384, 128000])
@pytest.mark.parametrize("k", [1, 8, 64])
def test_values_and_indices_match_torch_exactly(n, k):
    if k > n:
        pytest.skip("k > n")
    a = rand((n,), seed=n + k)
    g = grilly.from_numpy(a.copy()).topk(k)
    t = torch.from_numpy(a.copy()).topk(k)
    assert (g.values.numpy() == t.values.numpy()).all()
    assert (g.indices.numpy() == t.indices.numpy()).all()


@pytest.mark.parametrize("shape", [(4, 7), (3, 1000), (2, 3, 300), (64, 64), (200, 512)])
def test_rows_are_independent(shape):
    """Many rows take the one-chunk path, few rows get split — both have to
    land on torch's answer."""
    a = rand(shape, seed=len(shape))
    g = grilly.from_numpy(a.copy()).topk(5)
    t = torch.from_numpy(a.copy()).topk(5)
    assert (g.values.numpy() == t.values.numpy()).all()
    assert (g.indices.numpy() == t.indices.numpy()).all()


def test_largest_false_is_the_ascending_order():
    a = rand((5000,), seed=7)
    g = grilly.from_numpy(a.copy()).topk(16, largest=False)
    t = torch.from_numpy(a.copy()).topk(16, largest=False)
    assert (g.values.numpy() == t.values.numpy()).all()
    assert (g.indices.numpy() == t.indices.numpy()).all()


def test_ties_keep_their_original_position():
    """The half of the order a random input never exercises. Ties are where
    a merge that is not careful reorders silently."""
    a = np.zeros(5000, dtype=np.float32)
    a[::7] = 1.0
    g = grilly.from_numpy(a.copy()).topk(12)
    assert (g.indices.numpy() == np.arange(12) * 7).all()
    a2 = np.full(300, 2.0, dtype=np.float32)
    assert (grilly.from_numpy(a2).topk(9).indices.numpy() == np.arange(9)).all()


def test_nan_sorts_first_descending():
    """torch's NaN placement, and grilly's own tie rule on top of it.

    torch returns the NaNs in an arbitrary order — measured [1999, 5] for
    this input — because equal keys have no defined order there. grilly's
    sort defines one for every key including NaN, which is stricter, so
    this asserts grilly's and only compares the finite tail with torch."""
    a = rand((2000,), seed=11)
    a[[5, 1999]] = np.nan
    g = grilly.from_numpy(a.copy()).topk(4)
    t = torch.from_numpy(a.copy()).topk(4)
    assert np.isnan(g.values.numpy()[:2]).all()
    assert (g.indices.numpy()[:2] == [5, 1999]).all()     # ties by position
    assert (g.values.numpy()[2:] == t.values.numpy()[2:]).all()
    assert (g.indices.numpy()[2:] == t.indices.numpy()[2:]).all()


def test_a_k_past_the_fast_path_still_answers():
    """k > 64 falls through to the bitonic sort, and the fall-through has
    to be invisible."""
    a = rand((4096,), seed=13)
    for k in (64, 65, 200):
        g = grilly.from_numpy(a.copy()).topk(k)
        t = torch.from_numpy(a.copy()).topk(k)
        assert (g.values.numpy() == t.values.numpy()).all(), k
        assert (g.indices.numpy() == t.indices.numpy()).all(), k


def test_an_int_tensor_takes_the_other_path_and_still_matches():
    """int64 keys are not on the fast path — the shared lists hold one word
    an entry — so this is the fall-through, asserted so it stays correct.

    Values are compared exactly; indices are compared through the values
    they select, because this input has ties and torch orders those
    arbitrarily (measured: it returns 3557 before 1317 for two equal 996s,
    grilly returns them by position)."""
    a = np.random.default_rng(3).integers(-1000, 1000, 4000).astype(np.int64)
    g = grilly.from_numpy(a.copy()).topk(10)
    t = torch.from_numpy(a.copy()).topk(10)
    assert (g.values.numpy() == t.values.numpy()).all()
    assert (a[g.indices.numpy()] == t.values.numpy()).all()
    ties = np.flatnonzero(a == 996)
    assert list(g.indices.numpy()[a[g.indices.numpy()] == 996]) == sorted(ties)[:3]


def test_topk_inside_a_capture_replays_bit_exactly():
    """A decode step's episodic read is a topk, so it has to record."""
    from grilly import _C

    scores = grilly.from_numpy(rand((4096,), seed=17))
    with grilly.no_grad():
        ref = scores.topk(8).indices.numpy().copy()
        _C.begin_capture()
        out = scores.topk(8)
        graph = _C.end_capture()
        graph.replay()
    assert (out.indices.numpy() == ref).all()
