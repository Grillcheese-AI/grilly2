"""grilly.infer.shortlist.ShortlistHead against a numpy implementation of the
same route.

Tolerance: exact. The union is integers (distances, first positions,
counts), so it must equal numpy's row for row; the token is numpy's float64
argmax over that union, on random uniform data where the top two dot
products are far apart next to fp32 rounding. The Rust side has the wider
cases (ties at the k-th distance, a short context list, a 4,096-wide sort):
crates/grilly-infer/tests/shortlist.rs.
"""

import numpy as np

import grilly
from grilly.infer.shortlist import ShortlistHead


def _problem(rng, vocab=3000, dim=256, clusters=24, max_len=300, width=3):
    w = rng.uniform(-1, 1, (vocab, dim)).astype(np.float32)
    cent = rng.uniform(-1, 1, (clusters, dim)).astype(np.float32)
    lists = [rng.choice(vocab, size=int(rng.integers(1, max_len)), replace=False) for _ in range(clusters)]
    offsets = np.concatenate([[0], np.cumsum([len(l) for l in lists])]).astype(np.int64)
    ids = np.concatenate(lists).astype(np.int64)
    succ = rng.integers(-1, vocab, (vocab, width))
    return w, cent, ids, offsets, succ


def _reference(x, prev, w, cent, ids, offsets, succ, near, context):
    order = np.argsort(-(cent.astype(np.float64) @ x.astype(np.float64)), kind="stable")
    seen, ctx = set(), []
    for c in order:
        for t in ids[offsets[c]:offsets[c + 1]]:
            if t not in seen:
                seen.add(t)
                ctx.append(t)
            if len(ctx) == context:
                break
        if len(ctx) == context:
            break
    dist = np.count_nonzero((w > 0) != (x > 0), axis=1)
    nearest = np.argsort(dist, kind="stable")[:near]
    extra = succ[prev][succ[prev] >= 0] if prev is not None else []
    rows = np.unique(np.concatenate([np.asarray(ctx, np.int64), nearest, np.asarray(extra, np.int64)]))
    scores = w[rows].astype(np.float64) @ x.astype(np.float64)
    return rows, int(rows[np.argmax(scores)])


def test_union_and_token_match_numpy():
    rng = np.random.default_rng(3)
    w, cent, ids, offsets, succ = _problem(rng)
    head = ShortlistHead(grilly.from_numpy(w), cent, ids, offsets, succ, near=200, context=700, ranks=24)
    assert head.shape == w.shape
    for i in range(12):
        x = rng.uniform(-1, 1, w.shape[1]).astype(np.float32)
        prev = int(rng.integers(0, w.shape[0])) if i % 2 else None
        rows, token = _reference(x, prev, w, cent, ids, offsets, succ, 200, 700)
        got, short = head.union(grilly.from_numpy(x), prev)
        assert not short
        np.testing.assert_array_equal(got, rows)
        assert head.argmax(grilly.from_numpy(x.reshape(1, 1, -1)), prev) == token
        assert head.last.rows == len(rows) and not head.last.short


def test_codes_are_numpy_packbits():
    rng = np.random.default_rng(5)
    w = rng.standard_normal((50, 128)).astype(np.float32)
    w[0, :7] = 0.0
    w[1, 3] = -0.0
    head = ShortlistHead(grilly.from_numpy(w), w[:2], np.array([0, 1]), np.array([0, 1, 2]), near=4, context=2)
    want = np.packbits(w > 0, axis=1, bitorder="little").view(np.int32)
    np.testing.assert_array_equal(head.codes().numpy(), want)
