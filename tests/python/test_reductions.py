"""sum / mean / amax / amin through each reduction path, against float64.

grilly picks a path by shape (grilly_core::generic::reduce):
  rows      the reduced dim is last: a workgroup per row (g2-reduce-rows),
            long rows split into chunks and finished by a second pass;
  columns   a strided dim with >= 8192 outputs: a thread per output;
  moved     a strided dim with fewer outputs: copied to rows first.
Tolerance: |err| <= eps * sum(|x|) for sums (Kahan per thread, two-sum
across threads; measured at most 0.14 of it), mean the same divided by
the count; max / min are exact. Every path is deterministic: the same
input gives the same bits.
"""

import numpy as np
import pytest
import torch

import grilly

EPS = np.finfo(np.float32).eps
RNG = np.random.default_rng(5)

PATHS = [
    ("rows, short", (4096, 64), 1),
    ("rows, one per workgroup", (32768, 512), -1),
    ("rows, chunked", (3, 200_000), 1),
    ("rows, whole tensor", (5_000_000,), 0),
    ("columns", (300, 9000), 0),
    ("moved", (100_000, 4), 0),
    ("moved, middle dim", (16, 5000, 7), 1),
]


@pytest.mark.parametrize("name,shape,dim", PATHS, ids=[p[0] for p in PATHS])
def test_sum_and_mean_within_bound(name, shape, dim):
    x = RNG.standard_normal(shape).astype(np.float32)
    g = grilly.tensor(x)
    ref = x.astype(np.float64).sum(dim)
    bound = EPS * np.abs(x.astype(np.float64)).sum(dim)
    assert np.all(np.abs(g.sum(dim).numpy() - ref) <= bound)
    n = shape[dim]
    assert np.all(np.abs(g.mean(dim).numpy() - ref / n) <= bound / n + EPS * np.abs(ref / n))
    for fn in ("amax", "amin"):
        np.testing.assert_array_equal(getattr(g, fn)(dim).numpy(), getattr(x, "max" if fn == "amax" else "min")(dim))


def test_full_sum_of_a_large_tensor():
    """sum() of 16M elements: one thread did it before, past the fence timeout."""
    x = RNG.standard_normal(16_777_216).astype(np.float32)
    got = grilly.tensor(x).sum().item()
    assert abs(got - x.astype(np.float64).sum()) <= EPS * np.abs(x.astype(np.float64)).sum()


@pytest.mark.parametrize("shape,dim", [((5, 200_000), 1), ((64, 512), 1), ((100_000, 4), 0)])
def test_special_values_in_every_path_match_torch(shape, dim):
    """Output line 0 gets an inf, 1 a NaN, 2 a -inf, 3 all +inf (inf + inf
    must stay inf, not become NaN through the compensation)."""
    x = RNG.standard_normal(shape).astype(np.float32)
    line = (lambda r, pos: (r, pos)) if dim == 1 else (lambda r, pos: (pos, r))
    mid = shape[dim] // 3
    x[line(0, mid)], x[line(1, mid)], x[line(2, mid)] = np.inf, np.nan, -np.inf
    x[line(3, slice(None))] = np.inf
    g, t = grilly.tensor(x), torch.tensor(x)
    for fn in ("sum", "mean", "amax", "amin"):
        got, want = getattr(g, fn)(dim).numpy(), getattr(t, fn)(dim).numpy()
        special = ~np.isfinite(want)
        assert special[:4].sum() >= 3
        np.testing.assert_array_equal(got[special], want[special])


@pytest.mark.parametrize("shape,dim", [((3, 200_000), 1), ((32768, 512), 1), ((100_000, 4), 0)])
def test_reductions_are_bitwise_repeatable(shape, dim):
    g = grilly.tensor(RNG.standard_normal(shape).astype(np.float32))
    runs = {g.sum(dim).numpy().tobytes() for _ in range(3)}
    assert len(runs) == 1


def test_multi_dim_reductions_match_torch():
    x = RNG.standard_normal((6, 7, 8, 9)).astype(np.float32)
    g, t = grilly.tensor(x), torch.tensor(x)
    for dims in ((2, 3), (1, 2, 3), (0, 1), (0, 2), (0, 1, 2, 3), (1, 3)):
        for keepdim in (False, True):
            got, want = g.sum(dims, keepdim=keepdim), t.sum(dims, keepdim=keepdim)
            assert tuple(got.shape) == tuple(want.shape)
            bound = EPS * np.abs(x.astype(np.float64)).sum(dims, keepdims=keepdim)
            assert np.all(np.abs(got.numpy() - x.astype(np.float64).sum(dims, keepdims=keepdim)) <= 2 * bound)
            np.testing.assert_array_equal(g.amax(dims, keepdim=keepdim).numpy(), t.amax(dims, keepdim=keepdim).numpy())


def test_empty_reductions_match_torch():
    for shape, dim in (((0, 5), 0), ((5, 0), 1), ((0, 5), 1)):
        g, t = grilly.zeros(*shape), torch.zeros(*shape)
        np.testing.assert_array_equal(g.sum(dim).numpy(), t.sum(dim).numpy())
        np.testing.assert_array_equal(g.mean(dim).numpy(), t.mean(dim).numpy())
