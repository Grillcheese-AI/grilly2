"""argmax / argmin, max / min along a dim, sort, argsort and topk against
torch. Indices are exact; values are gathered copies, so they are exact too.
Gradients route to the chosen elements and are exact.

Ties: grilly always takes the lower index first (the order is total, so
every result is deterministic). torch CPU does the same for argmax/argmin
and max(dim), and for sort(stable=True); tie tests compare against those."""

import numpy as np
import pytest
import torch

import grilly

RNG = np.random.default_rng(11)
X = RNG.standard_normal((4, 7, 5)).astype(np.float32)


def same(g, t):
    assert tuple(g.shape) == tuple(t.shape)
    np.testing.assert_array_equal(g.detach().numpy(), t.detach().numpy())


@pytest.mark.parametrize("fn", ["argmax", "argmin"])
@pytest.mark.parametrize("dim,keepdim", [(None, False), (None, True), (0, False), (1, True), (-1, False)])
def test_arg_extremes_match_torch(fn, dim, keepdim):
    g, t = grilly.tensor(X), torch.tensor(X)
    got = getattr(g, fn)(dim=dim, keepdim=keepdim)
    assert got.dtype is grilly.int64
    same(got, getattr(t, fn)(dim=dim, keepdim=keepdim))
    same(getattr(grilly, fn)(g, dim, keepdim), getattr(torch, fn)(t, dim, keepdim))


@pytest.mark.parametrize("values", [
    [[1.0, 3.0, 3.0, 2.0], [5.0, 5.0, 5.0, 5.0]],             # ties: first wins
    [[1.0, float("nan"), 7.0, float("nan")], [-1.0, -2.0, -2.0, -1.0]],  # NaN is the extreme
    [[float("-inf"), float("-inf"), 0.0, float("inf")], [0.0, -0.0, 0.0, -0.0]],
])
@pytest.mark.parametrize("is_max", [True, False])
def test_ties_and_nan_match_torch(values, is_max):
    g, t = grilly.tensor(values), torch.tensor(values)
    name = "max" if is_max else "min"
    r, rt = getattr(g, name)(1), getattr(t, name)(1)
    same(r.indices, rt.indices)
    same(r.values, rt.values)
    same(getattr(g, "arg" + name)(1), getattr(t, "arg" + name)(1))


@pytest.mark.parametrize("dtype", ["int64", "int32", "bool"])
def test_integer_extremes_match_torch(dtype):
    v = RNG.integers(-3, 3, (5, 6))
    g, t = grilly.tensor(v).to(getattr(grilly, dtype)), torch.tensor(v).to(getattr(torch, dtype))
    for name in ("max", "min"):
        r, rt = getattr(g, name)(dim=1), getattr(t, name)(dim=1)
        same(r.indices, rt.indices)
        assert r.values.dtype is g.dtype
        same(r.values, rt.values)
    if dtype != "bool":
        same(g.argmax(0), t.argmax(0))


def test_max_dim_returns_named_pair_and_gradient():
    g, t = grilly.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    r, rt = grilly.max(g, dim=-1, keepdim=True), torch.max(t, dim=-1, keepdim=True)
    values, indices = r
    assert values is r.values and indices is r[1]
    same(values, rt.values)
    same(indices, rt.indices)
    w = RNG.standard_normal(tuple(rt.values.shape)).astype(np.float32)
    (values * grilly.tensor(w)).sum().backward()
    (rt.values * torch.tensor(w)).sum().backward()
    np.testing.assert_array_equal(g.grad.numpy(), t.grad.numpy())
    assert "Max" in type(values.grad_fn).__name__ or "Max" in values.grad_fn.name()


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_matches_torch(dim, descending):
    g, t = grilly.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    r, rt = g.sort(dim=dim, descending=descending), t.sort(dim=dim, descending=descending)
    same(r.indices, rt.indices)
    same(r.values, rt.values)
    same(grilly.argsort(g, dim, descending), torch.argsort(t, dim, descending))
    w = RNG.standard_normal(X.shape).astype(np.float32)
    (r.values * grilly.tensor(w)).sum().backward()
    (rt.values * torch.tensor(w)).sum().backward()
    np.testing.assert_array_equal(g.grad.numpy(), t.grad.numpy())


@pytest.mark.parametrize("descending", [False, True])
def test_sort_ties_and_nan_match_stable_torch(descending):
    nan = float("nan")
    v = [[3.0, 1.0, 3.0, nan, 1.0, -0.0, 0.0, nan, float("inf"), 3.0]]
    r = grilly.sort(grilly.tensor(v), descending=descending)
    rt = torch.sort(torch.tensor(v), descending=descending, stable=True)
    same(r.indices, rt.indices)
    same(r.values, rt.values)


@pytest.mark.parametrize("k,dim,largest", [(1, -1, True), (3, 1, True), (3, 1, False), (5, -1, False), (0, 0, True), (4, 0, True)])
def test_topk_matches_torch(k, dim, largest):
    g, t = grilly.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)
    r, rt = g.topk(k, dim=dim, largest=largest), t.topk(k, dim=dim, largest=largest)
    same(r.indices, rt.indices)
    same(r.values, rt.values)
    if k:
        w = RNG.standard_normal(tuple(rt.values.shape)).astype(np.float32)
        (r.values * grilly.tensor(w)).sum().backward()
        (rt.values * torch.tensor(w)).sum().backward()
        np.testing.assert_array_equal(g.grad.numpy(), t.grad.numpy())


@pytest.mark.parametrize("n", [1, 2, 3, 1000, 32768, 40001])
def test_long_rows_match_torch(n):
    """Rows padded to a power of two, including exact powers and odd sizes.
    32768 float32 normal samples contain ~15 equal pairs, which torch's
    default (unstable) sort may order either way, so compare to stable=True."""
    v = RNG.standard_normal((3, n)).astype(np.float32)
    k = min(10, n)
    same(grilly.tensor(v).topk(k).indices, torch.tensor(v).topk(k).indices)
    for descending in (False, True):
        same(grilly.tensor(v).sort(descending=descending).indices,
             torch.tensor(v).sort(descending=descending, stable=True).indices)


def test_zero_dim_and_empty():
    g0, t0 = grilly.tensor(2.5), torch.tensor(2.5)
    same(g0.max(0).values, t0.max(0).values)
    same(g0.max(0).indices, t0.max(0).indices)
    same(g0.argmax(), t0.argmax())
    same(g0.sort().indices, t0.sort().indices)
    same(g0.topk(1).values, t0.topk(1).values)
    ge, te = grilly.zeros(0, 3), torch.zeros(0, 3)
    same(ge.sort(0).indices, te.sort(0).indices)
    same(ge.max(1).values, te.max(1).values)


def test_errors_match_torch():
    g, t = grilly.tensor(X), torch.tensor(X)
    for lib, x in ((torch, t), (grilly, g)):
        with pytest.raises(RuntimeError, match="selected index k out of range"):
            x.topk(6)
        with pytest.raises(IndexError):
            x.max(3)
        with pytest.raises(IndexError, match=r"argmax\(\): Expected reduction dim 1 to have non-zero size"):
            lib.zeros(2, 0).argmax(1)
        with pytest.raises(IndexError, match=r"min\(\): Expected reduction dim 1 to have non-zero size"):
            lib.zeros(2, 0).min(1)
        with pytest.raises(IndexError, match="Expected reduction dim to be specified"):
            lib.zeros(0).argmin()


@pytest.mark.parametrize("dtype", ["int64", "int32", "bool"])
@pytest.mark.parametrize("descending", [False, True])
def test_integer_sort_and_topk_match_torch(dtype, descending):
    v = RNG.integers(-2**40, 2**40, (6, 300))
    v[:, ::7] = 5  # ties
    g, t = grilly.tensor(v).to(getattr(grilly, dtype)), torch.tensor(v).to(getattr(torch, dtype))
    r, rt = g.sort(descending=descending), t.sort(descending=descending, stable=True)
    assert r.values.dtype is g.dtype
    same(r.indices, rt.indices)
    same(r.values, rt.values)
    if dtype != "bool":  # torch has no topk for bool
        # distinct keys: torch.topk does not promise an order among ties
        u = torch.randperm(600, generator=torch.Generator().manual_seed(0)).reshape(2, 300).numpy() - 300
        gu, tu = grilly.tensor(u).to(g.dtype), torch.tensor(u).to(t.dtype)
        same(gu.topk(9, largest=descending).indices, tu.topk(9, largest=descending).indices)
