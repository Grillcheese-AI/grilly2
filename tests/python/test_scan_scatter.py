"""cumsum, logcumsumexp, scatter and scatter_ against torch.

Tolerances (measured against a float64 reference, then doubled):
- cumsum float32: |err| <= eps * cumsum(|x|) (compensated sum; measured
  max 0.5, torch's CPU cumsum 0.39). Integer cumsum is exact.
- logcumsumexp: |err| <= 2 eps (|running max| + log(i+1) + 1) (measured max
  1.04 at length 8192). Near a result of 0 that is more ULPs than torch's
  CPU (which accumulates in double), but the absolute error stays eps-sized.
- cumsum gradients: the same bound (a cumsum from the end).
- logcumsumexp gradients: rtol 1e-5 against torch.
- minGRU's log-domain scan: within 2x of torch float32's own error
  against float64 (both lose ~1e-5 on h).
- scatter copies values and gathers gradients: exact.
"""

import numpy as np
import pytest
import torch

import grilly

EPS = np.finfo(np.float32).eps
RNG = np.random.default_rng(3)


def grads_match(fg, ft, x, rtol=1e-5, atol=1e-6):
    a, t = grilly.tensor(x, requires_grad=True), torch.tensor(x, requires_grad=True)
    yg, yt = fg(a), ft(t)
    w = RNG.standard_normal(tuple(yt.shape)).astype(np.float32)
    (yg * grilly.tensor(w)).sum().backward()
    (yt * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), t.grad.numpy(), rtol=rtol, atol=atol)


# ── cumsum ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape,dim", [((7,), 0), ((3, 50, 4), 1), ((3, 50, 4), -1), ((4, 4096), 1), ((2, 0, 3), 1)])
def test_cumsum_within_bound(shape, dim):
    x = RNG.standard_normal(shape).astype(np.float32)
    got = grilly.cumsum(grilly.tensor(x), dim)
    assert got.dtype is grilly.float32 and tuple(got.shape) == shape
    ref = np.cumsum(x.astype(np.float64), axis=dim)
    bound = EPS * np.cumsum(np.abs(x.astype(np.float64)), axis=dim)
    assert np.all(np.abs(got.numpy() - ref) <= bound + 1e-30)
    # the gradient is the cumsum of the output gradient from the end: same bound
    a = grilly.tensor(x, requires_grad=True)
    w = RNG.standard_normal(shape).astype(np.float64)
    (a.cumsum(dim) * grilly.tensor(w.astype(np.float32))).sum().backward()
    flip = lambda v: np.flip(v, axis=dim)  # noqa: E731
    w32 = w.astype(np.float32).astype(np.float64)
    ref = flip(np.cumsum(flip(w32), axis=dim))
    bound = EPS * flip(np.cumsum(flip(np.abs(w32)), axis=dim))
    assert np.all(np.abs(a.grad.numpy() - ref) <= bound + 1e-30)


@pytest.mark.parametrize("dtype", ["int64", "int32", "bool"])
def test_integer_cumsum_is_exact_int64(dtype):
    v = RNG.integers(-2**40, 2**40, (3, 17)) if dtype == "int64" else RNG.integers(-5, 5, (3, 17))
    g = grilly.tensor(v).to(getattr(grilly, dtype))
    t = torch.tensor(v).to(getattr(torch, dtype))
    got, want = g.cumsum(1), t.cumsum(1)
    assert got.dtype is grilly.int64 and want.dtype is torch.int64
    np.testing.assert_array_equal(got.numpy(), want.numpy())
    as_float = g.cumsum(0, dtype=grilly.float32)
    assert as_float.dtype is grilly.float32
    f64 = t.to(torch.float32).numpy().astype(np.float64)
    assert np.all(np.abs(as_float.numpy() - np.cumsum(f64, axis=0)) <= EPS * np.cumsum(np.abs(f64), axis=0))


def test_cumsum_zero_dim():
    assert grilly.tensor(2.5).cumsum(0).item() == torch.tensor(2.5).cumsum(0).item()
    with pytest.raises(IndexError):
        grilly.tensor(2.5).cumsum(1)


# ── logcumsumexp ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,scale", [(1, 1.0), (50, 1.0), (2000, 1.0), (2000, 30.0), (8192, 5.0), (8192, 0.01)])
def test_logcumsumexp_within_bound(n, scale):
    x = (RNG.standard_normal((4, n)) * scale).astype(np.float32)
    got = grilly.logcumsumexp(grilly.tensor(x), 1).numpy().astype(np.float64)
    x64 = x.astype(np.float64)
    ref = np.logaddexp.accumulate(x64, axis=1)
    bound = 2 * EPS * (np.abs(np.maximum.accumulate(x64, axis=1)) + np.log(np.arange(1, n + 1)) + 1)
    assert np.all(np.abs(got - ref) <= bound), np.max(np.abs(got - ref) / bound)


def test_logcumsumexp_special_values_match_torch():
    inf, nan = float("inf"), float("nan")
    x = [[-inf, -inf, 0.0, inf, 1.0], [1.0, nan, 2.0, 3.0, 4.0], [-inf] * 5, [1000.0, 1000.0, -1000.0, 1e30, 2.0]]
    np.testing.assert_array_equal(grilly.tensor(x).logcumsumexp(1).numpy(), torch.tensor(x).logcumsumexp(1).numpy())


@pytest.mark.parametrize("shape,dim", [((3, 50, 4), 1), ((5, 9), 0), ((6, 1), 1)])
def test_logcumsumexp_gradient_matches_torch(shape, dim):
    x = RNG.standard_normal(shape).astype(np.float32)
    grads_match(lambda a: a.logcumsumexp(dim), lambda t: t.logcumsumexp(dim), x)


def test_logcumsumexp_rejects_integers_like_torch():
    for lib in (torch, grilly):
        with pytest.raises(RuntimeError, match="not implemented for 'Long'"):
            lib.tensor([1, 2]).logcumsumexp(0)


def test_mingru_log_domain_scan_matches_torch():
    """CubbyLLM's minGRU parallel scan (Heinsen 2023), end to end with grads.
    The log-domain scan loses precision in float32 whoever runs it (torch
    float32 is ~1e-5 off float64 on h, ~1e-4 on da), so grilly is held to
    within 2x of torch float32's own error against a float64 reference."""
    def scan(lib, x_scan, a):
        log_a = a.clamp(min=1e-8).log()
        a_star = log_a.cumsum(dim=1)
        eps = 1e-30
        hp = (a_star + ((x_scan.clamp(min=0) + eps).log() - a_star).logcumsumexp(dim=1)).exp()
        hn = (a_star + (((-x_scan).clamp(min=0) + eps).log() - a_star).logcumsumexp(dim=1)).exp()
        return hp - hn

    xs = RNG.standard_normal((2, 64, 8)).astype(np.float32)
    av = RNG.uniform(0.2, 0.99, (2, 64, 8)).astype(np.float32)

    def run(lib, make):
        x, a = make(xs), make(av)
        h = scan(lib, x, a)
        h.sum().backward()
        return [v.astype(np.float64) for v in (h.detach().numpy(), x.grad.numpy(), a.grad.numpy())]

    got = run(grilly, lambda v: grilly.tensor(v, requires_grad=True))
    t32 = run(torch, lambda v: torch.tensor(v, requires_grad=True))
    ref = run(torch, lambda v: torch.tensor(v, dtype=torch.float64, requires_grad=True))
    for g, t, r in zip(got, t32, ref):
        assert np.abs(g - r).max() <= 2 * np.abs(t - r).max() + 1e-7


# ── scatter ──────────────────────────────────────────────────────────────

def distinct_index(shape, dim, size):
    """Indices without duplicates along dim (torch leaves duplicates unspecified)."""
    moved = list(shape)
    moved[dim], moved[-1] = moved[-1], moved[dim]
    rows = int(np.prod(moved[:-1]))
    idx = np.stack([RNG.permutation(size)[:moved[-1]] for _ in range(rows)]).reshape(moved)
    return np.swapaxes(idx, dim, -1)


@pytest.mark.parametrize("dst_shape,src_shape,index_shape,dim", [
    ((4, 6), (4, 6), (4, 3), 1),
    ((5, 3, 4), (3, 3, 4), (2, 3, 4), 0),
    ((3, 7), (4, 9), (2, 5), 1),  # src larger than the index
    ((3, 7), (3, 7), (3, 7), -1),
])
def test_scatter_matches_torch(dst_shape, src_shape, index_shape, dim):
    d = dim % len(dst_shape)
    x = RNG.standard_normal(dst_shape).astype(np.float32)
    src = RNG.standard_normal(src_shape).astype(np.float32)
    idx = distinct_index(index_shape, d, dst_shape[d])
    got = grilly.scatter(grilly.tensor(x), dim, grilly.tensor(idx), grilly.tensor(src))
    want = torch.scatter(torch.tensor(x), dim, torch.tensor(idx), torch.tensor(src))
    np.testing.assert_array_equal(got.numpy(), want.numpy())
    # Both inputs are differentiable. torch's src gradient is
    # grad.gather(dim, index), which raises when src is larger than the
    # index; grilly zero-pads it, so torch's reference uses the part of src
    # the index covers, padded with zeros.
    a, s = grilly.tensor(x, requires_grad=True), grilly.tensor(src, requires_grad=True)
    covered = src[tuple(slice(0, n) for n in index_shape)]
    ta, ts = torch.tensor(x, requires_grad=True), torch.tensor(covered, requires_grad=True)
    w = RNG.standard_normal(dst_shape).astype(np.float32)
    (a.scatter(dim, grilly.tensor(idx), s) * grilly.tensor(w)).sum().backward()
    (ta.scatter(dim, torch.tensor(idx), ts) * torch.tensor(w)).sum().backward()
    np.testing.assert_array_equal(a.grad.numpy(), ta.grad.numpy())
    want = np.zeros(src_shape, np.float32)
    want[tuple(slice(0, n) for n in index_shape)] = ts.grad.numpy()
    np.testing.assert_array_equal(s.grad.numpy(), want)


def test_scatter_inplace_and_scalar_value_match_torch():
    x = RNG.standard_normal((3, 8)).astype(np.float32)
    idx = distinct_index((3, 4), 1, 8)
    g, t = grilly.tensor(x), torch.tensor(x)
    assert g.scatter_(1, grilly.tensor(idx), -2.5) is g
    t.scatter_(1, torch.tensor(idx), -2.5)
    np.testing.assert_array_equal(g.numpy(), t.numpy())
    gi, ti = grilly.zeros(2, 5, dtype=grilly.int64), torch.zeros(2, 5, dtype=torch.int64)
    v = RNG.integers(-2**40, 2**40, (2, 5))
    i5 = distinct_index((2, 5), 1, 5)
    gi.scatter_(1, grilly.tensor(i5), grilly.tensor(v))
    ti.scatter_(1, torch.tensor(i5), torch.tensor(v))
    np.testing.assert_array_equal(gi.numpy(), ti.numpy())


def test_inplace_scatter_carries_the_gradient_of_src():
    """CubbyLLM's retrieval head: keep only the top-k logits, -inf elsewhere."""
    sims = RNG.standard_normal((4, 12)).astype(np.float32)

    def head(lib, s):
        top = s.topk(5, dim=-1)
        out = lib.full_like(s, float("-inf"))
        out.scatter_(-1, top.indices, top.values)
        return out

    g, t = grilly.tensor(sims, requires_grad=True), torch.tensor(sims, requires_grad=True)
    og, ot = head(grilly, g), head(torch, t)
    np.testing.assert_array_equal(og.detach().numpy(), ot.detach().numpy())
    assert og.requires_grad and not og.is_leaf
    w = np.arange(12, dtype=np.float32)
    (og.softmax(-1) * grilly.tensor(w)).sum().backward()
    (ot.softmax(-1) * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(g.grad.numpy(), t.grad.numpy(), rtol=1e-6, atol=1e-7)


def test_scatter_errors_match_torch():
    for lib in (torch, grilly):
        x = lib.zeros(3, 4)
        with pytest.raises(RuntimeError, match="out of bounds"):
            # grilly checks index bounds on the device and raises at the
            # next read, as torch does on CUDA; torch's CPU tensors have
            # raised before the synchronize is reached
            x.scatter(1, lib.tensor([[4]]), lib.ones(1, 1))
            grilly._C.synchronize()
        with pytest.raises(RuntimeError, match="same number of dimensions"):
            x.scatter(1, lib.tensor([1]), lib.ones(1))
        with pytest.raises(RuntimeError, match="self.dtype to be equal to src.dtype"):
            x.scatter(1, lib.tensor([[1]]), lib.ones(1, 1, dtype=lib.int64))
        with pytest.raises(RuntimeError, match="to be no larger than self"):
            x.scatter(1, lib.tensor([[0, 1, 2]]), lib.ones(1, 2))
        leaf = lib.zeros(3, 4, requires_grad=True)
        with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
            leaf.scatter_(1, lib.tensor([[1]]), 1.0)
