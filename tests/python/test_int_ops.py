"""Integer and bool arithmetic, promotion, comparisons, logic and selection
against torch. Integer results are exact; float results use rtol 1e-6."""

import numpy as np
import pytest
import torch

import grilly

DT = {grilly.float32: torch.float32, grilly.int64: torch.int64, grilly.int32: torch.int32, grilly.bool: torch.bool}


def same(g, t):
    assert DT[g.dtype] == t.dtype, (g.dtype, t.dtype)
    assert tuple(g.shape) == tuple(t.shape)
    if t.dtype.is_floating_point:
        np.testing.assert_allclose(g.numpy(), t.numpy(), rtol=1e-6, atol=1e-7)
    else:
        np.testing.assert_array_equal(g.numpy(), t.numpy())


def pair(values, dtype):
    return grilly.tensor(values, dtype=getattr(grilly, dtype)), torch.tensor(values, dtype=getattr(torch, dtype))


A = [[3, -7, 2**40], [0, 5, -1]]
B = [4, -2, 3]


@pytest.mark.parametrize("expr", [
    lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b,
    lambda a, b: a + 5, lambda a, b: 10 - a, lambda a, b: a * -3,
    lambda a, b: a / b, lambda a, b: a / 2, lambda a, b: 7 / b,
    lambda a, b: a * 0.5, lambda a, b: a + 1.25,
    lambda a, b: a.float() + b, lambda a, b: -a, lambda a, b: a.abs(), lambda a, b: a.sign(),
    lambda a, b: a.clamp(-2, 3), lambda a, b: a.sum(), lambda a, b: a.sum(0), lambda a, b: a.sum(1, keepdim=True),
    lambda a, b: a.amax(1), lambda a, b: a.amin(), lambda a, b: a.max(),
])
@pytest.mark.parametrize("dtype", ["int64", "int32"])
def test_integer_arithmetic_matches_torch(expr, dtype):
    (ga, ta), (gb, tb) = pair(A if dtype == "int64" else [[3, -7, 9], [0, 5, -1]], dtype), pair(B, dtype)
    same(expr(ga, gb), expr(ta, tb))


def test_promotion_matches_torch_result_type():
    g = {n: grilly.tensor([1, 2], dtype=getattr(grilly, n)) for n in DT_NAMES}
    t = {n: torch.tensor([1, 2], dtype=getattr(torch, n)) for n in DT_NAMES}
    g0 = {n: grilly.tensor(1, dtype=getattr(grilly, n)) for n in DT_NAMES}
    t0 = {n: torch.tensor(1, dtype=getattr(torch, n)) for n in DT_NAMES}
    for x in DT_NAMES:
        for y in DT_NAMES:
            assert DT[grilly.result_type(g[x], g[y])] == torch.result_type(t[x], t[y]), (x, y)
            assert DT[grilly.result_type(g[x], g0[y])] == torch.result_type(t[x], t0[y]), (x, "0-d", y)
        for num in (2, 2.5, True):
            assert DT[grilly.result_type(g[x], num)] == torch.result_type(t[x], num), (x, num)


DT_NAMES = ["float32", "int64", "int32", "bool"]


@pytest.mark.parametrize("op", ["eq", "ne", "gt", "ge", "lt", "le"])
def test_comparisons_return_bool_like_torch(op):
    fa = [[1.5, float("nan"), -2.0], [0.0, 3.0, float("inf")]]
    (ga, ta) = pair(fa, "float32")
    (gi, ti) = pair([[1, 2, 3], [4, 5, 6]], "int64")
    for g, t, other in [(ga, ta, 0.0), (gi, ti, 3), (gi, ti, 2.5)]:
        same(getattr(g, op)(other), getattr(t, op)(other))
    same(getattr(grilly, op)(gi, ga), getattr(torch, op)(ti, ta))
    same(gi < gi[:, :1], ti < ti[:, :1])


def test_logical_and_bitwise_ops_match_torch():
    (gm, tm), (gn, tn) = pair([True, False, True, False], "bool"), pair([True, True, False, False], "bool")
    for f in (lambda a, b: a & b, lambda a, b: a | b, lambda a, b: a ^ b, lambda a, b: ~a,
              lambda a, b: a.logical_and(b), lambda a, b: a.logical_or(b), lambda a, b: a.logical_not()):
        same(f(gm, gn), f(tm, tn))
    (gi, ti) = pair([5, -3, 12, 0], "int64")
    for f in (lambda a: a & 6, lambda a: a | 1, lambda a: a ^ -1, lambda a: ~a, lambda a: a.logical_not()):
        same(f(gi), f(ti))
    same((gm & gn).sum(), (tm & tn).sum())        # bools sum to int64


def test_where_and_masked_fill_match_torch():
    (gx, tx) = pair([[1.0, -2.0, 3.0], [-4.0, 5.0, float("inf")]], "float32")
    same(grilly.where(gx > 0, gx, 0.0), torch.where(tx > 0, tx, 0.0))
    same(gx.masked_fill(gx < 0, -1e9), tx.masked_fill(tx < 0, -1e9))
    (gi, ti) = pair([[1, 2, 3], [4, 5, 6]], "int64")
    same(grilly.where(gi > 2, gi, gi * 10), torch.where(ti > 2, ti, ti * 10))
    same(grilly.where(gi[:1] > 2, gx, 7), torch.where(ti[:1] > 2, tx, 7))      # broadcast + promotion
    for lib, x in ((grilly, gi), (torch, ti)):
        with pytest.raises(RuntimeError, match="boolean"):
            lib.where(x, x, x)


def test_where_gradient_goes_to_the_taken_branch():
    a = np.array([1.0, -2.0, 3.0], np.float32)
    for lib in (grilly, torch):
        x = lib.tensor(a, requires_grad=True)
        y = lib.tensor(a * 10, requires_grad=True)
        lib.where(x > 0, x, y).sum().backward()
        if lib is grilly:
            got = (x.grad.numpy(), y.grad.numpy())
        else:
            want = (x.grad.numpy(), y.grad.numpy())
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_predicates_all_any_equal_match_torch():
    (gx, tx) = pair([1.0, float("nan"), float("inf"), -float("inf"), 0.0], "float32")
    for f in ("isfinite", "isinf", "isnan"):
        same(getattr(grilly, f)(gx), getattr(torch, f)(tx))
    (gi, ti) = pair([1, 2], "int64")
    same(grilly.isfinite(gi), torch.isfinite(ti))
    (gb, tb) = pair([[True, False], [True, True]], "bool")
    same(gb.all(), tb.all())
    same(gb.any(1), tb.any(1))
    same(grilly.all(gb, 0), torch.all(tb, 0))
    assert grilly.equal(gi, gi.clone()) == torch.equal(ti, ti.clone()) is True
    assert grilly.equal(gi, gi + 1) == torch.equal(ti, ti + 1) is False


def test_outer_matches_torch():
    (ga, ta), (gb, tb) = pair([1, 2, 3], "int64"), pair([0.5, -1.0], "float32")
    same(grilly.outer(ga, gb), torch.outer(ta, tb))
    same(grilly.outer(ga, ga), torch.outer(ta, ta))


@pytest.mark.parametrize("call,exc", [
    (lambda L: L.tensor([True]) - L.tensor([True]), RuntimeError),
    (lambda L: -L.tensor([True]), RuntimeError),
    (lambda L: L.tensor([1, 2]).mean(), RuntimeError),
    (lambda L: L.tensor([1.0]) & L.tensor([1.0]), (TypeError, RuntimeError)),
])
def test_errors_match_torch(call, exc):
    with pytest.raises(exc):
        call(torch)
    with pytest.raises(exc):
        call(grilly)


@pytest.mark.parametrize("op", ["floor_divide", "remainder", "trunc"])
@pytest.mark.parametrize("dtype", ["int64", "int32"])
def test_integer_division_matches_torch(op, dtype):
    """Exact, with torch's signs: floor division rounds down, the remainder
    takes the divisor's sign, trunc division rounds toward zero."""
    big = 2**62 if dtype == "int64" else 2**30
    a = [7, -7, 7, -7, 0, big, -big + 1, 5, 6, -6]
    b = [2, 2, -2, -2, 3, 7, 3, -1, 3, 4]
    ga, gb = pair(a, dtype)[0], pair(b, dtype)[0]
    ta, tb = pair(a, dtype)[1], pair(b, dtype)[1]
    if op == "trunc":
        same(grilly.div(ga, gb, rounding_mode="trunc"), torch.div(ta, tb, rounding_mode="trunc"))
        return
    same(getattr(grilly, op)(ga, gb), getattr(torch, op)(ta, tb))
    sym = "//" if op == "floor_divide" else "%"
    same(eval(f"ga {sym} gb"), eval(f"ta {sym} tb"))
    same(eval(f"ga {sym} 3"), eval(f"ta {sym} 3"))
    same(eval(f"17 {sym} gb"), eval(f"17 {sym} tb"))
    same(grilly.div(ga, gb, rounding_mode="floor"), torch.div(ta, tb, rounding_mode="floor"))


def test_integer_division_by_zero_raises_like_torch():
    for lib in (torch, grilly):
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            lib.tensor([1, 2]) % lib.tensor([1, 0])
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            lib.tensor([1, 2]) // 0


def test_float_floor_division_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        grilly.tensor([1.5]) % 1.0
