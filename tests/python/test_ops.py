"""torch's calling conventions for tensor operations, against real torch.

The math and gradients are checked against float64 torch in the Rust tests
(crates/grilly-torch/tests/ops_autograd.rs). These check the Python surface:
Python numbers on either side, negative dims, keepdim, dim tuples, indexing
forms, in-place writes, shape helpers, and gradients flowing through them.
Tolerance: rtol 1e-5, atol 1e-6 (well above the measured op error).
"""

import numpy as np
import pytest
import torch

import grilly


def rand(*shape, seed=0, positive=False):
    a = np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)
    return np.abs(a) + 0.5 if positive else a


def both(a, requires_grad=False):
    return grilly.tensor(a, requires_grad=requires_grad), torch.tensor(a, requires_grad=requires_grad)


def close(g, t):
    t = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    g = g.detach().numpy() if isinstance(g, grilly.Tensor) else np.asarray(g)
    assert g.shape == t.shape, (g.shape, t.shape)
    np.testing.assert_allclose(g, t, rtol=1e-5, atol=1e-6)


EXPRS = [
    # (name, lambda lib, x, y: ..., positive)
    ("scalar ops", lambda L, x, y: (2 * x + 1 - x / 4) * 3 - 2 / (x * x + 1) + 1.5 ** x, False),
    ("rsub/rdiv", lambda L, x, y: (1 - x) + (3 / (x + 2)), True),
    ("broadcast", lambda L, x, y: x * y[0] - y[:, :1] / (y + 2), True),
    ("pow", lambda L, x, y: x ** 2 + x.pow(3) + L.pow(x, 0.5) + y ** x[:1], True),
    ("unary", lambda L, x, y: x.exp() + x.log() + x.sqrt() + x.rsqrt() + x.tanh() + x.sigmoid() + x.abs() + (-x), True),
    ("clamp", lambda L, x, y: x.clamp(-0.3, 0.2) + L.clamp(x, min=0.1) + x.clamp_max(0.0), False),
    ("reductions", lambda L, x, y: x.sum() + x.sum(-1).sum() + x.mean((0, 1)) + x.sum(0, keepdim=True).mean(), False),
    ("amax/amin", lambda L, x, y: x.amax(-1).sum() + L.amin(x, 0).sum() + x.max() - x.min(), False),
    ("softmax", lambda L, x, y: L.softmax(x, -1) * L.log_softmax(x, dim=0), False),
    ("shape", lambda L, x, y: x.t().unsqueeze(0).squeeze(0).T.flatten().reshape(3, -1).permute(1, 0), False),
    ("indexing", lambda L, x, y: x[1] + x[-1, :] + x[0:3:2].sum(0) + x[..., 1:3].sum(-1).sum(), False),
    ("none/ellipsis", lambda L, x, y: x[None, ..., None].squeeze(), False),
    ("cat/stack", lambda L, x, y: L.cat([x, y], 0).sum(0) + L.stack([x, x], -1).sum(-1).sum(0), False),
    ("split/chunk", lambda L, x, y: sum(p.sum() for p in x.split(2, 1)) + sum(p.mean() for p in L.chunk(x, 2, 0)), False),
    ("matmul", lambda L, x, y: (x @ y.T) @ x + L.matmul(x[0], x.T).unsqueeze(-1), False),
    ("expand", lambda L, x, y: x[:, :1].expand(-1, 4) * y[:1].expand(3, 4), False),
]


@pytest.mark.parametrize("name,expr,positive", EXPRS, ids=[e[0] for e in EXPRS])
def test_expression_and_its_gradient_match_torch(name, expr, positive):
    a, b = rand(3, 4, seed=1, positive=positive), rand(3, 4, seed=2, positive=positive)
    (gx, tx), (gy, ty) = both(a, True), both(b, True)
    g_out, t_out = expr(grilly, gx, gy), expr(torch, tx, ty)
    close(g_out, t_out)
    upstream = rand(*t_out.shape, seed=3)
    g_out.backward(grilly.tensor(upstream))
    t_out.backward(torch.tensor(upstream))
    for gl, tl in [(gx, tx), (gy, ty)]:
        if tl.grad is None:
            assert gl.grad is None
        else:
            close(gl.grad, tl.grad)


def test_setitem_like_torch():
    for lib in (grilly, torch):
        x = lib.zeros(3, 4)
        x[0] = 1.0
        x[1, 1:3] = lib.tensor([5.0, 6.0])
        x[:, -1] = lib.ones(3) * 2
        x[2, ::2] = 7.0
        if lib is grilly:
            got = x.numpy()
        else:
            want = x.numpy()
    np.testing.assert_array_equal(got, want)


def test_setitem_follows_inplace_rules():
    w = grilly.ones(3, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf Variable"):
        w[0] = 2.0
    with grilly.no_grad():
        w[0] = 2.0
    assert w.detach().tolist() == [2.0, 1.0, 1.0]


def test_view_copies_refuse_writes_until_made_contiguous():
    x = grilly.zeros(2, 3)
    tr = x.t()
    with pytest.raises(NotImplementedError, match="returns a view"):
        tr.fill_(1.0)
    c = tr.contiguous()
    c.fill_(1.0)                                   # a real copy: allowed
    assert x.sum().item() == 0.0


def test_iteration_len_and_size():
    gx, tx = both(rand(3, 2))
    assert len(gx) == len(tx) == 3
    for gr, tr in zip(gx, tx):
        close(gr, tr)
    assert gx.size(-1) == tx.size(-1) == 2


def test_like_constructors_and_allclose():
    gx, tx = both(rand(2, 3))
    close(grilly.zeros_like(gx), torch.zeros_like(tx))
    close(grilly.ones_like(gx), torch.ones_like(tx))
    close(grilly.full_like(gx, 2.5), torch.full_like(tx, 2.5))
    assert grilly.allclose(gx, gx + 1e-9) == torch.allclose(tx, tx + 1e-9)
    assert grilly.allclose(gx, gx + 1) == torch.allclose(tx, tx + 1) == False  # noqa: E712


@pytest.mark.parametrize("call,exc", [
    (lambda L: L.ones(2, 3).sum(2), IndexError),
    (lambda L: L.ones(2, 3)[5], IndexError),
    (lambda L: L.ones(2, 3)[0, 0, 0], IndexError),
    (lambda L: L.ones(2, 3) + L.ones(4), RuntimeError),
    (lambda L: L.ones(2, 3).t().t().t().permute(0), RuntimeError),
    (lambda L: L.clamp(L.ones(2)), RuntimeError),
])
def test_errors_match_torch(call, exc):
    with pytest.raises(exc):
        call(torch)
    with pytest.raises(exc):
        call(grilly)


def test_comparisons_say_they_need_bool():
    x = grilly.ones(2)
    with pytest.raises(NotImplementedError, match="bool"):
        x == x  # noqa: B015
    assert hash(x) == hash(x) and {x: 1}[x] == 1   # identity hashing, as torch


def test_python_scalar_conversions_match_torch():
    for lib in (grilly, torch):
        assert float(lib.tensor([2.5])) == 2.5
        assert int(lib.tensor(-2.7)) == -2 and int(lib.tensor(2.7)) == 2
        assert bool(lib.zeros(1)) is False and bool(lib.ones(1, 1)) is True
        with pytest.raises(ValueError, match="one element"):
            float(lib.ones(2))
        with pytest.raises(RuntimeError, match="ambiguous"):
            bool(lib.ones(2))

