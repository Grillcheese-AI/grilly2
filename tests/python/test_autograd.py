"""Autograd from Python, against real torch.

Gradient tolerances follow the Rust tests: linear's gradients are
reductions, bounded by sqrt(K) * eps * (the same reduction over |values|).
Everything else here is exact.
"""

import numpy as np
import pytest
import torch

import grilly
import grilly.nn.functional as GF

EPS = np.finfo(np.float32).eps


def rand(*shape, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)


def both(a, requires_grad=True):
    """The same values as a grilly leaf and a torch leaf."""
    return (grilly.tensor(a, requires_grad=requires_grad),
            torch.tensor(a, requires_grad=requires_grad))


def test_linear_relu_add_grads_match_torch():
    x, w, b = rand(8, 16, seed=1), rand(12, 16, seed=2), rand(12, seed=3)
    g = rand(8, 12, seed=4)
    (gx, tx), (gw, tw), (gb, tb) = both(x), both(w), both(b)

    h = GF.linear(gx, gw, gb)
    (grilly.relu(h) + h).backward(grilly.tensor(g))
    th = torch.nn.functional.linear(tx, tw, tb)
    (torch.relu(th) + th).backward(torch.tensor(g))

    ag = np.abs(g)
    bounds = [2 * np.sqrt(12) * EPS * (ag @ np.abs(w)),
              2 * np.sqrt(8) * EPS * (ag.T @ np.abs(x)),
              2 * np.sqrt(8) * EPS * ag.sum(0)]
    for (gl, tl), bound in zip([(gx, tx), (gw, tw), (gb, tb)], bounds):
        assert np.all(np.abs(gl.grad.numpy() - tl.grad.numpy()) <= bound)


def test_leaf_and_graph_attributes_match_torch():
    (gx, tx) = both(rand(3))
    gc, tc = both(rand(3), requires_grad=False)
    for x, c, lib in [(gx, gc, grilly), (tx, tc, torch)]:
        y = lib.relu(x)
        assert x.is_leaf and x.requires_grad and x.grad_fn is None
        assert not y.is_leaf and y.requires_grad
        assert c.is_leaf and not c.requires_grad
        d = y.detach()
        assert d.is_leaf and not d.requires_grad
    assert grilly.relu(gx).grad_fn.name() == torch.relu(tx).grad_fn.name() == "ReluBackward0"
    assert (gx + gx).grad_fn.name() == (tx + tx).grad_fn.name() == "AddBackward0"
    assert gx.clone().grad_fn.name() == tx.clone().grad_fn.name() == "CloneBackward0"
    assert gx.view(3).grad_fn.name() == tx.view(3).grad_fn.name() == "ViewBackward0"
    assert "grad_fn=<ReluBackward0>" in repr(grilly.relu(gx))
    assert "requires_grad=True" in repr(gx)


def test_requires_grad_toggles_like_torch():
    for lib in (grilly, torch):
        x = lib.zeros(2)
        assert x.requires_grad_() is x and x.requires_grad
        x.requires_grad = False
        assert not x.requires_grad
        y = lib.relu(lib.ones(2, requires_grad=True))
        with pytest.raises(RuntimeError, match="leaf variables"):
            y.requires_grad_(False)


def test_grad_accumulates_and_resets_like_torch():
    a = rand(4)
    gx, tx = both(a)
    for x, lib in [(gx, grilly), (tx, torch)]:
        (x + x).backward(lib.ones(4))
        first = x.grad
        (x + x).backward(lib.ones(4))
        assert x.grad is first                     # same object, updated in place
        np.testing.assert_array_equal(first.numpy(), np.full(4, 4.0, np.float32))
        x.grad.zero_()
        np.testing.assert_array_equal(x.grad.numpy(), np.zeros(4, np.float32))
        x.grad = None
        assert x.grad is None


def test_scalar_backward_needs_no_gradient():
    gx, tx = both(np.array(2.5, np.float32))
    (gx + gx).backward()
    (tx + tx).backward()
    assert gx.grad.item() == tx.grad.item() == 2.0


@pytest.mark.parametrize("case", ["second_backward", "no_requires_grad", "non_scalar", "numpy"])
def test_errors_match_torch(case):
    def run(lib):
        x = lib.tensor(rand(3), requires_grad=True)
        if case == "second_backward":
            y = lib.relu(x)
            y.backward(lib.ones(3))
            y.backward(lib.ones(3))
        elif case == "no_requires_grad":
            lib.relu(lib.ones(3)).backward(lib.ones(3))
        elif case == "non_scalar":
            lib.relu(x).backward()
        else:
            x.numpy()
    with pytest.raises(Exception) as want:
        run(torch)
    with pytest.raises(want.type):
        run(grilly)


def test_retain_graph_allows_a_second_backward():
    gx, tx = both(rand(3))
    for x, lib in [(gx, grilly), (tx, torch)]:
        y = lib.relu(x)
        y.backward(lib.ones(3), retain_graph=True)
        y.backward(lib.ones(3))
    np.testing.assert_array_equal(gx.grad.numpy(), tx.grad.numpy())


def test_grad_mode_context_managers_and_decorators():
    x = grilly.ones(2, requires_grad=True)
    assert grilly.is_grad_enabled()
    with grilly.no_grad():
        assert not grilly.is_grad_enabled()
        assert not grilly.relu(x).requires_grad
        with grilly.enable_grad():
            assert grilly.relu(x).requires_grad
        assert not grilly.is_grad_enabled()
    assert grilly.is_grad_enabled()

    @grilly.no_grad()
    def f():
        return grilly.is_grad_enabled()

    assert f() is False and grilly.is_grad_enabled()

    grilly.set_grad_enabled(False)          # function form applies immediately
    assert not grilly.is_grad_enabled()
    grilly.set_grad_enabled(True)
    with grilly.set_grad_enabled(False):
        assert not grilly.is_grad_enabled()
    assert grilly.is_grad_enabled()
    # the torch alias spelling
    import grilly.autograd as ag
    assert ag.no_grad is grilly.no_grad


def test_inplace_rules_match_torch():
    for lib in (grilly, torch):
        w = lib.ones(3, requires_grad=True)
        with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
            w.zero_()
        with lib.no_grad():
            w.fill_(0.5)
            w.add_(lib.ones(3))
        np.testing.assert_array_equal(w.detach().numpy(), np.full(3, 1.5, np.float32))
        assert w._version == 2


def test_modified_saved_tensor_is_caught_like_torch():
    for lib in (grilly, torch):
        x = lib.ones(2, 3, requires_grad=True)
        w = lib.ones(4, 3, requires_grad=True)
        y = (GF if lib is grilly else torch.nn.functional).linear(x, w)
        with lib.no_grad():
            x.add_(lib.ones(2, 3))
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.backward(lib.ones(2, 4))


def test_seeded_normal_matches_itself():
    a, b = grilly.zeros(100), grilly.zeros(100)
    a.normal_(1.0, 2.0, generator=grilly.Generator().manual_seed(3))
    b.normal_(1.0, 2.0, generator=grilly.Generator().manual_seed(3))
    np.testing.assert_array_equal(a.numpy(), b.numpy())


@pytest.mark.parametrize("call", [
    lambda: grilly.relu(grilly.ones(2, requires_grad=True)).backward(grilly.ones(2), create_graph=True),
    lambda: grilly.relu(grilly.ones(2, requires_grad=True)).add_(grilly.ones(2)),
])
def test_not_yet_implemented(call):
    with pytest.raises(NotImplementedError):
        call()
