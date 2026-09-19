"""Index ops against torch: gather, index_select, advanced indexing, masked
assignment, embedding, class-index cross_entropy / nll_loss.

Forward values are gathered copies, so they are exact. Gradients sum
duplicate indices: exact for the deterministic kernel on small integer-valued
sums, rtol 1e-6 for the atomic kernel (hardware summation order). Loss
values use rtol 1e-6, loss gradients rtol 1e-5 (softmax minus a one-hot,
then a mean)."""

import numpy as np
import pytest
import torch

import grilly
import grilly.nn.functional as F

RNG = np.random.default_rng(7)
X = RNG.standard_normal((4, 5, 3)).astype(np.float32)


def leaves():
    return grilly.tensor(X, requires_grad=True), torch.tensor(X, requires_grad=True)


@pytest.fixture(params=[False, True], ids=["atomic", "deterministic"])
def mode(request):
    grilly.use_deterministic_algorithms(request.param)
    yield request.param
    grilly.use_deterministic_algorithms(False)


def check(fg, ft, rtol=1e-6):
    a, t = leaves()
    yg, yt = fg(a), ft(t)
    assert tuple(yg.shape) == tuple(yt.shape)
    np.testing.assert_array_equal(yg.detach().numpy(), yt.detach().numpy())
    w = RNG.standard_normal(tuple(yt.shape)).astype(np.float32)
    (yg * grilly.tensor(w)).sum().backward()
    (yt * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(a.grad.numpy(), t.grad.numpy(), rtol=rtol, atol=1e-6)


GATHER = RNG.integers(0, 5, (4, 7, 3))


@pytest.mark.parametrize("dim,index", [
    (1, GATHER), (0, RNG.integers(0, 4, (2, 5, 3))), (2, RNG.integers(-3, 3, (4, 5, 6))),
    (1, RNG.integers(0, 5, (3, 2, 2))),  # index smaller than input in other dims
])
def test_gather_matches_torch(mode, dim, index):
    if index.min() < 0:
        with pytest.raises(RuntimeError):
            grilly.gather(grilly.tensor(X), dim, grilly.tensor(index))
        return
    check(lambda a: a.gather(dim, grilly.tensor(index)), lambda t: t.gather(dim, torch.tensor(index)))


@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_index_select_matches_torch(mode, dim):
    index = [2, 0, 2, 1, 2]
    check(lambda a: grilly.index_select(a, dim, grilly.tensor(index)),
          lambda t: torch.index_select(t, dim, torch.tensor(index)))


@pytest.mark.parametrize("key", [
    lambda T: T([3, 0, 3]),
    lambda T: [1, -1],
    lambda T: T([[0, 1], [3, 3]]),
    lambda T: (slice(None), T([4, 0, 4])),
    lambda T: (slice(None), slice(None), T([-1, 0])),
    lambda T: (T([0, 3, 3]), T([1, 4, 1])),
    lambda T: (T([[0], [3]]), T([1, 4, 2])),  # broadcast index tensors
    lambda T: (T([0, 3]), T([1, 4]), T([2, 0])),
    lambda T: (0, [3, 1]),                              # an int is basic indexing, as in torch
    lambda T: (1, slice(None), [0, 2, 2]),
    lambda T: (slice(None), 0, T([1, 2])),
    lambda T: (T([0, 2]), slice(None), T([1, 2])),      # non-adjacent: index dims go first
    lambda T: (T([0, 2]), 1, T([1, 2])),
    lambda T: (slice(None), T([1, 2]), T([0, 2])),      # adjacent: index dims stay in place
    lambda T: (slice(1, 3), T([[0], [3]]), T([2, 0])),
    lambda T: (Ellipsis, T([2, 0])),
])
def test_advanced_indexing_matches_torch(mode, key):
    def run(lib):
        k = key(lib.tensor)
        return lambda x: x[k]
    check(run(grilly), run(torch))


def test_bool_mask_indexing_matches_torch(mode):
    check(lambda a: a[a > 0.3], lambda t: t[t > 0.3])
    m = X[:, :, 0] > 0
    check(lambda a: a[grilly.tensor(m)], lambda t: t[torch.tensor(m)])


def test_masked_assignment_matches_torch():
    g, t = grilly.tensor(X), torch.tensor(X)
    g[g < 0] = 0.5
    t[t < 0] = 0.5
    np.testing.assert_array_equal(g.numpy(), t.numpy())
    m = X[:, 0, 0] > 0
    g[grilly.tensor(m)] = -1.0
    t[torch.tensor(m)] = -1.0
    np.testing.assert_array_equal(g.numpy(), t.numpy())


def test_advanced_assignment_matches_torch():
    g, t = grilly.tensor(X), torch.tensor(X)
    for lib, x in ((grilly, g), (torch, t)):
        x[0, [1, 3]] = x[0, [3, 1]]
        x[:, 1, [0, 2]] = 5.0
        x[lib.tensor([3, 0]), lib.tensor([4, 4])] = lib.tensor([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(g.numpy(), t.numpy())


def test_index_errors_match_torch():
    g, t = grilly.tensor(X), torch.tensor(X)
    for key in ([4], [-5], ([0], [5])):
        with pytest.raises(IndexError):
            t[tuple(torch.tensor(k) for k in key) if isinstance(key, tuple) else torch.tensor(key)]
        with pytest.raises(IndexError):
            g[tuple(grilly.tensor(k) for k in key) if isinstance(key, tuple) else grilly.tensor(key)]
    with pytest.raises(RuntimeError):
        grilly.index_select(g, 0, grilly.tensor([4]))
    with pytest.raises(RuntimeError):
        grilly.gather(g, 1, grilly.tensor(np.full((1, 1, 1), 5)))
    with pytest.raises(IndexError):
        g[grilly.tensor([True, False])]
    with pytest.raises(IndexError, match="must be long, int, byte or bool"):
        t[torch.tensor([0.0])]
    with pytest.raises(IndexError, match="must be long, int, byte or bool"):
        g[grilly.tensor([0.0])]


@pytest.mark.parametrize("padding_idx", [None, 0, -1])
def test_embedding_matches_torch(mode, padding_idx):
    w = RNG.standard_normal((10, 4)).astype(np.float32)
    idx = np.array([[0, 3, 9, 3], [9, 9, 1, 0]])
    wg, wt = grilly.tensor(w, requires_grad=True), torch.tensor(w, requires_grad=True)
    yg = F.embedding(grilly.tensor(idx), wg, padding_idx)
    yt = torch.nn.functional.embedding(torch.tensor(idx), wt, padding_idx)
    np.testing.assert_array_equal(yg.detach().numpy(), yt.detach().numpy())
    g = RNG.standard_normal(tuple(yt.shape)).astype(np.float32)
    (yg * grilly.tensor(g)).sum().backward()
    (yt * torch.tensor(g)).sum().backward()
    np.testing.assert_allclose(wg.grad.numpy(), wt.grad.numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(IndexError, match="index out of range in self"):
        F.embedding(grilly.tensor([10]), wg)


def test_embedding_module():
    e = grilly.nn.Embedding(6, 3, padding_idx=2)
    assert np.all(e.weight.detach().numpy()[2] == 0)
    e(grilly.tensor([[2, 1], [5, 2]])).sum().backward()
    np.testing.assert_array_equal(e.weight.grad.numpy()[:, 0], [0, 1, 0, 0, 0, 1])


def test_deterministic_backward_is_bitwise_repeatable():
    grilly.use_deterministic_algorithms(True)
    try:
        assert grilly.are_deterministic_algorithms_enabled()
        w = RNG.standard_normal((3, 8)).astype(np.float32)
        idx = grilly.tensor(RNG.integers(0, 3, 4096))
        g = grilly.tensor(RNG.standard_normal((4096, 8)).astype(np.float32))
        runs = []
        for _ in range(3):
            wg = grilly.tensor(w, requires_grad=True)
            (F.embedding(idx, wg) * g).sum().backward()
            runs.append(wg.grad.numpy().tobytes())
        assert runs[0] == runs[1] == runs[2]
    finally:
        grilly.use_deterministic_algorithms(False)
    assert not grilly.are_deterministic_algorithms_enabled()


@pytest.mark.parametrize("fn", ["cross_entropy", "nll_loss"])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("smoothing", [0.0, 0.2])
@pytest.mark.parametrize("shape,target", [
    ((6, 5), [1, 4, -100, 2, 0, 4]),
    ((2, 5, 3), [[0, 4, 1], [-100, 2, 2]]),  # (N, C, d1)
    ((5,), 3),
])
def test_class_index_losses_match_torch(fn, reduction, smoothing, shape, target):
    if fn == "nll_loss" and smoothing:
        pytest.skip("nll_loss has no label_smoothing")
    x = RNG.standard_normal(shape).astype(np.float32)
    kw = dict(reduction=reduction, **({"label_smoothing": smoothing} if smoothing else {}))
    xg, xt = grilly.tensor(x, requires_grad=True), torch.tensor(x, requires_grad=True)
    lg = getattr(F, fn)(xg, grilly.tensor(target), **kw)
    lt = getattr(torch.nn.functional, fn)(xt, torch.tensor(target), **kw)
    np.testing.assert_allclose(lg.detach().numpy(), lt.detach().numpy(), rtol=1e-6, atol=1e-6)
    w = RNG.standard_normal(tuple(lt.shape)).astype(np.float32)
    (lg * grilly.tensor(w)).sum().backward()
    (lt * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(xg.grad.numpy(), xt.grad.numpy(), rtol=1e-5, atol=1e-6)
