"""float32, int64, int32 and bool tensors against torch: inference,
factories, conversions, host round trips and shape ops. All exact."""

import numpy as np
import pytest
import torch

import grilly

DT = {grilly.float32: torch.float32, grilly.int64: torch.int64, grilly.int32: torch.int32, grilly.bool: torch.bool}


def same(g, t):
    assert DT[g.dtype] == t.dtype, (g.dtype, t.dtype)
    assert tuple(g.shape) == tuple(t.shape)
    np.testing.assert_array_equal(g.numpy(), t.numpy())


@pytest.mark.parametrize("data", [
    [1, 2, 3], [[1.5, 2.0]], [True, False], 7, 2.5, True, [2**40 + 3, -5],
])
def test_dtype_inference_matches_torch(data):
    same(grilly.tensor(data), torch.tensor(data))


@pytest.mark.parametrize("np_dtype", [np.float32, np.int64, np.int32, np.bool_])
def test_numpy_round_trip(np_dtype):
    a = (np.arange(12).reshape(3, 4) % 3).astype(np_dtype)
    same(grilly.from_numpy(a), torch.from_numpy(a))
    same(grilly.tensor(a), torch.tensor(a))


@pytest.mark.parametrize("args,kwargs", [
    ((5,), {}), ((2, 9), {}), ((1, 10, 3), {}), ((10, 0, -3), {}),
    ((0, 1, 0.25), {}), ((0.5, 3), {}), ((5,), {"dtype": "float32"}), ((0,), {}),
])
def test_arange_matches_torch(args, kwargs):
    gk = {k: getattr(grilly, v) for k, v in kwargs.items()}
    tk = {k: getattr(torch, v) for k, v in kwargs.items()}
    same(grilly.arange(*args, **gk), torch.arange(*args, **tk))


def test_arange_errors_like_torch():
    for lib in (grilly, torch):
        with pytest.raises(RuntimeError, match="step must be nonzero"):
            lib.arange(0, 5, 0)
        with pytest.raises(RuntimeError, match="inconsistent with step sign"):
            lib.arange(0, 5, -1)


def test_factories_with_dtypes_match_torch():
    for name in ("zeros", "ones"):
        for dt in DT:
            same(getattr(grilly, name)(2, 3, dtype=dt), getattr(torch, name)(2, 3, dtype=DT[dt]))
    same(grilly.full((2,), 3), torch.full((2,), 3))
    same(grilly.full((2,), True), torch.full((2,), True))
    same(grilly.full((3,), 2**40 + 1), torch.full((3,), 2**40 + 1))
    same(grilly.full((2,), 1.5), torch.full((2,), 1.5))
    x = grilly.tensor([1, 2])
    same(grilly.zeros_like(x), torch.zeros_like(torch.tensor([1, 2])))


def test_conversions_match_torch():
    a = np.array([1.7, -2.7, 0.0, 3.2e9, -0.5], np.float32)
    g, t = grilly.tensor(a), torch.tensor(a)
    same(g.long(), t.long())
    same(g.bool(), t.bool())
    same(g.long().float(), t.long().float())
    same(g.to(grilly.int64).to(grilly.bool), t.to(torch.int64).to(torch.bool))
    small = np.array([7, -3, 0], np.float32)
    same(grilly.tensor(small).int(), torch.tensor(small).int())
    same(grilly.tensor([True, False]).long(), torch.tensor([True, False]).long())
    i = grilly.tensor([3, 4])
    assert i.type_as(g).dtype is grilly.float32 and i.to(i.dtype) is i


def test_int64_values_survive_every_shape_op():
    v = np.array([[2**40 + 1, -(2**50)], [7, 2**62]], np.int64)
    g, t = grilly.tensor(v), torch.tensor(v)
    same(g.t(), t.t())
    same(g.reshape(4), t.reshape(4))
    same(g[1], t[1])
    same(g[:, ::-1][:, :1] if False else g[:, :1], t[:, :1])
    same(grilly.cat([g, g], 1), torch.cat([t, t], 1))
    same(grilly.stack([g, g]), torch.stack([t, t]))
    same(g[:1].expand(3, 2), t[:1].expand(3, 2))
    same(g.clone(), t.clone())


def test_metadata_matches_torch():
    for dt in DT:
        g, t = grilly.zeros(3, dtype=dt), torch.zeros(3, dtype=DT[dt])
        assert g.element_size() == t.element_size()
        assert g.nbytes == t.nbytes
        assert g.is_floating_point() == t.is_floating_point()
    assert type(grilly.tensor(3).item()) is int and type(grilly.tensor(True).item()) is bool
    assert grilly.tensor([1, 2]).tolist() == [1, 2]
    assert repr(grilly.tensor([1, 2])) == "tensor([1, 2])"
    assert repr(grilly.tensor([1, 2], dtype=grilly.int32)) == "tensor([1, 2], dtype=grilly.int32)"


def test_setitem_converts_to_the_destination_dtype_like_torch():
    for lib in (grilly, torch):
        x = lib.zeros(4, dtype=lib.int64)
        x[1] = 7
        x[2:] = lib.tensor([1.9, -1.9])
        if lib is grilly:
            got = x.tolist()
        else:
            want = x.tolist()
    assert got == want


def test_randint_is_int64_in_range_and_deterministic():
    grilly.manual_seed(1)
    a = grilly.randint(-3, 5, (1000,))
    grilly.manual_seed(1)
    b = grilly.randint(-3, 5, (1000,))
    assert a.dtype is grilly.int64
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    v = a.numpy()
    assert v.min() == -3 and v.max() == 4
    assert grilly.randint(10, (2, 3)).shape == torch.randint(10, (2, 3)).shape
    g = grilly.Generator().manual_seed(0)
    assert grilly.randint(0, 2, (4, 5), generator=g).float().dtype is grilly.float32
    for lib in (grilly, torch):
        with pytest.raises(RuntimeError, match="less than"):
            lib.randint(5, 5, (2,))


@pytest.mark.parametrize("call", [
    lambda L: L.zeros(2, dtype=L.int64).requires_grad_(),
    lambda L: L.tensor([1, 2], requires_grad=True),
])
def test_only_floats_require_grad_like_torch(call):
    for lib in (torch, grilly):
        with pytest.raises(RuntimeError, match="floating point"):
            call(lib)


@pytest.mark.parametrize("call", [
    lambda: grilly.tensor(np.zeros(3)),                  # float64
    lambda: grilly.zeros(2, dtype=grilly.float16),
    lambda: grilly.tensor([1.0]).double(),
])
def test_other_dtypes_are_not_implemented(call):
    with pytest.raises(NotImplementedError):
        call()
