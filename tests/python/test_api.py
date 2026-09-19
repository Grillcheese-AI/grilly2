"""grilly's Python surface against real torch (CPU).

Tolerances: relu, add and creation are exact. linear uses the K-scaled
bound from crates/grilly-torch/tests/ops_vs_torch.rs:
|grilly - torch| <= sqrt(K) * eps_f32 * (|x| @ |w|.T + |b|).
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

import grilly
import grilly.nn.functional as GF

EPS = np.finfo(np.float32).eps


def rand(*shape, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, shape).astype(np.float32)


def g(a):
    return grilly.from_numpy(a)


def t(a):
    return torch.from_numpy(a)


# ── creation and conversion ──────────────────────────────────────────────

def test_round_trip_and_metadata():
    a = rand(2, 3, 4)
    x = g(a)
    assert x.shape == (2, 3, 4) and isinstance(x.shape, grilly.Size)
    assert x.size() == torch.Size([2, 3, 4])
    assert x.size(-1) == 4 and x.dim() == 3 and x.ndim == 3 and x.numel() == 24
    assert x.dtype is grilly.float32 and len(x) == 2
    np.testing.assert_array_equal(x.numpy(), a)
    assert x.tolist() == t(a).tolist()


def test_factories_match_torch():
    for name in ("zeros", "ones"):
        for args in ((2, 3), ((2, 3),), ([4],), (0, 5)):
            got = getattr(grilly, name)(*args)
            want = getattr(torch, name)(*args)
            assert got.shape == want.shape
            np.testing.assert_array_equal(got.numpy(), want.numpy())
    assert grilly.empty(3, 2).shape == (3, 2)


def test_tensor_from_python_data():
    data = [[1.5, -2.0], [0.25, 4.0]]
    np.testing.assert_array_equal(grilly.tensor(data).numpy(), torch.tensor(data).numpy())
    assert grilly.tensor(3.0).shape == torch.tensor(3.0).shape == ()
    assert grilly.tensor(3.0).item() == 3.0


def test_tensor_copies():
    x = g(rand(4))
    y = grilly.tensor(x)
    assert y is not x and y._t is not x._t
    np.testing.assert_array_equal(y.numpy(), x.numpy())


def test_reshape_and_view():
    a = rand(2, 3, 4)
    x, xt = g(a), t(a)
    for shape in [(6, 4), (-1,), (4, -1), (2, -1, 2)]:
        np.testing.assert_array_equal(x.reshape(*shape).numpy(), xt.reshape(*shape).numpy())
        assert x.view(shape).shape == xt.view(shape).shape
    with pytest.raises(RuntimeError):
        x.reshape(5, -1)
    with pytest.raises(RuntimeError):
        x.reshape(5, 5)


def test_size_out_of_range_matches_torch_error_type():
    x = g(rand(2, 3))
    with pytest.raises(IndexError):
        x.size(2)
    with pytest.raises(IndexError):
        t(rand(2, 3)).size(2)


# ── what is not there yet says so ────────────────────────────────────────

@pytest.mark.parametrize("make", [
    lambda: grilly.tensor([1, 2, 3]),                   # torch: int64
    lambda: grilly.from_numpy(np.zeros(3)),             # torch: float64
    lambda: grilly.zeros(3, dtype=grilly.int64),
    lambda: grilly.zeros(3, device="cpu"),
    lambda: grilly.zeros(3, requires_grad=True),
    lambda: g(rand(3)) + 1.0,
    lambda: grilly.add(g(rand(3)), g(rand(3)), alpha=2),
    lambda: g(rand(2, 3)) + g(rand(3)),                 # broadcasting
    lambda: grilly.Tensor(2, 3),
])
def test_unimplemented_raises_not_implemented(make):
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        make()


# ── ops ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(1000,), (3, 7, 11), (0,)])
def test_relu_exact(shape):
    a = rand(*shape, seed=1)
    np.testing.assert_array_equal(grilly.relu(g(a)).numpy(), torch.relu(t(a)).numpy())
    np.testing.assert_array_equal(g(a).relu().numpy(), t(a).relu().numpy())
    np.testing.assert_array_equal(GF.relu(g(a)).numpy(), torch.nn.functional.relu(t(a)).numpy())


@pytest.mark.parametrize("shape", [(1000,), (4, 9, 13)])
def test_add_exact(shape):
    a, b = rand(*shape, seed=2), rand(*shape, seed=3)
    want = (t(a) + t(b)).numpy()
    np.testing.assert_array_equal((g(a) + g(b)).numpy(), want)
    np.testing.assert_array_equal(grilly.add(g(a), g(b)).numpy(), want)


@pytest.mark.parametrize("in_shape,out_f,bias", [
    ((37, 70), 23, True),
    ((2, 5, 33), 17, False),
    ((64,), 10, True),
    ((128, 1024), 256, True),
])
def test_linear_within_bound(in_shape, out_f, bias):
    x, w, b = rand(*in_shape, seed=4), rand(out_f, in_shape[-1], seed=5), rand(out_f, seed=6)
    got = GF.linear(g(x), g(w), g(b) if bias else None).numpy()
    tf = torch.nn.functional
    want = tf.linear(t(x), t(w), t(b) if bias else None).numpy()
    mag = tf.linear(t(x).abs(), t(w).abs(), t(b).abs() if bias else None).numpy()
    bound = np.sqrt(in_shape[-1]) * EPS * mag
    assert got.shape == want.shape
    assert np.all(np.abs(got - want) <= bound), np.max(np.abs(got - want) / bound)


def test_shape_errors_are_runtime_errors_like_torch():
    with pytest.raises(RuntimeError, match="features"):
        GF.linear(g(rand(2, 3)), g(rand(4, 5)))


# ── the torch alias ──────────────────────────────────────────────────────

def test_torch_alias_resolves_to_grilly():
    code = (
        "import grilly.torch_alias, torch, torch.nn.functional as F, grilly;"
        "assert torch is grilly and F is grilly.nn.functional;"
        "x = torch.ones(2, 3); print(F.linear(x, torch.ones(4, 3)).shape)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "grilly.Size([2, 4])"


def test_torch_alias_missing_module_fails_loudly():
    code = "import grilly.torch_alias; import torch.distributed"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode != 0 and "ModuleNotFoundError" in out.stderr


def test_import_is_quiet():
    out = subprocess.run([sys.executable, "-c", "import grilly; grilly.zeros(1).numpy()"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout == ""
