"""F.rms_norm / nn.RMSNorm against torch float64, forward and backward.

The kernel is composed from grilly ops: x * rsqrt(mean(x^2) + eps) * w,
torch's own formula.

Tolerance: within max(2x torch float32's own error, 8 eps) of torch
float64, with errors measured as |got - ref| / (|ref| + 1), as in
test_layer_norm.py.
"""

import numpy as np
import pytest
import torch

import grilly
import grilly.nn as gnn
import grilly.nn.functional as F

EPS = np.finfo(np.float32).eps
RNG = np.random.default_rng(23)


def err(a, ref):
    return float(np.max(np.abs(np.asarray(a, np.float64) - ref) / (np.abs(ref) + 1)))


@pytest.mark.parametrize("shape,norm", [((4, 896), (896,)), ((2, 3, 64), (64,)), ((5, 6, 7), (6, 7)), ((3,), (3,))])
@pytest.mark.parametrize("eps", [None, 1e-6])
def test_rms_norm_matches_float64(shape, norm, eps):
    x = (RNG.standard_normal(shape) * 3 + 0.5).astype(np.float32)
    w = RNG.standard_normal(norm).astype(np.float32)
    g = RNG.standard_normal(shape).astype(np.float32)

    def run(lib, dtype):
        xt = lib.tensor(x.astype(dtype), requires_grad=True)
        wt = lib.tensor(w.astype(dtype), requires_grad=True)
        y = (torch.nn.functional if lib is torch else F).rms_norm(xt, norm, wt, eps)
        (y * lib.tensor(g.astype(dtype))).sum().backward()
        return [t.detach().numpy().astype(np.float64) for t in (y, xt.grad, wt.grad)]

    ref = run(torch, np.float64)
    t32 = run(torch, np.float32)
    got = run(grilly, np.float32)
    for name, a, b, r in zip(("y", "dx", "dw"), got, t32, ref):
        bound = max(2 * err(b, r), 8 * EPS)
        assert err(a, r) <= bound, f"{name}: {err(a, r):.2e} > {bound:.2e}"


def test_module_and_errors_match_torch():
    m = gnn.RMSNorm(8, eps=1e-6)
    t = torch.nn.RMSNorm(8, eps=1e-6)
    assert repr(m) == repr(t)
    assert tuple(m.weight.shape) == (8,) and list(m.state_dict()) == ["weight"]
    np.testing.assert_array_equal(m.weight.detach().numpy(), np.ones(8, np.float32))
    assert gnn.RMSNorm(8, elementwise_affine=False).weight is None
    x = RNG.standard_normal((2, 8)).astype(np.float32)
    np.testing.assert_allclose(m(grilly.from_numpy(x)).detach().numpy(), t(torch.from_numpy(x)).detach().numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(RuntimeError, match="normalized_shape"):
        F.rms_norm(grilly.zeros(2, 8), (7,))
