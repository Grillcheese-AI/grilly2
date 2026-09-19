"""F.layer_norm / nn.LayerNorm (fused g2-layer-norm kernels) against float64.

Tolerance: each of y, dx, dw, db is within max(2 x torch float32's own
error, 8 eps) of torch float64, errors measured as |got - ref| / (|ref| + 1).
Measured, grilly's error was at or below torch float32's except on rows of
3 elements, where both are ~1e-7 (y 2.0e-7 vs 1.6e-7, dw 6.0e-7 vs
2.5e-7). The mean and variance are compensated sums and the variance takes
a second pass over (x - mean)^2, so a large mean with a small spread (where
a one-pass variance cancels) stays accurate.
"""

import numpy as np
import pytest
import torch

import grilly
import grilly.nn.functional as F

EPS = np.finfo(np.float32).eps
RNG = np.random.default_rng(17)


def run(shape, norm, affine=(True, True), scale=1.0, offset=0.0, eps=1e-5, needs=(True, True, True)):
    x = (RNG.standard_normal(shape) * scale + offset).astype(np.float32)
    w = RNG.standard_normal(norm).astype(np.float32) if affine[0] else None
    b = RNG.standard_normal(norm).astype(np.float32) if affine[1] else None
    g = RNG.standard_normal(shape).astype(np.float32)

    def once(lib, dtype):
        make = (lambda a, rg: lib.tensor(a, dtype=dtype, requires_grad=rg)) if lib is torch else \
               (lambda a, rg: lib.tensor(a, requires_grad=rg))
        xs = make(x, needs[0])
        ws = None if w is None else make(w, needs[1])
        bs = None if b is None else make(b, needs[2])
        fn = torch.nn.functional.layer_norm if lib is torch else F.layer_norm
        y = fn(xs, norm, ws, bs, eps)
        (y * lib.tensor(g, dtype=dtype) if lib is torch else y * lib.tensor(g)).sum().backward()
        grad = lambda t: None if t is None or not t.requires_grad else t.grad.numpy().astype(np.float64)  # noqa: E731
        return [y.detach().numpy().astype(np.float64), grad(xs), grad(ws), grad(bs)]

    got, t32, ref = once(grilly, None), once(torch, torch.float32), once(torch, torch.float64)
    for name, a, b32, r in zip(("y", "dx", "dw", "db"), got, t32, ref):
        if r is None:
            assert a is None, name
            continue
        err = lambda v: np.max(np.abs(v - r) / (np.abs(r) + 1)) if r.size else 0.0  # noqa: E731
        assert err(a) <= max(2 * err(b32), 8 * EPS), (name, err(a), err(b32))


@pytest.mark.parametrize("shape,norm", [
    ((64, 128, 768), (768,)), ((4096, 2048), (2048,)), ((8, 3, 5, 7), (5, 7)), ((2, 3, 4, 5), (3, 4, 5)),
    ((10, 3), (3,)), ((7,), (7,)), ((70_000, 8), (8,)),  # > 65535 rows: the 2-D workgroup grid
])
def test_layer_norm_matches_float64(shape, norm):
    run(shape, norm)


@pytest.mark.parametrize("affine", [(False, False), (True, False), (False, True)])
def test_layer_norm_without_full_affine(affine):
    run((32, 96), (96,), affine=affine)


def test_large_mean_small_spread():
    """x ~ 1e4 + N(0, 1e-2): a one-pass E[x^2] - E[x]^2 would lose it all."""
    run((64, 512), (512,), scale=1e-2, offset=1e4)


@pytest.mark.parametrize("needs", [(True, False, False), (False, True, False), (False, False, True), (False, True, True)])
def test_partial_requires_grad(needs):
    run((16, 64), (64,), needs=needs)


def test_eps_and_module():
    run((16, 64), (64,), eps=0.1)
    m, tm = grilly.nn.LayerNorm(12), torch.nn.LayerNorm(12)
    x = RNG.standard_normal((5, 12)).astype(np.float32)
    np.testing.assert_allclose(m(grilly.tensor(x)).detach().numpy(), tm(torch.tensor(x)).detach().numpy(),
                               rtol=1e-5, atol=1e-6)


def test_empty_and_errors_match_torch():
    assert tuple(F.layer_norm(grilly.zeros(0, 4), (4,)).shape) == (0, 4)
    for lib, fn in ((torch, torch.nn.functional.layer_norm), (grilly, F.layer_norm)):
        with pytest.raises(RuntimeError, match="normalized_shape"):
            fn(lib.zeros(3, 4), (5,))
