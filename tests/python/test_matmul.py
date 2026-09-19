"""matmul / bmm over every rank combination torch accepts, including
broadcast batch dimensions, against a float64 reference.

Tolerance: |got - ref| <= sqrt(K) * eps * (|a| @ |b|), the bound used for
linear (crates/grilly-torch/tests/ops_vs_torch.rs); g2-bmm measured at most
0.41 of it. Gradients are matmuls too and get the same bound with their own
K (dA = g @ B^T sums over N, dB = A^T @ g over M and the broadcast batch)."""

import numpy as np
import pytest
import torch

import grilly

EPS = np.finfo(np.float32).eps
RNG = np.random.default_rng(8)


def within(got, a, b, k):
    ref = np.matmul(a.astype(np.float64), b.astype(np.float64))
    mag = np.matmul(np.abs(a).astype(np.float64), np.abs(b).astype(np.float64))
    assert got.shape == ref.shape, (got.shape, ref.shape)
    assert np.all(np.abs(got - ref) <= np.sqrt(k) * EPS * mag + 1e-30)


SHAPES = [
    ((5,), (5,)), ((5,), (5, 3)), ((4, 5), (5,)), ((4, 5), (5, 3)),
    ((2, 4, 5), (5,)), ((5,), (2, 5, 3)),
    ((3, 17, 33), (3, 33, 5)), ((2, 8, 64, 16), (2, 8, 16, 64)),
    ((2, 1, 7, 40), (1, 4, 40, 9)),   # broadcast batch dims
    ((6, 20), (4, 6, 20, 30)),        # 2-D @ batched
    ((4, 6, 20, 30), (30, 2)),        # batched @ 2-D (the linear path)
    ((3, 1, 5), (2, 1, 5, 4)),        # a missing leading batch dim
    ((2, 0, 5), (2, 5, 3)), ((2, 4, 0), (2, 0, 3)),  # empty
]


@pytest.mark.parametrize("sa,sb", SHAPES)
def test_matmul_values_within_bound(sa, sb):
    a, b = RNG.uniform(-1, 1, sa).astype(np.float32), RNG.uniform(-1, 1, sb).astype(np.float32)
    got = (grilly.tensor(a) @ grilly.tensor(b)).numpy()
    within(got, a, b, sa[-1])
    assert got.shape == tuple((torch.tensor(a) @ torch.tensor(b)).shape)


@pytest.mark.parametrize("sa,sb", [s for s in SHAPES if len(s[0]) > 1 and len(s[1]) > 1])
def test_matmul_gradients_within_bound(sa, sb):
    a, b = RNG.uniform(-1, 1, sa).astype(np.float32), RNG.uniform(-1, 1, sb).astype(np.float32)
    ga, gb = grilly.tensor(a, requires_grad=True), grilly.tensor(b, requires_grad=True)
    out = ga @ gb
    g = RNG.uniform(-1, 1, tuple(out.shape)).astype(np.float32)
    (out * grilly.tensor(g)).sum().backward()
    ta, tb = torch.tensor(a, dtype=torch.float64, requires_grad=True), torch.tensor(b, dtype=torch.float64, requires_grad=True)
    ((ta @ tb) * torch.tensor(g, dtype=torch.float64)).sum().backward()
    # magnitudes: the same products with |.|
    ma, mb = torch.tensor(np.abs(a), dtype=torch.float64, requires_grad=True), torch.tensor(np.abs(b), dtype=torch.float64, requires_grad=True)
    ((ma @ mb) * torch.tensor(np.abs(g), dtype=torch.float64)).sum().backward()
    batch = int(np.prod(out.shape[:-2])) if out.dim() > 2 else 1
    k_a = out.shape[-1] * batch            # dA sums over N (and broadcast batch)
    k_b = (out.shape[-2] if out.dim() > 1 else 1) * batch
    assert np.all(np.abs(ga.grad.numpy() - ta.grad.numpy()) <= np.sqrt(k_a) * EPS * ma.grad.numpy() + 1e-30)
    assert np.all(np.abs(gb.grad.numpy() - tb.grad.numpy()) <= np.sqrt(k_b) * EPS * mb.grad.numpy() + 1e-30)


def test_bmm_and_errors_match_torch():
    a, b = RNG.standard_normal((3, 4, 5)).astype(np.float32), RNG.standard_normal((3, 5, 2)).astype(np.float32)
    within(grilly.bmm(grilly.tensor(a), grilly.tensor(b)).numpy(), a, b, 5)
    for lib in (torch, grilly):
        with pytest.raises(RuntimeError):
            lib.zeros(2, 3, 4) @ lib.zeros(2, 5, 6)
        with pytest.raises(RuntimeError):
            lib.zeros(2, 3, 4) @ lib.zeros(3, 4, 6)   # batch 2 vs 3
        with pytest.raises(RuntimeError, match="at least 1D"):
            lib.matmul(lib.tensor(1.0), lib.zeros(2))
