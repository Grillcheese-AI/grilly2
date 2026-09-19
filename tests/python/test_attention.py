"""F.scaled_dot_product_attention against torch 2.10 (CPU).

Tolerance: outputs and gradients within rtol 1e-5, atol 1e-6 (float32
matmuls of depth <= 64 and a softmax; measured max 5e-7). Semantics checked
beyond the values: fully-masked rows give 0 with a zero gradient (torch's
_safe_softmax), is_causal keeps the top-left triangle when L != S, and a bool
mask with is_causal is their intersection (what torch 2.10 does on CPU)."""

import math

import numpy as np
import pytest
import torch

import grilly
import grilly.nn.functional as GF

RNG = np.random.default_rng(21)
TF = torch.nn.functional


def rand(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def compare(q, k, v, mask=None, rtol=1e-5, atol=1e-6, grad_mask=False, **kw):
    """Values and q/k/v (and float mask) gradients against torch."""
    G = [grilly.tensor(a, requires_grad=True) for a in (q, k, v)]
    T = [torch.tensor(a, requires_grad=True) for a in (q, k, v)]
    gm = tm = None
    if mask is not None:
        gm = grilly.tensor(mask, requires_grad=grad_mask)
        tm = torch.tensor(mask, requires_grad=grad_mask)
    og = GF.scaled_dot_product_attention(*G, attn_mask=gm, **kw)
    ot = TF.scaled_dot_product_attention(*T, attn_mask=tm, **kw)
    assert tuple(og.shape) == tuple(ot.shape)
    np.testing.assert_allclose(og.detach().numpy(), ot.detach().numpy(), rtol=rtol, atol=atol)
    w = rand(*ot.shape)
    (og * grilly.tensor(w)).sum().backward()
    (ot * torch.tensor(w)).sum().backward()
    pairs = list(zip(G, T)) + ([(gm, tm)] if grad_mask else [])
    for g, t in pairs:
        np.testing.assert_allclose(g.grad.numpy(), t.grad.numpy(), rtol=rtol, atol=atol)
    return og


@pytest.mark.parametrize("qs,ks", [
    ((2, 4, 6, 8), (2, 4, 6, 8)),
    ((2, 4, 5, 16), (2, 4, 9, 16)),   # L != S
    ((3, 7, 32), (3, 7, 32)),         # 3-D (N, L, E)
    ((1, 1, 1, 64), (1, 1, 40, 64)),  # one query (decode step)
])
def test_unmasked_matches_torch(qs, ks):
    compare(rand(*qs), rand(*ks), rand(*ks[:-1], 12))


@pytest.mark.parametrize("mask_shape", [(6, 6), (2, 1, 6, 6), (2, 4, 6, 6), (1, 6)])
def test_bool_masks_match_torch(mask_shape):
    mask = RNG.random(mask_shape) > 0.3
    mask[..., 0] = True  # keep at least one key per row here; dead rows have their own test
    compare(rand(2, 4, 6, 8), rand(2, 4, 6, 8), rand(2, 4, 6, 8), mask=mask)


def test_float_mask_is_added_and_differentiable():
    mask = rand(2, 1, 6, 6)
    mask[..., 2] = -np.inf
    compare(rand(2, 4, 6, 8), rand(2, 4, 6, 8), rand(2, 4, 6, 8), mask=mask, grad_mask=False)
    compare(rand(2, 4, 6, 8), rand(2, 4, 6, 8), rand(2, 4, 6, 8), mask=rand(6, 6), grad_mask=True)


@pytest.mark.parametrize("L,S", [(6, 6), (4, 7), (7, 4)])
def test_is_causal_top_left_matches_torch(L, S):
    compare(rand(2, 3, L, 8), rand(2, 3, S, 8), rand(2, 3, S, 8), is_causal=True)


def test_bool_mask_with_is_causal_is_their_intersection():
    mask = RNG.random((6, 6)) > 0.3
    compare(rand(1, 2, 6, 8), rand(1, 2, 6, 8), rand(1, 2, 6, 8), mask=mask, is_causal=True)


def test_scale_and_gqa_match_torch():
    compare(rand(2, 4, 5, 8), rand(2, 4, 5, 8), rand(2, 4, 5, 8), scale=0.37)
    compare(rand(2, 8, 5, 16), rand(2, 2, 5, 16), rand(2, 2, 5, 16), enable_gqa=True)


def test_fully_masked_rows_give_zero_and_zero_gradient():
    mask = np.ones((6, 6), bool)
    mask[2] = False
    fmask = np.zeros((6, 6), np.float32)
    fmask[4] = -np.inf
    for m in (mask, fmask):
        out = compare(rand(1, 2, 6, 8), rand(1, 2, 6, 8), rand(1, 2, 6, 8), mask=m)
        row = 2 if m.dtype == bool else 4
        assert np.all(out.detach().numpy()[:, :, row] == 0)


def test_cubbyllm_window_attention_matches_torch():
    """LocalCausalAttention.forward's mask: causal, within a window of 3."""
    S = 10
    i = np.arange(S)
    keep = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < 3)
    compare(rand(2, 4, S, 8), rand(2, 4, S, 8), rand(2, 4, S, 8), mask=keep)


def test_dropout():
    q, k, v = (grilly.tensor(rand(2, 2, 6, 8)) for _ in range(3))
    plain = GF.scaled_dot_product_attention(q, k, v).numpy()
    np.testing.assert_array_equal(GF.scaled_dot_product_attention(q, k, v, dropout_p=0.0).numpy(), plain)
    dropped = GF.scaled_dot_product_attention(q, k, v, dropout_p=0.5).numpy()
    assert dropped.shape == plain.shape and not np.allclose(dropped, plain)
    assert np.all(GF.scaled_dot_product_attention(q, k, v, dropout_p=1.0).numpy() == 0)


def test_default_scale_is_one_over_sqrt_e():
    q, k, v = rand(1, 1, 3, 25), rand(1, 1, 3, 25), rand(1, 1, 3, 4)
    manual = torch.softmax(torch.tensor(q) @ torch.tensor(k).transpose(-2, -1) / math.sqrt(25), -1) @ torch.tensor(v)
    got = GF.scaled_dot_product_attention(grilly.tensor(q), grilly.tensor(k), grilly.tensor(v)).numpy()
    np.testing.assert_allclose(got, manual.numpy(), rtol=1e-5, atol=1e-6)
