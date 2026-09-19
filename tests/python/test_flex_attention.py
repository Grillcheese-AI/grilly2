"""torch.nn.attention.flex_attention (dense) against torch 2.10's own (CPU).

Tolerance: outputs, lse and gradients within rtol 1e-5, atol 1e-6, as for
scaled_dot_product_attention (the same math: matmuls of depth <= 16 and a
softmax). Masks are exact.
"""

import warnings

import numpy as np
import pytest
import torch
from torch.nn.attention import flex_attention as tfa

import grilly
from grilly.nn.attention import flex_attention as gfa

RNG = np.random.default_rng(31)
B, H, L, S, E = 2, 4, 12, 12, 8

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")  # torch: "called without torch.compile()"


def rand(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def causal(b, h, q, k):
    return q >= k


def window(b, h, q, k):  # CubbyLLM's _window_block_mask
    return (q >= k) & (q - k < 3)


def per_batch_head(b, h, q, k):
    return (k <= q + b) & ((h % 2 == 0) | (k % 2 == 0))


def compare(make_mask=None, make_score=None, mask_bh=(None, None), q_shape=(B, H, L, E), kv_shape=(B, H, S, E),
            extra=None, **kw):
    """Run both libraries on the same inputs; compare out, lse, all grads.
    make_mask / make_score take the library (for captured tensors)."""
    q, k, v = rand(*q_shape), rand(*kv_shape), rand(*kv_shape)
    results = []
    for lib, fa in ((grilly, gfa), (torch, tfa)):
        Q, K, V = (lib.tensor(a, requires_grad=True) for a in (q, k, v))
        cap = lib.tensor(extra, requires_grad=True) if extra is not None else None
        bm = None
        if make_mask is not None:
            bm = fa.create_block_mask(make_mask(lib), *mask_bh, q_shape[2], kv_shape[2], device="cpu")
        sm = make_score(lib, cap) if make_score is not None else None
        out, aux = fa.flex_attention(Q, K, V, score_mod=sm, block_mask=bm, return_aux=fa.AuxRequest(lse=True), **kw)
        w = lib.tensor(np.linspace(-1, 1, out.numel(), dtype=np.float32).reshape(tuple(out.shape)))
        (out * w).sum().backward()
        grads = [t.grad.numpy() for t in (Q, K, V)] + ([cap.grad.numpy()] if cap is not None else [])
        results.append((out.detach().numpy(), aux.lse.detach().numpy(), grads))
    (og, lg, gg), (ot, lt, gt) = results
    np.testing.assert_allclose(og, ot, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(lg, lt, rtol=1e-5, atol=1e-6)
    for a, b in zip(gg, gt):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    return og, lg


@pytest.mark.parametrize("mod,bh", [(causal, (None, None)), (window, (None, None)), (per_batch_head, (B, H))])
def test_block_masks_match_torch(mod, bh):
    compare(lambda lib: mod, mask_bh=bh)


def test_combined_masks_match_torch():
    for lib, fa in ((grilly, gfa), (torch, tfa)):
        pass
    compare(lambda lib: (gfa if lib is grilly else tfa).and_masks(causal, lambda b, h, q, k: q - k < 4))
    compare(lambda lib: (gfa if lib is grilly else tfa).or_masks(window, lambda b, h, q, k: k == 0))
    compare(lambda lib: (gfa if lib is grilly else tfa).noop_mask)


def test_document_mask_with_a_captured_tensor():
    doc = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])

    def make(lib):
        ids = lib.tensor(doc)
        return lambda b, h, q, k: (ids[q] == ids[k]) & (q >= k)
    compare(make)


@pytest.mark.parametrize("name", ["relative_bias", "alibi", "softcap"])
def test_score_mods_match_torch(name):
    if name == "relative_bias":  # a learned per-head bias: its gradient is compared too
        make = lambda lib, cap: (lambda s, b, h, q, k: s + cap[h] * (q - k))  # noqa: E731
        compare(lambda lib: causal, make, extra=rand(H))
    elif name == "alibi":
        slopes = np.exp2(-np.arange(1, H + 1, dtype=np.float32))
        make = lambda lib, cap: (lambda s, b, h, q, k: s - lib.tensor(slopes)[h] * (q - k).abs())  # noqa: E731
        compare(None, make)
    else:
        make = lambda lib, cap: (lambda s, b, h, q, k: 5.0 * (s / 5.0).tanh())  # noqa: E731
        compare(lambda lib: window, make)


def test_scale_gqa_and_uneven_lengths_match_torch():
    compare(lambda lib: causal, scale=0.3)
    compare(lambda lib: causal, q_shape=(B, 8, L, E), kv_shape=(B, 2, S, E), enable_gqa=True)
    compare(lambda lib: lambda b, h, q, k: k <= q + 4, q_shape=(B, H, 5, E), kv_shape=(B, H, 9, E))


def test_rows_with_no_kept_key_give_zero_and_minus_inf_lse():
    out, lse = compare(lambda lib: lambda b, h, q, k: (q != 3) & (k <= q))
    assert np.all(out[:, :, 3] == 0) and np.all(np.isneginf(lse[:, :, 3]))


def test_create_mask_matches_torch():
    for mod, bh in ((causal, (None, None)), (per_batch_head, (B, H))):
        got = gfa.create_mask(mod, *bh, L, S)
        want = tfa.create_mask(mod, *bh, L, S, device="cpu")
        assert got.dtype is grilly.bool
        np.testing.assert_array_equal(got.numpy(), want.numpy())


def test_return_lse_is_deprecated_like_torch():
    q = grilly.tensor(rand(1, 1, 4, 8))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, lse = gfa.flex_attention(q, q, q, return_lse=True)
    assert any(issubclass(w.category, FutureWarning) for w in caught)
    assert tuple(lse.shape) == (1, 1, 4)


def test_max_scores():
    """torch refuses max_scores on CPU; grilly returns the row max of the
    (score_mod-ed, masked) scores."""
    q, k = rand(1, 2, 5, 8), rand(1, 2, 5, 8)
    bm = gfa.create_block_mask(causal, None, None, 5, 5)
    _, aux = gfa.flex_attention(grilly.tensor(q), grilly.tensor(k), grilly.tensor(k), block_mask=bm,
                                return_aux=gfa.AuxRequest(max_scores=True))
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(8)
    scores = np.where(np.tril(np.ones((5, 5), bool)), scores, -np.inf)
    np.testing.assert_allclose(aux.max_scores.numpy(), scores.max(-1), rtol=1e-5, atol=1e-6)
    assert aux.lse is None


def test_errors_match_torch():
    q = rand(1, 2, 4, 8)
    for lib, fa in ((torch, tfa), (grilly, gfa)):
        t = lib.tensor(q)
        with pytest.raises(ValueError, match="block_mask was created for"):
            fa.flex_attention(t, t, t, block_mask=fa.create_block_mask(causal, None, None, 6, 6, device="cpu"))
        with pytest.raises(ValueError, match="enable_gqa=True"):
            fa.flex_attention(t, lib.tensor(rand(1, 1, 4, 8)), lib.tensor(rand(1, 1, 4, 8)))


def test_cubbyllm_flex_path_equals_the_masked_sdpa_path():
    """What CubbyLLM's test_flex_attention_path_matches_the_masked_sdpa_reference checks."""
    q, k, v = (grilly.tensor(rand(B, H, 16, E)) for _ in range(3))
    bm = gfa.create_block_mask(window, B=None, H=None, Q_LEN=16, KV_LEN=16, device=str(q.device))
    i = grilly.arange(16)
    keep = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < 3)
    ref = grilly.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep)
    np.testing.assert_allclose(gfa.flex_attention(q, k, v, block_mask=bm).numpy(), ref.numpy(), rtol=0, atol=1e-6)
