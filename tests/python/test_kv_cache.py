"""grilly.infer.cache.KVCache: decode-step attention against a float64
reference and against torch's scaled_dot_product_attention(enable_gqa=True)
over the same keys and values.

Tolerances:
- against float64: |got - ref| <= 1e-5 * |ref| + 1e-6 * max|v|, as in
  crates/grilly-infer/tests/kv_cache.rs. That is a float32 kernel against
  float64: head_dim-term dot products, then compensated sums.
- against torch's float32 SDPA: rtol 1e-5, atol 1e-6 * max|v|. That is case
  (b) in CLAUDE.md §9: torch sums in another order.
"""

import gc

import numpy as np
import pytest
import torch
import torch.nn.functional as TF

import grilly
from grilly.infer.cache import KVCache

RNG = np.random.default_rng(5)


def f64_attention(q, k, v):
    """q (B, H, 1, D), k/v (B, Hkv, L, D): float64 decode attention with GQA."""
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    rep = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    e = np.exp(s - s.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)) @ v


@pytest.mark.parametrize("batch,heads,kv_heads,dim", [(1, 14, 2, 64), (2, 6, 2, 64), (1, 8, 8, 128), (3, 3, 1, 80)])
def test_decode_steps_match_float64_and_torch(batch, heads, kv_heads, dim):
    cache = KVCache(2, batch, kv_heads, dim, 300)
    ks = [np.zeros((batch, kv_heads, 0, dim), np.float32) for _ in range(2)]
    vs = [k.copy() for k in ks]
    worst = 0.0
    for step in (37, 1, 1, 16, 1, 200):
        for layer in range(2):
            kn = RNG.standard_normal((batch, kv_heads, step, dim)).astype(np.float32)
            vn = RNG.standard_normal((batch, kv_heads, step, dim)).astype(np.float32)
            cache.append(layer, grilly.from_numpy(kn), grilly.from_numpy(vn))
            ks[layer] = np.concatenate([ks[layer], kn], 2)
            vs[layer] = np.concatenate([vs[layer], vn], 2)
            assert cache.get_seq_length(layer) == ks[layer].shape[2]
            q = RNG.standard_normal((batch, heads, 1, dim)).astype(np.float32)
            got = cache.attend(layer, grilly.from_numpy(q)).numpy()
            ref = f64_attention(q, ks[layer], vs[layer])
            vmax = np.abs(vs[layer]).max()
            ratio = np.abs(got - ref) / (1e-5 * np.abs(ref) + 1e-6 * vmax)
            worst = max(worst, float(ratio.max()))
            assert (ratio <= 1).all()
            want = TF.scaled_dot_product_attention(torch.from_numpy(q), torch.from_numpy(ks[layer]),
                                                   torch.from_numpy(vs[layer]), enable_gqa=True).numpy()
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6 * vmax)
    print(f"({batch}, {heads}, {kv_heads}, {dim}) worst error / bound = {worst:.3f}")


def test_prefill_then_decode_equals_attention_over_the_whole_sequence():
    """A prompt's causal SDPA, then decode steps from the cache, equal causal
    attention over the whole sequence, row by row."""
    batch, heads, kv_heads, dim, prompt, steps = 1, 14, 2, 64, 12, 5
    total = prompt + steps
    q = RNG.standard_normal((batch, heads, total, dim)).astype(np.float32)
    k = RNG.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    v = RNG.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    want = TF.scaled_dot_product_attention(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v),
                                           is_causal=True, enable_gqa=True).numpy()
    cache = KVCache(1, batch, kv_heads, dim, total)
    got = [grilly.nn.functional.scaled_dot_product_attention(
        grilly.from_numpy(q[:, :, :prompt]), grilly.from_numpy(k[:, :, :prompt]), grilly.from_numpy(v[:, :, :prompt]),
        is_causal=True, enable_gqa=True).numpy()]
    cache.append(0, grilly.from_numpy(k[:, :, :prompt]), grilly.from_numpy(v[:, :, :prompt]))
    for t in range(prompt, total):
        cache.append(0, grilly.from_numpy(k[:, :, t:t + 1]), grilly.from_numpy(v[:, :, t:t + 1]))
        got.append(cache.attend(0, grilly.from_numpy(q[:, :, t:t + 1])).numpy())
    np.testing.assert_allclose(np.concatenate(got, 2), want, rtol=1e-5, atol=1e-6 * np.abs(v).max())


def test_memory_is_persistent_and_exact_and_errors_are_clear():
    gc.collect()
    before = grilly._C.submit_stats()["persistent_bytes"]
    cache = KVCache(24, 1, 2, 64, 1000)  # Qwen2.5-0.5B's layout, 1000 positions
    assert cache.nbytes == 24 * 2 * 2 * 1000 * 64 * 4
    assert grilly._C.submit_stats()["persistent_bytes"] - before == cache.nbytes
    kv = grilly.zeros(1, 2, 1, 64)
    with pytest.raises(RuntimeError, match="empty"):
        cache.attend(0, grilly.zeros(1, 14, 1, 64))
    with pytest.raises(RuntimeError, match=r"must be \(1, 2, T, 64\)"):
        cache.append(0, grilly.zeros(1, 3, 1, 64), kv)
    with pytest.raises(IndexError, match="layer 24"):
        cache.append(24, kv, kv)
    small = KVCache(1, 1, 2, 64, 2)
    small.append(0, grilly.zeros(1, 2, 2, 64), grilly.zeros(1, 2, 2, 64))
    with pytest.raises(RuntimeError, match="KV cache full"):
        small.append(0, kv, kv)
    small.reset()
    assert small.get_seq_length() == 0
    del cache, small
    gc.collect()
    assert grilly._C.submit_stats()["persistent_bytes"] == before
