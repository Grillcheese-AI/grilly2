"""Deprecated: grilly v1's ``_bridge``. Use ``grilly`` itself.

v1's bridge was 122 functions reaching the GPU by their own routes. Most of
them were a second, private spelling of something the torch API already
names — ``linear``, ``gelu``, ``softmax``, ``layernorm``, ``rope``,
``conv2d`` — and those are **not** reimplemented here. They raise, pointing
at the torch name, because providing them again would rebuild the parallel
framework the rewrite exists to remove (CLAUDE.md §1, §5).

What is here is the part that was never a torch op: the two VSA algebras.
They keep v1's exact signatures — flat ``(batch, num_blocks * block_size)``
float32 for the block codes, uint32 words for the packed ones, numpy in and
numpy out — so a caller that only imports this keeps working, and now runs
on the GPU.

A capability check like v1's own ``hasattr(_bridge, "blockcode_bind")``
still answers True, and now it means what it says.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly import vsa
from grilly._v1 import moved, v1_module_getattr
from grilly.vsa import packed as _packed

__all__ = [
    "blockcode_bind", "blockcode_similarity", "blockcode_unbind", "hamming_topk",
    "hdc_bind_packed", "hdc_bundle_packed", "hdc_permute_packed", "hdc_similarity_packed",
    "is_available",
]


def is_available() -> bool:
    """Whether the GPU is usable. v1 answered False when its extension was
    missing; grilly2 has no separate extension to miss."""
    try:
        grilly.zeros(1)
        return True
    except Exception:
        return False


# ── sparse block codes ───────────────────────────────────────────────────
#
# v1's bridge took vectors FLAT, (batch, num_blocks * block_size), with the
# block shape passed alongside; grilly.vsa takes them as (..., k, l). The
# reshape is the whole difference, and it is the trap in CLAUDE.md §8.2:
# these are batches of ONE-HOT vectors, not dense ones.


def _blocks_in(x, num_blocks: int, block_size: int):
    a = np.ascontiguousarray(x, dtype=np.float32)
    squeezed = a.ndim == 1
    a = np.atleast_2d(a)
    return grilly.from_numpy(a.reshape(a.shape[0], num_blocks, block_size)), squeezed


def _blocks_out(t, squeezed: bool):
    a = t.numpy().reshape(t.shape[0], -1)
    return a[0] if squeezed else a


def blockcode_bind(a, b, num_blocks, block_size):
    """Per-block circular convolution of two block-code vectors.

    ``a``, ``b``: float32, ``(batch, num_blocks * block_size)`` or flat.
    Returns the same shape.
    """
    moved("_bridge.blockcode_bind", "grilly.vsa.bind")
    ta, squeezed = _blocks_in(a, num_blocks, block_size)
    tb, _ = _blocks_in(b, num_blocks, block_size)
    return _blocks_out(vsa.bind(ta, tb), squeezed)


def blockcode_unbind(composite, key, num_blocks, block_size):
    """Per-block circular correlation: recovers the other factor."""
    moved("_bridge.blockcode_unbind", "grilly.vsa.unbind")
    tc, squeezed = _blocks_in(composite, num_blocks, block_size)
    tk, _ = _blocks_in(key, num_blocks, block_size)
    return _blocks_out(vsa.unbind(tc, tk), squeezed)


def blockcode_similarity(query, codebook, num_blocks, block_size):
    """``(1 / k) * sum(query * entry)`` for every codebook entry, as (n,)."""
    moved("_bridge.blockcode_similarity", "grilly.vsa.similarity_batch")
    q = np.ascontiguousarray(query, dtype=np.float32).reshape(num_blocks, block_size)
    cb = np.ascontiguousarray(codebook, dtype=np.float32)
    cb = cb.reshape(-1, num_blocks, block_size)
    return vsa.similarity_batch(grilly.from_numpy(q), grilly.from_numpy(cb)).numpy()


# ── packed binary hypervectors ───────────────────────────────────────────
#
# A different algebra, on uint32 words. v1 never wired these to the GPU
# ("GPU path not yet wired into grilly_core"); they run on one now.


def _words_in(x):
    return grilly.from_numpy(np.ascontiguousarray(x, dtype=np.uint32).view(np.int32))


def _words_out(t):
    return t.numpy().astype(np.int32).view(np.uint32)


def hdc_bind_packed(a, b):
    """XOR of two packed binary hypervectors. Its own inverse."""
    moved("_bridge.hdc_bind_packed", "grilly.vsa.packed.bind")
    return _words_out(_packed.bind(_words_in(a), _words_in(b)))


def hdc_bundle_packed(vectors, words_per_vec):
    """Majority vote over ``(num_vectors, words_per_vec)``, as ``(words_per_vec,)``."""
    moved("_bridge.hdc_bundle_packed", "grilly.vsa.packed.bundle")
    v = np.ascontiguousarray(vectors, dtype=np.uint32).reshape(-1, words_per_vec)
    return _words_out(_packed.bundle(_words_in(v)))


def hdc_similarity_packed(query, codebook, dim):
    """``1 - hamming / dim`` against every codebook entry, float32 ``(n,)``."""
    moved("_bridge.hdc_similarity_packed", "grilly.vsa.packed.similarity_batch")
    q = np.ascontiguousarray(query, dtype=np.uint32).reshape(-1)
    cb = np.ascontiguousarray(codebook, dtype=np.uint32).reshape(-1, q.shape[0])
    return _packed.similarity_batch(_words_in(q), _words_in(cb), dim).numpy()


def hdc_permute_packed(data, words_per_vec, shift):
    """Cyclic bit rotation within ``words_per_vec * 32`` bits."""
    moved("_bridge.hdc_permute_packed", "grilly.vsa.packed.permute")
    d = np.ascontiguousarray(data, dtype=np.uint32).reshape(-1)
    return _words_out(_packed.permute(_words_in(d), int(shift), words_per_vec * 32))


def hamming_topk(query, cache, dim, k=10):
    """The k nearest cache entries by Hamming distance.

    v1's dict: ``indices`` int32, ``distances`` int32 ascending,
    ``similarities`` float32 descending.
    """
    moved("_bridge.hamming_topk", "grilly.vsa.packed.topk")
    q = np.ascontiguousarray(query, dtype=np.uint32).reshape(-1)
    cb = np.ascontiguousarray(cache, dtype=np.uint32).reshape(-1, q.shape[0])
    indices, distances, sims = _packed.topk(_words_in(q), _words_in(cb), k, dim)
    return {
        "indices": indices.numpy().astype(np.int32),
        "distances": distances.numpy().astype(np.int32),
        "similarities": sims.numpy().astype(np.float32),
    }


# ── everything v1 had that the torch API already names ───────────────────

_REPLACED = {
    "linear": "grilly.nn.Linear, or grilly.nn.functional.linear",
    "linear_backward": "autograd — call .backward() on the loss",
    "gelu": "grilly.nn.functional.gelu",
    "gelu_backward": "autograd — call .backward() on the loss",
    "relu": "grilly.nn.functional.relu",
    "relu_backward": "autograd — call .backward() on the loss",
    "silu": "grilly.nn.functional.silu",
    "silu_backward": "autograd — call .backward() on the loss",
    "tanh": "grilly.tanh",
    "tanh_backward": "autograd — call .backward() on the loss",
    "softmax": "grilly.nn.functional.softmax",
    "softmax_backward": "autograd — call .backward() on the loss",
    "mf_softmax": "grilly.nn.functional.softmax",
    "mf_sigmoid": "grilly.sigmoid",
    "mf_softplus": "grilly.nn.functional.softplus",
    "layernorm": "grilly.nn.LayerNorm, or grilly.nn.functional.layer_norm",
    "layernorm_backward": "autograd — call .backward() on the loss",
    "rmsnorm": "grilly.nn.RMSNorm",
    "fused_layernorm_linear": "grilly.nn.LayerNorm then grilly.nn.Linear; grilly.compile fuses them",
    "fused_mlp_gelu": "grilly.nn.Linear and grilly.nn.functional.gelu; grilly.compile fuses them",
    "batchnorm2d_forward": "grilly.nn.BatchNorm2d",
    "dropout": "grilly.nn.Dropout",
    "embedding_lookup": "grilly.nn.Embedding",
    "conv2d": "grilly.nn.Conv2d",
    "conv2d_3x3_gelu": "grilly.nn.Conv2d then grilly.nn.functional.gelu",
    "conv2d_backward_input": "autograd — call .backward() on the loss",
    "conv2d_backward_weight": "autograd — call .backward() on the loss",
    "maxpool2d": "grilly.nn.MaxPool2d",
    "maxpool2x2": "grilly.nn.MaxPool2d",
    "avgpool2d": "grilly.nn.AvgPool2d",
    "adaptive_avgpool_3x3": "grilly.nn.AdaptiveAvgPool2d",
    "mean_pool": "Tensor.mean",
    "attention_scores": "grilly.nn.functional.scaled_dot_product_attention",
    "attention_scores_softmax_output": "grilly.nn.functional.scaled_dot_product_attention",
    "attention_output": "grilly.nn.functional.scaled_dot_product_attention",
    "attention_concat_heads": "Tensor.transpose then Tensor.reshape",
    "attention_mask": "the is_causal or attn_mask argument of scaled_dot_product_attention",
    "flash_attention2": "grilly.nn.functional.scaled_dot_product_attention",
    "rope": "grilly.infer.models' rotary embedding, or apply it with ordinary ops",
    "cross_entropy_loss": "grilly.nn.functional.cross_entropy",
    "cross_entropy_backward": "autograd — call .backward() on the loss",
    "adam_update": "grilly.optim.Adam",
    "adamw_update": "grilly.optim.AdamW",
    "tensor_bmm": "grilly.bmm",
    "tensor_cat": "grilly.cat",
    "tensor_index_select": "grilly.index_select",
    "tensor_transpose": "Tensor.transpose",
    "tensor_where": "grilly.where",
    "create_kv_cache": "grilly.infer.cache.KVCache",
    "kv_cache_append": "grilly.infer.cache.KVCache.append",
    "kv_cache_decode": "grilly.infer.cache.KVCache.attend",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
