"""Deprecated: grilly v1's ``_bridge``. Use ``grilly`` itself.

v1's bridge was 122 functions reaching the GPU by their own routes. Most of
them were a second, private spelling of something the torch API already
names — ``linear``, ``gelu``, ``softmax``, ``layernorm``, ``rope``,
``conv2d`` — and those are **not** reimplemented here. They raise, pointing
at the torch name, because providing them again would rebuild the parallel
framework the rewrite exists to remove (CLAUDE.md §1, §5).

What is here is the part that was never a torch op: the VSA algebras —
sparse block codes, packed bits, bipolar ±1. They keep v1's exact
signatures, flat ``(batch, num_blocks * block_size)`` float32 for the block
codes, uint32 words for the packed ones, int8 for the bipolar ones, numpy
in and numpy out — so a caller that only imports this keeps working, and
now runs on the GPU.

A capability check like v1's own ``hasattr(_bridge, "blockcode_bind")``
still answers True, and now it means what it says.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly import vsa
from grilly._v1 import moved, v1_module_getattr
from grilly.vsa import bipolar as _bipolar
from grilly.vsa import packed as _packed

__all__ = [
    "blake3_role", "blockcode_bind", "blockcode_similarity", "blockcode_unbind",
    "continuous_to_spikes", "ewc_penalty", "fisher_info_update", "hebbian_learning",
    "natural_gradient", "oja_learning",
    "nlms_predict", "nlms_update", "place_cell", "spikes_to_continuous", "time_cell",
    "whitening_transform",
    "hamming_topk", "hdc_bind_packed", "hdc_bundle_packed", "hdc_permute_packed",
    "hdc_similarity_packed", "is_available", "vsa_bind", "vsa_bitpack",
    "vsa_bundle", "vsa_unbind",
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


# ── bipolar hypervectors ─────────────────────────────────────────────────
#
# These four returned **None in every case** in v1, and not because a GPU
# was missing: the Python wrappers passed a device handle that the compiled
# functions do not take (`_core.vsa_bind(dev, a, b)` against a two-argument
# binding), and `_core.vsa_unbind` does not exist at all. Every call raised
# inside the try, was swallowed, and came back None — v1's "declined, fall
# through" signal, permanently on (CLAUDE.md §8.4).
#
# So these now work where they never did, which cannot break a caller: one
# that handled None still handles a value.
#
# The semantics are the compiled ones, checked against v1's extension
# directly rather than against its Python docstrings, which say "circular
# convolution" and are wrong: it is an element-wise product on bipolar int8.


def _bipolar_in(x):
    return grilly.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def _bipolar_out(t):
    return t.numpy().astype(np.int8)


def vsa_bind(a, b):
    """Bipolar binding: element-wise product. int8 out, as v1's C++ gave."""
    moved("_bridge.vsa_bind", "grilly.vsa.bipolar.bind")
    return _bipolar_out(_bipolar.bind(_bipolar_in(a), _bipolar_in(b)))


def vsa_unbind(composite, key):
    """Bipolar unbinding, which is binding again."""
    moved("_bridge.vsa_unbind", "grilly.vsa.bipolar.unbind")
    return _bipolar_out(_bipolar.unbind(_bipolar_in(composite), _bipolar_in(key)))


def vsa_bundle(a, b):
    """Bipolar bundling: majority vote. v1's Python took two vectors where
    its C++ took a list; this keeps the Python signature."""
    moved("_bridge.vsa_bundle", "grilly.vsa.bipolar.bundle")
    return _bipolar_out(_bipolar.bundle([_bipolar_in(a), _bipolar_in(b)]))


def vsa_bitpack(x):
    """Bipolar int8 to packed uint32 words: ``+1`` sets the bit."""
    moved("_bridge.vsa_bitpack", "grilly.vsa.bipolar.pack")
    return _words_out(_bipolar.pack(_bipolar_in(x)))


def blake3_role(seed_string, dim):
    """A deterministic bipolar role vector for a string, int8.

    Byte-for-byte v1's, which was checked against its compiled
    ``blake3_role`` rather than assumed: the same BLAKE3 stream under the
    domain ``grilly.cubemind``.
    """
    moved("_bridge.blake3_role", "grilly.vsa.bipolar.from_key")
    return _bipolar.from_key(str(seed_string), int(dim), domain="grilly.cubemind").numpy().astype(np.int8)


# ── hippocampal cells, and the spike bridge ──────────────────────────────
#
# v1's functional layer called these and fell through to numpy when the
# bridge declined, which it always did: the compiled extension has no
# place_cell or time_cell. They run on the GPU here.


def place_cell(agent_position, field_centers, field_width=1.0, max_rate=20.0, baseline_rate=0.1):
    """Place-cell firing rates, ``(n_neurons,)`` or ``(batch, n_neurons)``."""
    from grilly.functional import cells as _cells

    return _cells.place_cell(agent_position, field_centers, field_width, max_rate, baseline_rate)


def time_cell(current_time, preferred_times, time_constant=1.0, max_rate=15.0,
              baseline_rate=0.1, membrane_state=None):
    """``(firing_rates, membrane_state)``. v1 spelled the width
    ``time_constant`` here and ``temporal_width`` one layer up."""
    from grilly.functional import cells as _cells

    return _cells.time_cell(current_time, preferred_times, time_constant, max_rate,
                            baseline_rate, membrane_state)


def continuous_to_spikes(continuous, num_timesteps=10, encoding_type=0,
                         projection_weights=None, projection_bias=None):
    """Rate-code values into spike trains."""
    from grilly.functional import bridge as _bridge_fn

    return _bridge_fn.continuous_to_spikes(continuous, num_timesteps, encoding_type,
                                           projection_weights, projection_bias)


def spikes_to_continuous(spikes, encoding_type=0, time_window=5, temporal_weights=None,
                         projection_weights=None, projection_bias=None):
    """Decode spike trains back to values."""
    from grilly.functional import bridge as _bridge_fn

    return _bridge_fn.spikes_to_continuous(spikes, encoding_type, time_window, temporal_weights,
                                           projection_weights, projection_bias)


# ── learning rules that are not gradient descent ─────────────────────────


def fisher_info_update(gradients, fisher, momentum=0.9, use_ema=True, reset=False):
    """Update a running diagonal Fisher estimate. v1 spelled it
    ``fisher_info_update`` here and ``fisher_info`` one layer up."""
    from grilly.functional import learning as _learning

    return _learning.fisher_info(gradients, fisher, momentum, use_ema, reset)


def ewc_penalty(current_params, important_params, fisher, lambda_ewc=1.0):
    """``0.5 * lambda * sum(F * (theta - theta_old) ** 2)``."""
    from grilly.functional import learning as _learning

    return _learning.ewc_penalty(current_params, important_params, fisher, lambda_ewc)


def natural_gradient(gradients, fisher, eps=1e-8):
    """``gradients / (fisher + eps)``."""
    from grilly.functional import learning as _learning

    return _learning.natural_gradient(gradients, fisher, eps)


def nlms_predict(x, w, bias=0.0):
    """``x . w + bias``."""
    from grilly.functional import learning as _learning

    return _learning.nlms_predict(x, w, bias)


def nlms_update(x, y_true, w, bias, mu=0.5, eps=1e-6):
    """One NLMS step: ``(weights, bias)``."""
    from grilly.functional import learning as _learning

    return _learning.nlms_update(x, y_true, w, bias, mu, eps)


def whitening_transform(data, mean=None, std=None):
    """``(whitened, mean, std)``."""
    from grilly.functional import learning as _learning

    return _learning.whitening_transform(data, mean, std)


# ── plasticity ───────────────────────────────────────────────────────────


def hebbian_learning(pre, post, weights, batch_size=1, time_steps=1,
                     learning_rate=0.01, weight_decay=0.0):
    """``W + lr * <post outer pre> - decay * W``, numpy in and numpy out.

    v1's rule, unchanged: its shader's arithmetic was right.
    """
    moved("_bridge.hebbian_learning", "grilly.snn.plasticity.hebbian_update")
    from grilly.snn import plasticity as _p

    shape = (int(batch_size), int(time_steps), -1)
    return _p.hebbian_update(
        grilly.from_numpy(np.ascontiguousarray(pre, dtype=np.float32).reshape(shape)),
        grilly.from_numpy(np.ascontiguousarray(post, dtype=np.float32).reshape(shape)),
        grilly.from_numpy(np.ascontiguousarray(weights, dtype=np.float32)),
        learning_rate, weight_decay,
    ).numpy()


def oja_learning(memories, inputs, num_vectors, dim, eta=0.01):
    """``m + eta * y * (x - y * m)`` with ``y = <m, x>``, per vector.

    This **never ran** in v1: `oja_learning` is not in its compiled
    extension at all, so the wrapper's call raised and the function
    returned None every time. It works here, which cannot break a caller
    that was handling None.
    """
    moved("_bridge.oja_learning", "grilly.snn.plasticity.oja_update")
    from grilly.snn import plasticity as _p

    shape = (int(num_vectors), int(dim))
    return _p.oja_update(
        grilly.from_numpy(np.ascontiguousarray(memories, dtype=np.float32).reshape(shape)),
        grilly.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32).reshape(shape)),
        eta,
    ).numpy().reshape(np.shape(memories))


_STDP = (
    "grilly.snn.plasticity.STDPLearner, which is real STDP. v1's rule was not: its "
    "shader computed `ltp += pre_t * post_t` and `ltd += post_t * pre_t`, the same "
    "product twice, so the update reduced to (lr_pot - lr_dep) * pre_t * post_t and "
    "reversing the spike order could not change its sign. If the Hebbian behaviour is "
    "what you actually wanted, it is grilly.snn.plasticity.hebbian_update"
)


# ── MoQE, and why it is not here ─────────────────────────────────────────
#
# v1's moqe_* were a numpy sketch of a kernel it never built: quantize the
# activations block-wise too, so the dot product runs in integers. The
# docstrings name shaders (moqe-fused-gemv-dp4a.glsl) the Python never
# calls.
#
# It was measured before being ported, and the measurement says not to
# (tools/int8_linear.py, docs/inference.md). At one row -- decode -- the
# existing int8 weight-only linear reads Qwen2.5-0.5B's down_proj at 377
# GB/s against this card's 432 GB/s of spec bandwidth. It is memory-bound
# at 87% of peak. Quantizing the activations as well reads exactly the same
# int8 weight bytes and only changes the arithmetic, which is not the
# bottleneck, so W8A8 cannot make a decode matmul faster here.
#
# Adding it anyway would put a second route to the GPU beside a path that
# is already near the hardware limit (CLAUDE.md 5).

_MOQE = (
    "grilly.infer.quantization.Int8WeightOnlyConfig, through grilly.nn.Linear. v1's "
    "moqe_* were numpy sketches of a fused W8A8 GEMV that was never built, and the "
    "measurement says it would not pay: at decode the int8 linear already runs at 377 "
    "GB/s of this card's 432 GB/s, so it is memory-bound and quantizing activations "
    "reads the same weight bytes (tools/int8_linear.py, docs/inference.md)"
)


# ── everything v1 had that the torch API already names ───────────────────

_REPLACED = {
    "stdp_learning": _STDP,
    "moqe_dynamic_quantize": _MOQE,
    "moqe_fused_gemv": _MOQE,
    "moqe_route_and_gemv": _MOQE,
    "ternary_gemm": _MOQE,
    "pack_ternary": _MOQE,
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
