"""GPTQ: quantize a weight so its errors cancel in the layer's *output*.

Round-to-nearest makes each weight's own error as small as it can be, and
`tools/int4_accuracy.py` measures where that ends: fitting an optimal
16-level codebook to every group of 128 — which cannot be built — still
leaves int4 nine times worse than int8. That bound applies to every scheme
that rounds each weight on its own.

GPTQ (Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers") leaves that family. It quantizes a
row's columns in order and, after each one, **adjusts the columns not yet
quantized** so that the error just made is cancelled in the output, in the
metric of the layer's own inputs:

    H = 2 X^T X          the Hessian of ||X (W - W')^T||^2 for that layer
    w_j  -> q_j          quantize column j
    e    = (w_j - dequant(q_j)) / [H^-1]_jj
    W[:, j+1:] -= e (x) [H^-1]_{j, j+1:}

So the weights move *away* from their originals on purpose, because what
matters is what the layer computes, not what the weights are. `X` comes
from running the model on calibration text.

This module is the numerics; `grilly.infer.calibration` runs a model to
collect the Hessians and applies it layer by layer.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gptq_quantize", "quantize_group_int4", "rtn_quantize"]

LEVELS = 16  # int4


def _group_parameters(block: np.ndarray):
    """torchao's asymmetric int4 parameters for a group: (scale, min)."""
    lo = block.min(axis=1)
    hi = block.max(axis=1)
    scale = np.maximum((hi - lo) / (LEVELS - 1), np.finfo(np.float32).eps)
    return scale, lo


def quantize_group_int4(values: np.ndarray, scale: np.ndarray, lo: np.ndarray):
    """Levels and their dequantized values, for one column of each row."""
    level = np.clip(np.round((values - lo) / scale), 0, LEVELS - 1)
    return level, level * scale + lo


def rtn_quantize(weight: np.ndarray, group_size: int = 128):
    """Plain round-to-nearest, the baseline: what `Int4Tensor.from_hp` does,
    in numpy. Returns (levels, scale, zero)."""
    rows, k = weight.shape
    groups = (k + group_size - 1) // group_size
    levels = np.zeros((rows, k), dtype=np.uint8)
    scales = np.zeros((rows, groups), dtype=np.float32)
    zeros = np.zeros((rows, groups), dtype=np.float32)
    for g in range(groups):
        start, end = g * group_size, min((g + 1) * group_size, k)
        scale, lo = _group_parameters(weight[:, start:end])
        level, _ = quantize_group_int4(weight[:, start:end], scale[:, None], lo[:, None])
        levels[:, start:end] = level.astype(np.uint8)
        scales[:, g], zeros[:, g] = scale, lo + 8.0 * scale
    return levels, scales, zeros


def _inverse_cholesky(hessian: np.ndarray, percdamp: float):
    """The upper triangular `U` with `U^T U = H^-1`, damped as GPTQ does.

    Columns whose input was always zero carry no information; the reference
    sets their Hessian diagonal to 1 and zeroes the weight, which this
    reports so the caller can zero those columns too.
    """
    h = np.array(hessian, dtype=np.float64, copy=True)
    dead = np.diag(h) == 0.0
    h[dead, dead] = 1.0
    damp = percdamp * np.mean(np.diag(h))
    h[np.diag_indices_from(h)] += damp
    lower = np.linalg.cholesky(h)                      # H = L L^T
    inverse = np.linalg.inv(lower)
    inverse = inverse.T @ inverse                      # H^-1
    upper = np.linalg.cholesky(inverse).T              # U^T U = H^-1
    return upper, dead


def gptq_quantize(weight: np.ndarray, hessian: np.ndarray, group_size: int = 128,
                  blocksize: int = 128, percdamp: float = 0.01, act_order: bool = False,
                  static_groups: bool | None = None):
    """Quantize `weight` (out, in) to int4 against the calibration Hessian
    `hessian` (in, in), returning (levels, scales, zeros) in the layout
    `Int4Tensor` uses.

    With `hessian` proportional to the identity this is exactly
    round-to-nearest: the inverse Cholesky is then diagonal, so no error is
    propagated anywhere. `tests/python/test_gptq.py` asserts that.

    `act_order` quantizes the columns the calibration data excited most
    first, so their error is compensated by everything that follows. It
    requires `static_groups`, and defaults it on.

    `static_groups` fixes every group's scale and zero point from the
    original weights, before any compensation. Without it, a group's
    parameters are computed when the loop reaches it, from weights earlier
    columns have already pushed their error into — which is more accurate,
    and is why plain GPTQ leaves it off. With `act_order` it is not
    optional: the groups must stay the contiguous runs of 128 that
    `g2-linear-q4` reads, and the visiting order no longer matches them.
    AutoGPTQ spells these `desc_act` and `static_groups` too.

    The two are worth separating, because they do not always pull the same
    way (docs/inference.md: ordering helped Qwen2.5-0.5B and hurt
    Qwen3-1.7B).
    """
    rows, k = weight.shape
    if hessian.shape != (k, k):
        raise ValueError(f"hessian must be ({k}, {k}) for a weight with {k} input features, "
                         f"got {tuple(hessian.shape)}")
    if group_size <= 0 or k % group_size:
        raise ValueError(f"in_features {k} must be a whole number of groups of {group_size}")

    if static_groups is None:
        static_groups = act_order
    if act_order and not static_groups:
        raise ValueError("act_order needs static_groups: reordering the columns leaves the "
                         "groups scattered, which the int4 kernel cannot read")
    work = np.array(weight, dtype=np.float32, copy=True)
    groups = k // group_size
    scales = np.zeros((rows, groups), dtype=np.float32)
    zeros = np.zeros((rows, groups), dtype=np.float32)
    static = None
    order = permutation = None

    if static_groups:
        # every group's parameters, from the weights before any compensation
        static = [_group_parameters(work[:, g * group_size:(g + 1) * group_size]) for g in range(groups)]
        for g, (scale, lo) in enumerate(static):
            scales[:, g], zeros[:, g] = scale, lo + 8.0 * scale
    if act_order:
        # visit the columns the calibration data excited most, first
        order = np.argsort(-np.diag(hessian))
        permutation = np.argsort(order)
        work = work[:, order]
        hessian = hessian[order][:, order]

    upper, dead = _inverse_cholesky(hessian, percdamp)
    work[:, dead] = 0.0
    levels = np.zeros((rows, k), dtype=np.uint8)
    scale = lo = None

    def parameters(column_index):
        """The group parameters for the column sitting at `column_index`
        now: its original group when reordered, else recomputed from the
        weights as they stand, after earlier columns pushed error here."""
        nonlocal scale, lo
        if static_groups:
            original = order[column_index] if act_order else column_index
            scale, lo = static[original // group_size]
        elif column_index % group_size == 0:
            g = column_index // group_size
            scale, lo = _group_parameters(work[:, column_index:column_index + group_size])
            scales[:, g], zeros[:, g] = scale, lo + 8.0 * scale
        return scale, lo

    for start in range(0, k, blocksize):
        end = min(start + blocksize, k)
        errors = np.zeros((rows, end - start), dtype=np.float32)
        for j in range(start, end):
            scale, lo = parameters(j)
            column = work[:, j]
            level, dequantized = quantize_group_int4(column, scale, lo)
            levels[:, j] = level.astype(np.uint8)
            error = (column - dequantized) / upper[j, j]
            errors[:, j - start] = error
            if j + 1 < end:  # push it into the rest of this block
                work[:, j + 1:end] -= np.outer(error, upper[j, j + 1:end]).astype(np.float32)
        if end < k:  # and then into every column after the block, at once
            work[:, end:] -= (errors @ upper[start:end, end:]).astype(np.float32)
    if act_order:
        levels = levels[:, permutation]
    return levels, scales, zeros


def dequantize(levels: np.ndarray, scales: np.ndarray, zeros: np.ndarray, group_size: int = 128):
    """The float32 weight those levels stand for."""
    rows, k = levels.shape
    groups = scales.shape[1]
    per_column = np.repeat(scales, group_size, axis=1)[:, :k]
    mins = np.repeat(zeros - 8.0 * scales, group_size, axis=1)[:, :k]
    assert per_column.shape == levels.shape and groups == (k + group_size - 1) // group_size
    return (levels.astype(np.float32) * per_column + mins).astype(np.float32)
