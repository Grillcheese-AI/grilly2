"""torch.nn.attention.flex_attention, dense.

``mask_mod(b, h, q_idx, kv_idx)`` and ``score_mod(score, b, h, q_idx,
kv_idx)`` are called once, on broadcast index tensors (b: (B,1,1,1),
h: (1,H,1,1), q_idx: (1,1,Q,1), kv_idx: (1,1,1,KV)), where torch vmaps them
over scalars. Any mod written with tensor ops works either way; Python
control flow on the index values does not (torch rejects it too).

Without a ``score_mod`` (and head dims <= 128) it runs a fused block-sparse
kernel (g2-flash-fwd): only the 32x32 tiles the mask touches are computed,
and no score matrix reaches memory. A ``score_mod`` runs on the dense path
(the full (Q, KV) score matrix, as torch's uncompiled flex_attention), with
a one-time warning: a Python function cannot be compiled into the kernel.
``BLOCK_SIZE``, ``kernel_options`` and ``_compile`` are accepted and have no
effect.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass

import numpy as np

from ... import Tensor, _ops, _require_tensor, arange, tensor
from .. import functional as F

__all__ = [
    "AuxOutput", "AuxRequest", "BlockMask", "FlexKernelOptions", "and_masks", "create_block_mask",
    "create_mask", "flex_attention", "noop_mask", "or_masks",
]

FlexKernelOptions = dict  # torch's is a TypedDict of backend tuning knobs; none apply here


@dataclass(frozen=True)
class AuxRequest:
    """Which extra outputs ``flex_attention(..., return_aux=...)`` returns."""
    lse: bool = False
    max_scores: bool = False


@dataclass(frozen=True)
class AuxOutput:
    lse: Tensor | None = None
    max_scores: Tensor | None = None


def _indices(B: int, H: int, Q: int, KV: int):
    return (arange(B).reshape(B, 1, 1, 1), arange(H).reshape(1, H, 1, 1),
            arange(Q).reshape(1, 1, Q, 1), arange(KV).reshape(1, 1, 1, KV))


def noop_mask(b, h, q_idx, kv_idx) -> Tensor:
    """Keeps every position."""
    return tensor(True)


def and_masks(*mask_mods):
    """A mask_mod keeping the positions every one of ``mask_mods`` keeps."""
    if not all(callable(m) for m in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def mask(b, h, q_idx, kv_idx):
        result = tensor(True)
        for m in mask_mods:
            result = result & m(b, h, q_idx, kv_idx)
        return result
    return mask


def or_masks(*mask_mods):
    """A mask_mod keeping the positions any one of ``mask_mods`` keeps."""
    if not all(callable(m) for m in mask_mods):
        raise RuntimeError(f"All inputs should be callable mask_mods: {mask_mods}")

    def mask(b, h, q_idx, kv_idx):
        result = tensor(False)
        for m in mask_mods:
            result = result | m(b, h, q_idx, kv_idx)
        return result
    return mask


def create_mask(mod_fn, B, H, Q_LEN: int, KV_LEN: int, device=None) -> Tensor:
    """The dense (B, H, Q_LEN, KV_LEN) bool mask of a mask_mod (B or H None
    means 1, broadcasting over the batch or heads)."""
    from ... import _resolve_device
    _resolve_device(device)
    B, H = B or 1, H or 1
    keep = mod_fn(*_indices(B, H, Q_LEN, KV_LEN))
    keep = _require_tensor("mask_mod result", keep)
    if keep.dtype is not _bool():
        keep = keep != 0
    return keep.expand(B, H, Q_LEN, KV_LEN) if tuple(keep.shape) != (B, H, Q_LEN, KV_LEN) else keep


def _bool():
    from ... import bool as b_
    return b_


# The fused kernels' tile: 32 queries x 32 keys, so one query row's keys in
# a tile are one uint32 of the bit mask.
TILE = 32
_FULL = 1 << 30  # set in a tile-list entry when every position in the tile is kept


class _Tiles:
    """A mask at the fused kernels' tile granularity (int32 tensors on the
    device), for (mb, mh) = (1 or B, 1 or H) mask batches and heads:
      bits      (mb, mh, Q, nk)  bit c of word [.., q, t]: key t*32 + c kept
      kv_count  (mb, mh, nq)     key tiles each query tile touches
      kv_list   (mb, mh, nq, nk) those tiles, ascending, _FULL flagged
      q_count   (mb, mh, nk)     and the transpose, for the backward
      q_list    (mb, mh, nk, nq)
    Built once, on the host, from the dense mask."""

    @classmethod
    def everything(cls, Q: int, KV: int) -> "_Tiles":
        """No mask: every tile full, so the bit mask is never read."""
        t = cls.__new__(cls)
        nq, nk = -(-Q // TILE), -(-KV // TILE)
        t.bits = np.zeros((1, 1, 1, 1), np.int32)
        t.kv_count = np.full((1, 1, nq), nk, np.int32)
        t.kv_list = np.broadcast_to(np.arange(nk, dtype=np.int32) | _FULL, (1, 1, nq, nk)).copy()
        t.q_count = np.full((1, 1, nk), nq, np.int32)
        t.q_list = np.broadcast_to(np.arange(nq, dtype=np.int32) | _FULL, (1, 1, nk, nq)).copy()
        t.shape = (1, 1, nq, nk)
        return t

    def __init__(self, dense: np.ndarray) -> None:
        mb, mh, Q, KV = dense.shape
        nq, nk = -(-Q // TILE), -(-KV // TILE)
        padded = np.zeros((mb, mh, nq * TILE, nk * TILE), bool)
        padded[..., :Q, :KV] = dense
        valid = np.zeros_like(padded)
        valid[..., :Q, :KV] = True
        words = np.packbits(padded[..., :Q, :].reshape(mb, mh, Q, nk, TILE), axis=-1, bitorder="little")
        self.bits = words.view(np.uint32).reshape(mb, mh, Q, nk).view(np.int32)
        blocks = padded.reshape(mb, mh, nq, TILE, nk, TILE)
        touched = blocks.any(axis=(3, 5))                                                    # (mb, mh, nq, nk)
        full = (blocks | ~valid.reshape(blocks.shape)).all(axis=(3, 5)) & touched
        self.kv_count, self.kv_list = self._lists(touched, full)
        self.q_count, self.q_list = self._lists(touched.swapaxes(-1, -2), full.swapaxes(-1, -2))
        self.shape = (mb, mh, nq, nk)

    @staticmethod
    def _lists(touched: np.ndarray, full: np.ndarray):
        count = touched.sum(-1).astype(np.int32)
        order = np.argsort(~touched, axis=-1, kind="stable").astype(np.int32)  # touched first, ascending
        flags = np.take_along_axis(full, order, -1)
        entries = np.where(np.arange(order.shape[-1]) < count[..., None], order | (flags * _FULL), 0)
        return count, entries.astype(np.int32)

    def on_device(self):
        """(bits, kv_count, kv_list, q_count, q_list) as grilly int32 tensors (cached)."""
        if not hasattr(self, "_device"):
            from ... import int32, tensor
            self._device = tuple(tensor(a, dtype=int32) for a in (self.bits, self.kv_count, self.kv_list,
                                                                  self.q_count, self.q_list))
        return self._device


class BlockMask:
    """What ``create_block_mask`` returns and ``flex_attention`` takes: the
    dense mask (``mask``, used when a ``score_mod`` forces the dense path)
    and its tile lists for the fused kernels (32x32 tiles; ``BLOCK_SIZE`` is
    kept for torch's API and does not change them)."""

    def __init__(self, mask: Tensor, mask_mod, seq_lengths, BLOCK_SIZE) -> None:
        self.mask = mask
        self.mask_mod = mask_mod
        self.seq_lengths = tuple(seq_lengths)
        self.BLOCK_SIZE = BLOCK_SIZE if isinstance(BLOCK_SIZE, tuple) else (BLOCK_SIZE, BLOCK_SIZE)
        self._tiles = None

    def tiles(self) -> _Tiles:
        if self._tiles is None:
            self._tiles = _Tiles(self.mask.numpy())
        return self._tiles

    def sparsity(self) -> float:
        """Percent of key tiles the kernels skip (torch's BlockMask.sparsity,
        at grilly's 32x32 tiles)."""
        t = self.tiles()
        return 100.0 * (1.0 - t.kv_count.sum() / t.kv_count.size / t.shape[-1])

    @property
    def shape(self):
        return tuple(self.mask.shape)

    def to_dense(self) -> Tensor:
        """The dense mask (torch returns a block-level mask; grilly has no blocks yet)."""
        return self.mask

    def __repr__(self) -> str:
        return f"BlockMask(shape={self.shape}, sparsity={self.sparsity():.2f}%)"


@functools.lru_cache(maxsize=16)
def _unmasked(Q: int, KV: int) -> _Tiles:
    return _Tiles.everything(Q, KV)


def create_block_mask(mask_mod, B, H, Q_LEN: int, KV_LEN: int, device=None, BLOCK_SIZE=128,
                      _compile: bool = False) -> BlockMask:
    """A BlockMask from a mask_mod. Build it once and reuse it: the mask is
    evaluated here, not per call."""
    return BlockMask(create_mask(mask_mod, B, H, Q_LEN, KV_LEN, device), mask_mod, (Q_LEN, KV_LEN), BLOCK_SIZE)


def _logsumexp_rows(scores: Tensor) -> Tensor:
    """logsumexp over the last dim; -inf for rows that are entirely -inf."""
    top = scores.detach().amax(-1, keepdim=True)
    shift = _ops.where(top == float("-inf"), 0.0, top)
    return ((scores - shift).exp().sum(-1, keepdim=True).log() + shift).squeeze(-1)


_SCORE_MOD_WARNED = False


def _warn_score_mod() -> None:
    global _SCORE_MOD_WARNED
    if not _SCORE_MOD_WARNED:
        _SCORE_MOD_WARNED = True
        warnings.warn("flex_attention with a score_mod runs on grilly's dense path, which builds the full "
                      "(Q, KV) score matrix: grilly cannot compile a Python score_mod into its fused kernel "
                      "(see docs/flex-attention.md).", UserWarning, stacklevel=3)


def _fused(q: Tensor, k: Tensor, v: Tensor, block_mask, scale: float):
    """The fused kernel's (out, lse)."""
    from ... import _C
    L, S = q.shape[2], k.shape[2]
    tiles = block_mask.tiles() if block_mask is not None else _unmasked(L, S)
    bits, kv_count, kv_list, _, _ = tiles.on_device()
    out, lse = _C.flash_forward(q._t, k._t, v._t, bits._t, kv_count._t, kv_list._t, tiles.shape[0], tiles.shape[1],
                                float(scale))
    return Tensor._wrap(out), Tensor._wrap(lse)


def flex_attention(query: Tensor, key: Tensor, value: Tensor, score_mod=None, block_mask: BlockMask | None = None,
                   scale: float | None = None, enable_gqa: bool = False, return_lse: bool = False,
                   kernel_options=None, *, return_aux: AuxRequest | None = None):
    """softmax(score_mod(q @ k^T * scale) masked by block_mask) @ v, with
    queries (B, Hq, L, E), keys (B, Hkv, S, E) and values (B, Hkv, S, Ev).
    Rows with no kept key give 0 (and an lse of -inf), as in torch."""
    q, k, v = _require_tensor("query", query), _require_tensor("key", key), _require_tensor("value", value)
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise NotImplementedError("NYI: query, key, and value must be 4D tensors")
    B, Hq, L, E = q.shape
    S = k.shape[-2]
    if Hq != k.shape[1]:
        if not enable_gqa:
            raise ValueError(f"Expect query and key/value to have the same number of heads but got Hq={Hq} and "
                             f"Hkv={k.shape[1]}. Try setting enable_gqa=True for GQA.")
        if Hq % k.shape[1]:
            raise ValueError(f"Expect number of query heads to be a multiple of kv heads for GQA but got "
                             f"Hq={Hq} and Hkv={k.shape[1]}.")
        k, v = F._repeat_heads(k, Hq // k.shape[1]), F._repeat_heads(v, Hq // v.shape[1])
    scale = 1.0 / math.sqrt(E) if scale is None else float(scale)
    if block_mask is not None and block_mask.seq_lengths != (L, S):
        raise ValueError(f"block_mask was created for block_mask.shape={block_mask.shape} but got "
                         f"q_len={L} and kv_len={S}. Create a new block mask for these sequence lengths.")
    if score_mod is not None:
        _warn_score_mod()
    from ... import is_grad_enabled
    needs_grad = is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    fusable = (score_mod is None and E <= 128 and v.shape[-1] <= 128 and not needs_grad
               and not (return_aux is not None and return_aux.max_scores))
    if fusable:
        out, lse = _fused(q, k, v, block_mask, scale)
        if return_aux is not None:
            return out, AuxOutput(lse=lse if return_aux.lse else None)
        if return_lse:
            warnings.warn("return_lse is deprecated and will be removed in v2.10. Please use "
                          "return_aux=AuxRequest(lse=True) instead.", FutureWarning, stacklevel=2)
            return out, lse
        return out
    scores = (q @ k.transpose(-2, -1)) * scale
    if score_mod is not None:
        scores = score_mod(scores, *_indices(B, Hq, L, S))
        scores = _require_tensor("score_mod result", scores)
    if block_mask is not None:
        scores = _ops.where(block_mask.mask, scores, float("-inf"))
    out = F._safe_softmax(scores, -1) @ v
    if return_aux is not None:
        lse = _logsumexp_rows(scores) if return_aux.lse else None
        top = scores.amax(-1) if return_aux.max_scores else None
        return out, AuxOutput(lse=lse, max_scores=top)
    if return_lse:
        warnings.warn("return_lse is deprecated and will be removed in v2.10. Please use "
                      "return_aux=AuxRequest(lse=True) instead.", FutureWarning, stacklevel=2)
        return out, _logsumexp_rows(scores)
    return out

