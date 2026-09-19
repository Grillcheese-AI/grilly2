"""torch.nn.attention.flex_attention, dense.

``mask_mod(b, h, q_idx, kv_idx)`` and ``score_mod(score, b, h, q_idx,
kv_idx)`` are called once, on broadcast index tensors (b: (B,1,1,1),
h: (1,H,1,1), q_idx: (1,1,Q,1), kv_idx: (1,1,1,KV)), where torch vmaps them
over scalars. Any mod written with tensor ops works either way; Python
control flow on the index values does not (torch rejects it too).

This is the math torch's uncompiled flex_attention does: the full (Q, KV)
score matrix is built and masked, so a block mask saves no work yet.
``BLOCK_SIZE``, ``kernel_options`` and ``_compile`` are accepted and have no
effect.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

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


class BlockMask:
    """What ``create_block_mask`` returns and ``flex_attention`` takes. Here
    it holds the dense mask (``mask``), not torch's block-sparse indices."""

    def __init__(self, mask: Tensor, mask_mod, seq_lengths, BLOCK_SIZE) -> None:
        self.mask = mask
        self.mask_mod = mask_mod
        self.seq_lengths = tuple(seq_lengths)
        self.BLOCK_SIZE = BLOCK_SIZE if isinstance(BLOCK_SIZE, tuple) else (BLOCK_SIZE, BLOCK_SIZE)

    @property
    def shape(self):
        return tuple(self.mask.shape)

    def to_dense(self) -> Tensor:
        """The dense mask (torch returns a block-level mask; grilly has no blocks yet)."""
        return self.mask

    def __repr__(self) -> str:
        return f"BlockMask(shape={self.shape}, dense=True)"


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
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(E) if scale is None else float(scale))
    if score_mod is not None:
        scores = score_mod(scores, *_indices(B, Hq, L, S))
        scores = _require_tensor("score_mod result", scores)
    if block_mask is not None:
        if block_mask.seq_lengths != (L, S):
            raise ValueError(f"block_mask was created for block_mask.shape={block_mask.shape} but got "
                             f"q_len={L} and kv_len={S}. Create a new block mask for these sequence lengths.")
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

