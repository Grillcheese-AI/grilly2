"""grilly.nn.functional, like torch.nn.functional."""

from __future__ import annotations

import functools
import math
import warnings

from .. import Tensor, _C, _not_yet, _require_tensor
from .. import _ops
from .. import relu as _relu

__all__ = [
    "cosine_similarity", "cross_entropy", "dropout", "embedding", "gelu", "layer_norm", "linear",
    "log_softmax", "mse_loss", "nll_loss", "normalize", "relu", "scaled_dot_product_attention", "sigmoid",
    "silu", "softmax", "tanh",
]


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``input @ weight.T + bias`` over the last dimension of ``input``."""
    own = getattr(type(weight), "_grilly_linear", None)
    if own is not None:  # a weight with its own kernel, e.g. a quantized one (grilly.infer)
        return own(weight, input, bias)
    x = _require_tensor("input", input)
    w = _require_tensor("weight", weight)
    b = None if bias is None else _require_tensor("bias", bias)._t
    return Tensor._wrap(_C.linear(x._t, w._t, b))


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        raise _not_yet("relu(inplace=True)")
    return _relu(input)


def tanh(input: Tensor) -> Tensor:
    return _ops.tanh(input)


def sigmoid(input: Tensor) -> Tensor:
    return _ops.sigmoid(input)


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    if approximate == "none":
        return _ops._unary(input, "gelu")
    if approximate == "tanh":
        x = _require_tensor("input", input)
        inner = (x + 0.044715 * x ** 3) * math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + inner.tanh())
    raise RuntimeError(f"approximate argument must be either none or tanh, got {approximate!r}")


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        raise _not_yet("silu(inplace=True)")
    return Tensor._wrap(_C.silu(_require_tensor("input", input)._t))


def _implicit_dim(name: str, ndim: int) -> int:
    warnings.warn(f"Implicit dimension choice for {name} has been deprecated. "
                  "Change the call to include dim=X as an argument.", stacklevel=3)
    return 0 if ndim in (0, 1, 3) else 1


def softmax(input: Tensor, dim=None, _stacklevel: int = 3, dtype=None) -> Tensor:
    x = _require_tensor("input", input)
    return _ops.softmax(x, _implicit_dim("softmax", x.dim()) if dim is None else dim, dtype)


def log_softmax(input: Tensor, dim=None, _stacklevel: int = 3, dtype=None) -> Tensor:
    x = _require_tensor("input", input)
    return _ops.log_softmax(x, _implicit_dim("log_softmax", x.dim()) if dim is None else dim, dtype)


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12, out=None) -> Tensor:
    if p != 2.0:
        raise _not_yet(f"normalize(p={p})")
    if out is not None:
        raise _not_yet("normalize(out=...)")
    x = _require_tensor("input", input)
    return Tensor._wrap(_C.normalize(x._t, _ops._dim(dim, x.dim()), float(eps)))


def layer_norm(input: Tensor, normalized_shape, weight=None, bias=None, eps: float = 1e-5) -> Tensor:
    x = _require_tensor("input", input)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    shape = tuple(normalized_shape)
    if not shape or tuple(x.shape[-len(shape):]) != shape:
        raise RuntimeError(f"Given normalized_shape={list(shape)}, expected input with shape [*, "
                           f"{', '.join(map(str, shape))}], but got input of size{list(x.shape)}")
    w = None if weight is None else _require_tensor("weight", weight)._t
    b = None if bias is None else _require_tensor("bias", bias)._t
    return Tensor._wrap(_C.layer_norm(x._t, len(shape), w, b, float(eps)))


def _clamped_norm(x: Tensor, dim: int, eps: float) -> Tensor:
    """max(||x||, eps) in value, but with the gradient of ||x||: torch clamps
    the norms in place under no_grad (ATen Distance.cpp), so its gradient
    ignores the clamp."""
    n = (x * x).sum(dim, keepdim=True).sqrt()
    return n + (n.detach().clamp_min(eps) - n.detach())


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Each input is divided by max(its norm, eps) before the dot product,
    as torch 2.x does, with torch's gradient (see _clamped_norm)."""
    a, b = _require_tensor("x1", x1), _require_tensor("x2", x2)
    d = _ops._dim(dim, max(a.dim(), b.dim()))
    return ((a / _clamped_norm(a, d, eps)) * (b / _clamped_norm(b, d, eps))).sum(d)


def _reduce(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"{reduction} is not a valid value for reduction")


def _nll(logp: Tensor, target: Tensor, ignore_index: int, reduction: str, smoothing: float) -> Tensor:
    """Negative log-likelihood of class indices along dim 1 (or 0 for 1-D
    input), with ignore_index; 'mean' averages over the non-ignored targets."""
    class_dim = 1 if logp.dim() > 1 else 0
    if target.is_floating_point():
        raise RuntimeError("expected scalar type Long but found Float")
    keep = target != ignore_index
    safe = _ops.where(keep, target, 0)
    picked = _ops.gather(logp, class_dim, safe.unsqueeze(class_dim)).squeeze(class_dim)
    loss = -picked
    if smoothing:
        loss = loss * (1.0 - smoothing) - logp.mean(class_dim) * smoothing
    loss = _ops.where(keep, loss, 0.0)
    if reduction == "mean":
        return loss.sum() / keep.sum()
    return _reduce(loss, reduction)


def nll_loss(input: Tensor, target: Tensor, weight=None, size_average=None, ignore_index: int = -100,
             reduce=None, reduction: str = "mean") -> Tensor:
    if weight is not None or size_average is not None or reduce is not None:
        raise _not_yet("nll_loss(weight / size_average / reduce)")
    return _nll(_require_tensor("input", input), _require_tensor("target", target), ignore_index, reduction, 0.0)


def cross_entropy(input: Tensor, target: Tensor, weight=None, size_average=None, ignore_index: int = -100,
                  reduce=None, reduction: str = "mean", label_smoothing: float = 0.0) -> Tensor:
    """Class-index targets (int64, the input's shape without the class dim)
    or class-probability targets (the input's shape)."""
    x, t = _require_tensor("input", input), _require_tensor("target", target)
    if weight is not None or size_average is not None or reduce is not None:
        raise _not_yet("cross_entropy(weight / size_average / reduce)")
    class_dim = 1 if x.dim() > 1 else 0
    if tuple(t.shape) != tuple(x.shape):
        return _nll(log_softmax(x, class_dim), t, ignore_index, reduction, label_smoothing)
    if label_smoothing:
        t = t * (1.0 - label_smoothing) + label_smoothing / x.shape[class_dim]
    loss = -(t * log_softmax(x, class_dim)).sum(class_dim)
    return _reduce(loss, reduction)


def embedding(input: Tensor, weight: Tensor, padding_idx=None, max_norm=None, norm_type: float = 2.0,
              scale_grad_by_freq: bool = False, sparse: bool = False) -> Tensor:
    if max_norm is not None or scale_grad_by_freq or sparse:
        raise _not_yet("embedding(max_norm / scale_grad_by_freq / sparse)")
    w = _require_tensor("weight", weight)
    if padding_idx is not None and padding_idx < 0:
        padding_idx += w.shape[0]
    return Tensor._wrap(_C.embedding(_require_tensor("input", input)._t, w._t, padding_idx))


def mse_loss(input: Tensor, target: Tensor, size_average=None, reduce=None, reduction: str = "mean", weight=None) -> Tensor:
    if size_average is not None or reduce is not None or weight is not None:
        raise _not_yet("mse_loss(size_average / reduce / weight)")
    return _reduce((_require_tensor("input", input) - _require_tensor("target", target)) ** 2, reduction)


def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    x = _require_tensor("input", input)
    if inplace:
        raise _not_yet("dropout(inplace=True)")
    if not training or p == 0.0:
        return x
    if p == 1.0:
        return x * 0.0
    from .. import rand
    keep = Tensor._wrap(_C.binary(rand(*x.shape)._t, _ops._const(p)._t, "ge"))
    return x * keep * (1.0 / (1.0 - p))


def _safe_softmax(x: Tensor, dim: int) -> Tensor:
    """softmax, but 0 (with a zero gradient) on rows that are entirely -inf,
    as torch's SDPA does (its _safe_softmax); plain softmax gives NaN there.
    The dead rows are zeroed before the softmax too, so no NaN reaches the
    backward."""
    dead = x.detach().amax(dim, keepdim=True) == float("-inf")  # the row max is -inf
    return _ops.where(dead, 0.0, _ops.softmax(_ops.where(dead, 0.0, x), dim))


def _repeat_heads(t: Tensor, r: int) -> Tensor:
    """torch's repeat_interleave(r, dim=-3): each head r times in a row."""
    shape = list(t.shape)
    wide = t.unsqueeze(-3).expand(*shape[:-2], r, *shape[-2:])
    return wide.reshape(*shape[:-3], shape[-3] * r, *shape[-2:])


@functools.lru_cache(maxsize=16)
def _causal_tiles(L: int, S: int):
    from .. import arange
    from .attention.flex_attention import _Tiles
    keep = arange(L).unsqueeze(1) >= arange(S).unsqueeze(0)
    return _Tiles.from_mask(keep.reshape(1, 1, L, S))


def _fused_sdpa(q: Tensor, k: Tensor, v: Tensor, attn_mask, is_causal: bool, scale: float):
    """The fused block-sparse kernels' output, or None when they do not apply
    (a float mask, 5-D inputs, head dims > 128)."""
    from .attention.flex_attention import _Tiles, _run, _unmasked
    b_ = _bool_dtype()
    if q.dim() not in (3, 4) or k.dim() != q.dim() or v.dim() != q.dim():
        return None
    if q.shape[-1] > 128 or v.shape[-1] > 128 or (attn_mask is not None and attn_mask.dtype is not b_):
        return None
    three = q.dim() == 3
    if three:
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    B, H, L, S = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
    if attn_mask is None:
        tiles = _causal_tiles(L, S) if is_causal else _unmasked(L, S)
    else:
        from .. import arange
        m = attn_mask
        if three and m.dim() == 3:
            m = m.unsqueeze(1)
        m = m.reshape(*([1] * (4 - m.dim())), *m.shape) if m.dim() < 4 else m
        if is_causal:
            m = m & (arange(L).unsqueeze(1) >= arange(S).unsqueeze(0))
        mb, mh = m.shape[0], m.shape[1]
        if mb not in (1, B) or mh not in (1, H):
            return None
        if tuple(m.shape[2:]) != (L, S):
            m = m.expand(mb, mh, L, S)
        tiles = _Tiles.from_mask(m)
    out, _ = _run(q, k, v, tiles, scale)
    return out.squeeze(1) if three else out


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask=None, dropout_p: float = 0.0,
                                 is_causal: bool = False, scale=None, enable_gqa: bool = False) -> Tensor:
    """``softmax(query @ key^T * scale + mask) @ value``. Masks follow torch
    2.10: a bool mask keeps True positions, a float mask is added,
    ``is_causal`` keeps the top-left triangle (combined with a bool mask when
    both are given, as torch does on CPU), and fully-masked rows give 0.

    Without dropout or a float mask (and with head dims <= 128) this runs the
    fused block-sparse kernels flex_attention uses: masked-out 32x32 tiles
    are skipped and no score matrix reaches memory. Otherwise it computes the
    (L, S) scores as torch's math backend does."""
    q, k, v = _require_tensor("query", query), _require_tensor("key", key), _require_tensor("value", value)
    L, S, E = q.shape[-2], k.shape[-2], q.shape[-1]
    if enable_gqa and q.dim() >= 3 and q.shape[-3] != k.shape[-3]:
        hq, hk = q.shape[-3], k.shape[-3]
        if hq % hk or hq % v.shape[-3]:
            raise RuntimeError(f"number of heads in key/value must divide number of heads in query ({hk} vs {hq})")
        k, v = _repeat_heads(k, hq // hk), _repeat_heads(v, hq // v.shape[-3])
    scale = 1.0 / math.sqrt(E) if scale is None else float(scale)
    if dropout_p == 0.0:
        m = None if attn_mask is None else _require_tensor("attn_mask", attn_mask)
        out = _fused_sdpa(q, k, v, m, is_causal, scale)
        if out is not None:
            return out
    scores = _ops.matmul(q, k.transpose(-2, -1)) * scale
    keep = None
    if is_causal:
        from .. import arange
        keep = arange(L).unsqueeze(1) >= arange(S).unsqueeze(0)
    if attn_mask is not None:
        m = _require_tensor("attn_mask", attn_mask)
        if m.dtype is _bool_dtype():
            keep = m if keep is None else m & keep
        else:
            scores = scores + m
    if keep is not None:
        scores = _ops.where(keep, scores, float("-inf"))
    attn = _safe_softmax(scores, -1)
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p)
    return _ops.matmul(attn, v)


def _bool_dtype():
    from .. import bool as b_
    return b_
