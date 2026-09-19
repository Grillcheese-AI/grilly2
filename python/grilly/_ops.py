"""torch's tensor operations on grilly's kernels: argument conventions
(negative dims, Python numbers as operands, ``dim=None``) are handled here;
the math and its gradients are in the Rust core. Each function is exported
at grilly's top level, and most are installed as ``Tensor`` methods."""

from __future__ import annotations

import builtins
import math

import numpy as np

from . import Tensor, _C, _not_yet, _require_tensor

_Number = (builtins.int, builtins.float, builtins.bool, np.number)


def _w(handle) -> Tensor:
    return Tensor._wrap(handle)


def _is_number(v) -> bool:
    return isinstance(v, _Number) and not isinstance(v, Tensor)


def _const(v) -> Tensor:
    return _w(_C.Tensor.full([], builtins.float(v)))


def _dim(d, ndim: builtins.int, extra: builtins.int = 0) -> builtins.int:
    """Normalize a possibly negative dim; `extra` allows positions past the end."""
    n = builtins.max(ndim + extra, 1)
    if not isinstance(d, (builtins.int, np.integer)) or isinstance(d, builtins.bool):
        raise TypeError(f"dim must be an int, not {type(d).__name__}")
    if not -n <= d < n:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-n}, {n - 1}], but got {d})")
    return builtins.int(d) % n


def _dims(dim, ndim) -> list:
    if dim is None or dim == ():
        return []
    if isinstance(dim, (tuple, list)):
        return [_dim(d, ndim) for d in dim]
    return [_dim(dim, ndim)]


def _unary(x, op, a=0.0, b=0.0) -> Tensor:
    return _w(_C.unary(_require_tensor("input", x)._t, op, builtins.float(a), builtins.float(b)))


def _binary(a, b, op) -> Tensor:
    return _w(_C.binary(a._t, b._t, op))


# ── arithmetic ───────────────────────────────────────────────────────────

def add(input, other, *, alpha=1) -> Tensor:
    if alpha != 1:
        other = mul(other, alpha)
    if _is_number(other):
        return _unary(input, "add_scalar", other)
    if _is_number(input):
        return _unary(other, "add_scalar", input)
    return _binary(_require_tensor("input", input), _require_tensor("other", other), "add")


def sub(input, other, *, alpha=1) -> Tensor:
    if alpha != 1:
        other = mul(other, alpha)
    if _is_number(other):
        return _unary(input, "add_scalar", -other)
    if _is_number(input):
        return _unary(other, "rsub_scalar", input)
    return _binary(_require_tensor("input", input), _require_tensor("other", other), "sub")


def mul(input, other) -> Tensor:
    if _is_number(other):
        return _unary(input, "mul_scalar", other)
    if _is_number(input):
        return _unary(other, "mul_scalar", input)
    return _binary(_require_tensor("input", input), _require_tensor("other", other), "mul")


def div(input, other, *, rounding_mode=None) -> Tensor:
    if rounding_mode is not None:
        raise _not_yet(f"div(rounding_mode={rounding_mode!r})")
    if _is_number(input):
        return _unary(other, "rdiv_scalar", input)
    # x / c as a true division (not x * (1/c)), as torch computes it
    b = _const(other) if _is_number(other) else _require_tensor("other", other)
    return _binary(_require_tensor("input", input), b, "div")


true_divide = divide = div
multiply = mul
subtract = sub


def pow(input, exponent) -> Tensor:
    if _is_number(exponent):
        return _unary(input, "pow_scalar", exponent)
    base = _const(input) if _is_number(input) else _require_tensor("input", input)
    return _binary(base, _require_tensor("exponent", exponent), "pow")


def maximum(input, other) -> Tensor:
    return _binary(_require_tensor("input", input), _require_tensor("other", other), "maximum")


def minimum(input, other) -> Tensor:
    return _binary(_require_tensor("input", input), _require_tensor("other", other), "minimum")


def neg(input) -> Tensor:
    return _unary(input, "neg")


negative = neg


def abs(input) -> Tensor:  # noqa: A001 - torch.abs
    return _unary(input, "abs")


def sign(input) -> Tensor:
    return _unary(input, "sign")


def exp(input) -> Tensor:
    return _unary(input, "exp")


def log(input) -> Tensor:
    return _unary(input, "log")


def sqrt(input) -> Tensor:
    return _unary(input, "sqrt")


def rsqrt(input) -> Tensor:
    return _unary(input, "rsqrt")


def tanh(input) -> Tensor:
    return _unary(input, "tanh")


def sigmoid(input) -> Tensor:
    return _unary(input, "sigmoid")


def square(input) -> Tensor:
    return _unary(input, "pow_scalar", 2.0)


def clamp(input, min=None, max=None) -> Tensor:  # noqa: A002 - torch's names
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")
    if isinstance(min, Tensor) or isinstance(max, Tensor):
        raise _not_yet("clamp with tensor bounds")
    lo = -math.inf if min is None else min
    hi = math.inf if max is None else max
    return _unary(input, "clamp", lo, hi)


clip = clamp


def clamp_min(input, min) -> Tensor:  # noqa: A002
    return clamp(input, min=min)


def clamp_max(input, max) -> Tensor:  # noqa: A002
    return clamp(input, max=max)


# ── reductions ───────────────────────────────────────────────────────────

def _no_dtype(dtype):
    from . import float32
    if dtype is not None and dtype is not float32:
        raise _not_yet(f"reductions with dtype={dtype!r}")


def sum(input, dim=None, keepdim=False, *, dtype=None) -> Tensor:  # noqa: A001
    _no_dtype(dtype)
    x = _require_tensor("input", input)
    return _w(_C.sum(x._t, _dims(dim, x.dim()), builtins.bool(keepdim)))


def mean(input, dim=None, keepdim=False, *, dtype=None) -> Tensor:
    _no_dtype(dtype)
    x = _require_tensor("input", input)
    return _w(_C.mean(x._t, _dims(dim, x.dim()), builtins.bool(keepdim)))


def amax(input, dim=(), keepdim=False) -> Tensor:
    x = _require_tensor("input", input)
    return _w(_C.amax(x._t, _dims(dim, x.dim()), builtins.bool(keepdim), True))


def amin(input, dim=(), keepdim=False) -> Tensor:
    x = _require_tensor("input", input)
    return _w(_C.amax(x._t, _dims(dim, x.dim()), builtins.bool(keepdim), False))


def max(input, dim=None, keepdim=False):  # noqa: A001
    if isinstance(dim, Tensor):
        return maximum(input, dim)  # torch.max(a, b)
    if dim is None:
        return amax(input)
    raise _not_yet("max(dim) (it returns int64 indices; integer dtypes are not implemented)")


def min(input, dim=None, keepdim=False):  # noqa: A001
    if isinstance(dim, Tensor):
        return minimum(input, dim)
    if dim is None:
        return amin(input)
    raise _not_yet("min(dim) (it returns int64 indices; integer dtypes are not implemented)")


def softmax(input, dim, dtype=None) -> Tensor:
    _no_dtype(dtype)
    x = _require_tensor("input", input)
    return _w(_C.softmax(x._t, _dim(dim, x.dim())))


def log_softmax(input, dim, dtype=None) -> Tensor:
    _no_dtype(dtype)
    x = _require_tensor("input", input)
    return _w(_C.log_softmax(x._t, _dim(dim, x.dim())))


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False) -> builtins.bool:
    a = _require_tensor("input", input)._host()
    b = _require_tensor("other", other)._host()
    return builtins.bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


# ── shape ────────────────────────────────────────────────────────────────

def _size_list(sizes) -> list:
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        return list(sizes[0])
    return list(sizes)


def transpose(input, dim0, dim1) -> Tensor:
    x = _require_tensor("input", input)
    return _w(_C.transpose(x._t, _dim(dim0, x.dim()), _dim(dim1, x.dim())))


swapaxes = swapdims = transpose


def t(input) -> Tensor:
    x = _require_tensor("input", input)
    if x.dim() > 2:
        raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {x.dim()}D")
    return x if x.dim() < 2 else transpose(x, 0, 1)


def permute(input, *dims) -> Tensor:
    x = _require_tensor("input", input)
    return _w(_C.permute(x._t, [_dim(d, x.dim()) for d in _size_list(dims)]))


def unsqueeze(input, dim) -> Tensor:
    x = _require_tensor("input", input)
    d = _dim(dim, x.dim(), extra=1)
    shape = list(x.shape)
    shape.insert(d, 1)
    return x.reshape(shape)


def squeeze(input, dim=None) -> Tensor:
    x = _require_tensor("input", input)
    dims = range(x.dim()) if dim is None else _dims(dim, x.dim())
    shape = [s for i, s in enumerate(x.shape) if not (i in dims and s == 1)]
    return x.reshape(shape)


def flatten(input, start_dim=0, end_dim=-1) -> Tensor:
    x = _require_tensor("input", input)
    if x.dim() == 0:
        return x.reshape(1)
    s, e = _dim(start_dim, x.dim()), _dim(end_dim, x.dim())
    if s > e:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")
    shape = list(x.shape)
    return x.reshape(shape[:s] + [math.prod(shape[s:e + 1])] + shape[e + 1:])


def expand(input, *sizes) -> Tensor:
    x = _require_tensor("input", input)
    sizes = _size_list(sizes)
    lead = len(sizes) - x.dim()
    if lead < 0:
        raise RuntimeError(
            f"expand: the number of sizes provided ({len(sizes)}) must be greater or equal to "
            f"the number of dimensions in the tensor ({x.dim()})"
        )
    shape = [x.shape[i - lead] if s == -1 and i >= lead else s for i, s in enumerate(sizes)]
    return _w(_C.expand(x._t, shape))


def narrow(input, dim, start, length) -> Tensor:
    x = _require_tensor("input", input)
    d = _dim(dim, x.dim())
    start = start + x.shape[d] if start < 0 else start
    return _index(x, (slice(None),) * d + (slice(start, start + length),))


def select(input, dim, index) -> Tensor:
    x = _require_tensor("input", input)
    return _index(x, (slice(None),) * _dim(dim, x.dim()) + (index,))


def chunk(input, chunks, dim=0) -> tuple:
    x = _require_tensor("input", input)
    d = _dim(dim, x.dim())
    size = -(-x.shape[d] // chunks) if x.shape[d] else 0
    return split(x, builtins.max(size, 1), d)


def split(tensor, split_size_or_sections, dim=0) -> tuple:
    x = _require_tensor("tensor", tensor)
    d = _dim(dim, x.dim())
    n = x.shape[d]
    if isinstance(split_size_or_sections, (builtins.int, np.integer)):
        step = builtins.int(split_size_or_sections)
        sizes = [builtins.min(step, n - i) for i in range(0, n, step)] or [0]
    else:
        sizes = list(split_size_or_sections)
        if builtins.sum(sizes) != n:
            raise RuntimeError(f"split_with_sizes expects split_sizes to sum exactly to {n} (input tensor's size at dimension {d}), but got split_sizes={sizes}")
    out, at = [], 0
    for s in sizes:
        out.append(narrow(x, d, at, s))
        at += s
    return tuple(out)


def cat(tensors, dim=0) -> Tensor:
    ts = [_require_tensor("tensors", t) for t in tensors]
    if not ts:
        raise RuntimeError("cat(): expected a non-empty list of Tensors")
    return _w(_C.cat([t._t for t in ts], _dim(dim, ts[0].dim())))


concat = concatenate = cat


def stack(tensors, dim=0) -> Tensor:
    ts = [_require_tensor("tensors", t) for t in tensors]
    if not ts:
        raise RuntimeError("stack expects a non-empty TensorList")
    return _w(_C.stack([t._t for t in ts], _dim(dim, ts[0].dim(), extra=1)))


def matmul(input, other) -> Tensor:
    return _w(_C.matmul(_require_tensor("input", input)._t, _require_tensor("other", other)._t))


mm = matmul


def bmm(input, mat2) -> Tensor:
    a, b = _require_tensor("input", input), _require_tensor("mat2", mat2)
    if a.dim() != 3 or b.dim() != 3:
        raise RuntimeError("batch1 must be a 3D tensor")
    return matmul(a, b)


def contiguous(input) -> Tensor:
    return _w(_C.contiguous(_require_tensor("input", input)._t))


# ── like-constructors ────────────────────────────────────────────────────

def _like(input, fill, dtype, device, requires_grad) -> Tensor:
    from . import _check_factory_args, _code, _leaf
    _check_factory_args(dtype, device)
    x = _require_tensor("input", input)
    return _leaf(_w(_C.Tensor.full(list(x.shape), builtins.float(fill), _code(dtype or x.dtype))), requires_grad)


def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None) -> Tensor:
    return _like(input, 0.0, dtype, device, requires_grad)


def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None) -> Tensor:
    return _like(input, 1.0, dtype, device, requires_grad)


def full_like(input, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None) -> Tensor:
    return _like(input, fill_value, dtype, device, requires_grad)


def empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None) -> Tensor:
    return _like(input, 0.0, dtype, device, requires_grad)


# ── indexing ─────────────────────────────────────────────────────────────

def _parse_index(x: Tensor, idx):
    """Basic indexing -> (offset, strides, shape, only_ints) over x's
    contiguous layout. Advanced (tensor/list/bool) indexing is not supported."""
    if not isinstance(idx, tuple):
        idx = (idx,)
    consumed = builtins.sum(1 for i in idx if i is not None and i is not Ellipsis)
    if builtins.sum(1 for i in idx if i is Ellipsis) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if consumed > x.dim():
        raise IndexError(f"too many indices for tensor of dimension {x.dim()}")
    expanded = []
    for i in idx:
        if i is Ellipsis:
            expanded.extend([slice(None)] * (x.dim() - consumed))
        else:
            expanded.append(i)
    shape, strides = list(x.shape), [1] * x.dim()
    for d in range(x.dim() - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    offset, out_shape, out_strides, d, only_ints = 0, [], [], 0, True
    for item in expanded:
        if item is None:
            out_shape.append(1)
            out_strides.append(0)
            only_ints = False
        elif isinstance(item, (builtins.int, np.integer)) and not isinstance(item, builtins.bool):
            n = shape[d]
            i = builtins.int(item)
            if not -n <= i < n:
                raise IndexError(f"index {i} is out of bounds for dimension {d} with size {n}")
            offset += (i % n) * strides[d]
            d += 1
        elif isinstance(item, slice):
            start, stop, step = item.indices(shape[d])
            if step <= 0:
                raise ValueError("step must be greater than zero")
            length = len(range(start, stop, step))
            offset += (start if length else 0) * strides[d]
            out_shape.append(length)
            out_strides.append(strides[d] * step)
            d += 1
            only_ints = False
        else:
            raise _not_yet(f"indexing with {type(item).__name__} (advanced indexing)")
    out_shape += shape[d:]
    out_strides += strides[d:]
    return offset, out_strides, out_shape, only_ints


def _index(x: Tensor, idx) -> Tensor:
    offset, strides, shape, only_ints = _parse_index(x, idx)
    return _w(_C.strided_read(x._t, offset, strides, shape, only_ints))


def _setitem(x: Tensor, idx, value) -> None:
    offset, strides, shape, _ = _parse_index(x, idx)
    if _is_number(value):
        from . import _code
        v = _w(_C.Tensor.full([], builtins.float(value), _code(x.dtype)))
    else:
        v = _require_tensor("value", value)
    _C.write_region(x._t, offset, strides, shape, v._t)


# ── installation as Tensor methods ───────────────────────────────────────

def _comparison(name):
    def op(self, other):
        raise _not_yet(f"tensor comparison ({name}), which returns a bool tensor; bool dtype")
    return op


def _install() -> None:
    T = Tensor
    for name in ("add", "sub", "mul", "div", "pow", "neg", "abs", "sign", "exp", "log",
                 "sqrt", "rsqrt", "tanh", "sigmoid", "square", "clamp", "clip", "clamp_min",
                 "clamp_max", "sum", "mean", "amax", "amin", "max", "min", "softmax",
                 "log_softmax", "transpose", "swapaxes", "t", "permute", "unsqueeze",
                 "squeeze", "flatten", "expand", "narrow", "select", "chunk", "split",
                 "matmul", "mm", "bmm", "contiguous", "maximum", "minimum", "true_divide",
                 "multiply", "subtract", "negative", "divide"):
        setattr(T, name, globals()[name])
    T.__add__ = add
    T.__radd__ = lambda self, other: add(other, self)
    T.__sub__ = sub
    T.__rsub__ = lambda self, other: sub(other, self)
    T.__mul__ = mul
    T.__rmul__ = lambda self, other: mul(other, self)
    T.__truediv__ = div
    T.__rtruediv__ = lambda self, other: div(other, self)
    T.__pow__ = pow
    T.__rpow__ = lambda self, other: pow(other, self)
    T.__neg__ = neg
    T.__abs__ = abs
    T.__matmul__ = matmul
    T.__rmatmul__ = lambda self, other: matmul(other, self)
    T.__getitem__ = _index
    T.__setitem__ = _setitem
    for dunder, sym in (("__eq__", "=="), ("__ne__", "!="), ("__lt__", "<"), ("__le__", "<="),
                        ("__gt__", ">"), ("__ge__", ">=")):
        setattr(T, dunder, _comparison(sym))
    T.__hash__ = object.__hash__  # identity, as torch's
    T.T = property(lambda self: permute(self, list(range(self.dim() - 1, -1, -1))))
    T.expand_as = lambda self, other: expand(self, list(other.shape))
    T.view_as = lambda self, other: self.view(list(other.shape))
    T.reshape_as = lambda self, other: self.reshape(list(other.shape))
    T.__iter__ = lambda self: (self[i] for i in range(len(self)))
    T.__float__ = _scalar(builtins.float)
    T.__int__ = _scalar(builtins.int)       # truncates toward zero, as torch
    T.__bool__ = _truth


def _scalar(convert):
    def to_python(self):
        if self.numel() != 1:
            raise ValueError("only one element tensors can be converted to Python scalars")
        return convert(self._host().reshape(()).item())
    return to_python


def _truth(self) -> builtins.bool:
    if self.numel() != 1:
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
    return builtins.bool(self._host().reshape(()).item())
