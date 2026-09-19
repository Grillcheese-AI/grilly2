"""grilly: the torch API on Vulkan.

Use it the way you use torch; names, argument order and semantics follow
torch. What is not implemented yet raises NotImplementedError saying so,
rather than behaving differently from torch.
"""

from __future__ import annotations

import builtins as _builtins
import math as _math

import numpy as _np

from . import _C

__all__ = [
    "Size", "Tensor", "add", "dtype", "empty", "float32", "from_numpy",
    "is_tensor", "ones", "relu", "tensor", "zeros",
]


# ── dtypes ───────────────────────────────────────────────────────────────

class dtype:
    """A tensor element type, like ``torch.dtype``."""

    __slots__ = ("_name", "is_floating_point")

    def __init__(self, name: str, is_floating_point: bool) -> None:
        self._name = name
        self.is_floating_point = is_floating_point

    def __repr__(self) -> str:
        return f"grilly.{self._name}"

    def __reduce__(self):
        return (getattr, (_this_module(), self._name))


float32 = float = dtype("float32", True)
float64 = double = dtype("float64", True)
float16 = half = dtype("float16", True)
bfloat16 = dtype("bfloat16", True)
int64 = long = dtype("int64", False)
int32 = int = dtype("int32", False)
int16 = short = dtype("int16", False)
int8 = dtype("int8", False)
uint8 = dtype("uint8", False)
bool = dtype("bool", False)  # noqa: A001 - torch.bool

_SUPPORTED = (float32,)


def _this_module():
    import sys
    return sys.modules[__name__]


def _not_yet(what: str) -> NotImplementedError:
    return NotImplementedError(f"grilly: {what} is not implemented yet")


def _check_factory_args(dtype_, device, requires_grad) -> None:
    if dtype_ is not None and dtype_ not in _SUPPORTED:
        raise _not_yet(f"dtype {dtype_!r} (only grilly.float32 so far)")
    if device is not None:
        raise _not_yet(f"device={device!r} (tensors always live on the default GPU)")
    if requires_grad:
        raise _not_yet("autograd (requires_grad=True)")


# ── Size ─────────────────────────────────────────────────────────────────

class Size(tuple):
    """A tensor shape; a tuple, like ``torch.Size``."""

    def __repr__(self) -> str:
        return f"grilly.Size({list(self)!r})"

    def numel(self) -> _builtins.int:
        return _math.prod(self)


def _size_args(size) -> tuple[_builtins.int, ...]:
    """Accept f(2, 3), f((2, 3)), f([2, 3]) and f(Size)."""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    for d in size:
        if not isinstance(d, _builtins.int) or isinstance(d, _builtins.bool):
            raise TypeError(f"size must be ints, got {size!r}")
    return tuple(size)


def _resolve_shape(shape: tuple[_builtins.int, ...], numel: _builtins.int) -> list[_builtins.int]:
    """Fill in a single -1, as torch.reshape does."""
    if shape.count(-1) > 1:
        raise RuntimeError("only one dimension can be inferred")
    if -1 in shape:
        known = _math.prod(d for d in shape if d != -1)
        if known == 0 or numel % known:
            raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {numel}")
        shape = tuple(numel // known if d == -1 else d for d in shape)
    if any(d < 0 for d in shape):
        raise RuntimeError(f"invalid shape dimension in {list(shape)}")
    return list(shape)


# ── Tensor ───────────────────────────────────────────────────────────────

class Tensor:
    """A float32 tensor on the GPU, like ``torch.Tensor``."""

    __slots__ = ("_t",)

    def __init__(self, *args, **kwargs) -> None:
        raise _not_yet("the legacy grilly.Tensor(...) constructor; use grilly.tensor(...)")

    @classmethod
    def _wrap(cls, handle) -> "Tensor":
        t = object.__new__(cls)
        t._t = handle
        return t

    # shape and type
    @property
    def shape(self) -> Size:
        return Size(self._t.shape)

    def size(self, dim: _builtins.int | None = None):
        shape = self.shape
        if dim is None:
            return shape
        return shape[self._dim_index(dim)]

    def dim(self) -> _builtins.int:
        return len(self._t.shape)

    ndim = property(dim)

    def numel(self) -> _builtins.int:
        return _math.prod(self._t.shape)

    @property
    def dtype(self) -> dtype:
        return float32

    @property
    def requires_grad(self) -> _builtins.bool:
        return False

    def _dim_index(self, dim: _builtins.int) -> _builtins.int:
        n = self.dim()
        if not -n <= dim < n or n == 0:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-n}, {n - 1}], but got {dim})"
            )
        return dim % n

    # host conversion
    def numpy(self) -> _np.ndarray:
        data = _np.frombuffer(self._t.to_bytes(), dtype="<f4")
        return data.reshape(self._t.shape).copy()

    def tolist(self):
        return self.numpy().tolist()

    def item(self) -> _builtins.float:
        if self.numel() != 1:
            raise RuntimeError(
                f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
            )
        return _builtins.float(self.numpy().reshape(()))

    def __len__(self) -> _builtins.int:
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._t.shape[0]

    def __repr__(self) -> str:
        body = _np.array2string(
            self.numpy(), separator=", ", prefix="tensor(",
            formatter={"float_kind": lambda v: f"{v:.4f}"},
        )
        return f"tensor({body})"

    # views
    def reshape(self, *shape) -> "Tensor":
        return Tensor._wrap(self._t.reshape(_resolve_shape(_size_args(shape), self.numel())))

    def view(self, *shape) -> "Tensor":
        return self.reshape(*shape)

    # ops
    def relu(self) -> "Tensor":
        return relu(self)

    def add(self, other, *, alpha=1) -> "Tensor":
        return add(self, other, alpha=alpha)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(self, other)


def is_tensor(obj) -> _builtins.bool:
    return isinstance(obj, Tensor)


def _require_tensor(name: str, value) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (_builtins.int, _builtins.float)):
        raise _not_yet(f"a Python scalar as {name}")
    raise TypeError(f"{name} must be a Tensor, not {type(value).__name__}")


# ── creation ─────────────────────────────────────────────────────────────

def _from_array(arr: _np.ndarray) -> Tensor:
    # Not ascontiguousarray: it turns 0-d arrays into 1-d.
    arr = _np.asarray(arr, dtype="<f4", order="C")
    return Tensor._wrap(_C.Tensor.from_bytes(arr.tobytes(), list(arr.shape)))


def tensor(data, *, dtype=None, device=None, requires_grad=False) -> Tensor:
    """Copy ``data`` (nested sequences, a scalar, or an ndarray) to a tensor."""
    _check_factory_args(dtype, device, requires_grad)
    if isinstance(data, Tensor):  # torch copies here too
        return _from_array(data.numpy())
    if isinstance(data, _np.ndarray):
        if data.dtype != _np.float32 and dtype is None:
            raise _not_yet(f"tensors of numpy dtype {data.dtype} (only float32 so far)")
        return _from_array(data)
    arr = _np.asarray(data)
    # torch infers float32 from Python floats and int64/bool from ints/bools.
    if arr.dtype.kind == "f" or (dtype is float32 and arr.dtype.kind in "iub"):
        return _from_array(arr.astype(_np.float32))
    raise _not_yet(f"tensors of inferred dtype {arr.dtype} (only float32 so far)")


def from_numpy(ndarray: _np.ndarray) -> Tensor:
    """Like ``torch.from_numpy``, but the tensor is a GPU copy: later writes
    to ``ndarray`` are not seen by the tensor, and vice versa."""
    if not isinstance(ndarray, _np.ndarray):
        raise TypeError(f"expected np.ndarray (got {type(ndarray).__name__})")
    if ndarray.dtype != _np.float32:
        raise _not_yet(f"from_numpy for dtype {ndarray.dtype} (only float32 so far)")
    return _from_array(ndarray)


def empty(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    _check_factory_args(dtype, device, requires_grad)
    return Tensor._wrap(_C.Tensor.empty(list(_size_args(size))))


def zeros(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    _check_factory_args(dtype, device, requires_grad)
    return _from_array(_np.zeros(_size_args(size), dtype=_np.float32))


def ones(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    _check_factory_args(dtype, device, requires_grad)
    return _from_array(_np.ones(_size_args(size), dtype=_np.float32))


# ── ops ──────────────────────────────────────────────────────────────────

def relu(input: Tensor) -> Tensor:
    return Tensor._wrap(_C.relu(_require_tensor("input", input)._t))


def add(input, other, *, alpha=1) -> Tensor:
    if alpha != 1:
        raise _not_yet("add with alpha != 1")
    a = _require_tensor("input", input)
    b = _require_tensor("other", other)
    if a.shape != b.shape:
        raise _not_yet(f"broadcasting in add ({list(a.shape)} vs {list(b.shape)})")
    return Tensor._wrap(_C.add(a._t, b._t))


from . import nn  # noqa: E402  (needs Tensor defined)
