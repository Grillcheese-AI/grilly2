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
    "Generator", "Size", "Tensor", "add", "default_generator", "device", "dtype", "empty",
    "enable_grad", "float32", "from_numpy", "full", "initial_seed",
    "is_grad_enabled", "is_tensor", "manual_seed", "no_grad", "ones", "rand",
    "rand_like", "randint", "randn", "randn_like", "relu", "seed",
    "set_grad_enabled", "tensor", "zeros",
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


# ── devices ──────────────────────────────────────────────────────────────

_DEVICE_TYPES = (
    "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga",
    "maia", "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia", "privateuseone",
)


class device:
    """A device, like ``torch.device``. grilly runs on one Vulkan device,
    ``device(type='vulkan', index=0)``; ``"cpu"`` is accepted anywhere a device
    is expected as an alias for it (see docs/devices.md)."""

    __slots__ = ("type", "index")

    def __init__(self, type, index=None) -> None:  # noqa: A002 - torch's name
        if isinstance(type, device):
            self.type, self.index = type.type, type.index
            return
        if isinstance(type, _builtins.int):
            type, index = "cuda", type
        if not isinstance(type, str):
            raise TypeError(f"device() received an invalid combination of arguments: {type!r}")
        name = type
        if ":" in name:
            name, _, idx = name.partition(":")
            if index is not None:
                raise RuntimeError("type (string) must not include an index because index was passed explicitly")
            index = _builtins.int(idx)
        if name not in _DEVICE_TYPES:
            raise RuntimeError(f"Expected one of {', '.join(_DEVICE_TYPES)} device type at start of device string: {type}")
        self.type, self.index = name, index

    def __eq__(self, other) -> _builtins.bool:
        return isinstance(other, device) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self) -> _builtins.int:
        return hash((self.type, self.index))

    def __repr__(self) -> str:
        if self.index is None:
            return f"device(type='{self.type}')"
        return f"device(type='{self.type}', index={self.index})"

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


_DEVICE = device("vulkan", 0)


def _resolve_device(d) -> device:
    """The grilly device, for None, vulkan[:0] or the "cpu" alias."""
    if d is None:
        return _DEVICE
    dev = device(d)
    if dev.type in ("vulkan", "cpu") and dev.index in (None, 0):
        return _DEVICE
    raise _not_yet(f"device {str(dev)!r} (grilly runs on one Vulkan device; 'cpu' is accepted as an alias for it)")


def _check_factory_args(dtype_, device_, requires_grad=False) -> None:
    if dtype_ is not None and dtype_ not in _SUPPORTED:
        raise _not_yet(f"dtype {dtype_!r} (only grilly.float32 so far)")
    _resolve_device(device_)


def _cast(t: "Tensor", dt) -> "Tensor":
    raise _not_yet(f"converting to {dt!r} (only float32 so far)")


def _leaf(t: "Tensor", requires_grad) -> "Tensor":
    if requires_grad:
        t._t.set_requires_grad(True)
    return t


class _GradFn:
    """The node that produced a tensor, as torch exposes ``grad_fn``."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"<{self._name} object>"


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

    __slots__ = ("_t", "_grad")

    def __init__(self, *args, **kwargs) -> None:
        raise _not_yet("the legacy grilly.Tensor(...) constructor; use grilly.tensor(...)")

    @classmethod
    def _wrap(cls, handle) -> "Tensor":
        t = object.__new__(cls)
        t._t = handle
        t._grad = None
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
    def device(self) -> device:
        return _DEVICE

    is_cuda = False

    def cpu(self) -> "Tensor":
        return self  # "cpu" is an alias for grilly's device

    def cuda(self, *args, **kwargs) -> "Tensor":
        raise _not_yet("Tensor.cuda() (grilly runs on Vulkan)")

    def to(self, *args, **kwargs) -> "Tensor":
        """``to(dtype)``, ``to(device, dtype)``, ``to(other)``; returns self when
        nothing changes, as torch does."""
        target_dtype, target_device = kwargs.pop("dtype", None), kwargs.pop("device", None)
        copy = kwargs.pop("copy", False)
        kwargs.pop("non_blocking", None)
        kwargs.pop("memory_format", None)
        if kwargs:
            raise TypeError(f"to() got unexpected keyword arguments {sorted(kwargs)}")
        for a in args:
            if isinstance(a, Tensor):
                target_dtype, target_device = a.dtype, a.device
            elif isinstance(a, dtype):
                target_dtype = a
            elif isinstance(a, (str, device, _builtins.int)):
                target_device = a
            elif isinstance(a, _builtins.bool):
                pass  # non_blocking passed positionally
            else:
                raise TypeError(f"to() received an invalid argument {a!r}")
        _resolve_device(target_device)
        out = self
        if target_dtype is not None and target_dtype is not self.dtype:
            out = _cast(self, target_dtype)
        return out.clone() if copy and out is self else out

    # autograd
    @property
    def requires_grad(self) -> _builtins.bool:
        return self._t.requires_grad

    @requires_grad.setter
    def requires_grad(self, value) -> None:
        self._t.set_requires_grad(_builtins.bool(value))

    def requires_grad_(self, requires_grad: _builtins.bool = True) -> "Tensor":
        self._t.set_requires_grad(_builtins.bool(requires_grad))
        return self

    @property
    def is_leaf(self) -> _builtins.bool:
        return self._t.is_leaf

    @property
    def grad_fn(self):
        name = self._t.grad_fn_name
        return None if name is None else _GradFn(name)

    @property
    def grad(self):
        handle = self._t.grad
        if handle is None:
            self._grad = None
            return None
        # Hand back the same object while it is the same tensor, as torch does.
        if self._grad is None or not self._grad._t.same_tensor(handle):
            self._grad = Tensor._wrap(handle)
        return self._grad

    @grad.setter
    def grad(self, value) -> None:
        if value is not None:
            _require_tensor("grad", value)
        self._t.set_grad(None if value is None else value._t)
        self._grad = value

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None) -> None:
        if create_graph:
            raise _not_yet("backward(create_graph=True)")
        if inputs is not None:
            raise _not_yet("backward(inputs=...)")
        g = None if gradient is None else _require_tensor("gradient", gradient)._t
        self._t.backward(g, _builtins.bool(retain_graph) if retain_graph is not None else create_graph)

    def detach(self) -> "Tensor":
        return Tensor._wrap(self._t.detach())

    def clone(self) -> "Tensor":
        return Tensor._wrap(self._t.clone_data())

    @property
    def _version(self) -> _builtins.int:
        return self._t.version

    def __deepcopy__(self, memo):
        """A copy with its own storage, as torch's: leaves only; keeps
        requires_grad and a copy of .grad."""
        if not self.is_leaf:
            raise RuntimeError("Only Tensors created explicitly by the user (graph leaves) support "
                               "the deepcopy protocol at the moment.")
        if id(self) in memo:
            return memo[id(self)]
        new = type(self)._wrap(self._t.detach().clone_data())
        new.requires_grad_(self.requires_grad)
        if self.grad is not None:
            new.grad = self.grad.__deepcopy__(memo)
        memo[id(self)] = new
        return new

    def __reduce_ex__(self, protocol):
        raise _not_yet("pickling tensors")

    # in place
    def add_(self, other, *, alpha=1) -> "Tensor":
        if alpha != 1:
            raise _not_yet("add_ with alpha != 1")
        other = _require_tensor("other", other)
        if other.shape != self.shape:
            raise _not_yet(f"broadcasting in add_ ({list(self.shape)} vs {list(other.shape)})")
        self._t.add_(other._t)
        return self

    def fill_(self, value) -> "Tensor":
        self._t.fill_(_builtins.float(value))
        return self

    def zero_(self) -> "Tensor":
        return self.fill_(0.0)

    def copy_(self, src) -> "Tensor":
        self._t.copy_(_require_tensor("src", src)._t)
        return self

    def mul_(self, other) -> "Tensor":
        if not isinstance(other, (_builtins.int, _builtins.float)):
            raise _not_yet("mul_ by a tensor")
        self._t.mul_(_builtins.float(other))
        return self

    def normal_(self, mean=0.0, std=1.0, *, generator=None) -> "Tensor":
        self._t.normal_(_builtins.float(mean), _builtins.float(std), _generator(generator)._g)
        return self

    def uniform_(self, from_=0.0, to=1.0, *, generator=None) -> "Tensor":
        self._t.uniform_(_builtins.float(from_), _builtins.float(to), _generator(generator)._g)
        return self

    def _dim_index(self, dim: _builtins.int) -> _builtins.int:
        n = self.dim()
        if not -n <= dim < n or n == 0:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{-n}, {n - 1}], but got {dim})"
            )
        return dim % n

    # host conversion
    def numpy(self) -> _np.ndarray:
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. "
                "Use tensor.detach().numpy() instead."
            )
        return self._host()

    def _host(self) -> _np.ndarray:
        data = _np.frombuffer(self._t.to_bytes(), dtype="<f4")
        return data.reshape(self._t.shape).copy()

    def tolist(self):
        return self._host().tolist()

    def item(self) -> _builtins.float:
        if self.numel() != 1:
            raise RuntimeError(
                f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
            )
        return _builtins.float(self._host().reshape(()))

    def __len__(self) -> _builtins.int:
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._t.shape[0]

    def __repr__(self) -> str:
        body = _np.array2string(
            self._host(), separator=", ", prefix="tensor(",
            formatter={"float_kind": lambda v: f"{v:.4f}"},
        )
        name = self._t.grad_fn_name
        if name is not None:
            return f"tensor({body}, grad_fn=<{name}>)"
        if self.requires_grad:
            return f"tensor({body}, requires_grad=True)"
        return f"tensor({body})"

    # views
    def reshape(self, *shape) -> "Tensor":
        return Tensor._wrap(self._t.reshape(_resolve_shape(_size_args(shape), self.numel())))

    def view(self, *shape) -> "Tensor":
        return self.reshape(*shape)

    # ops
    def relu(self) -> "Tensor":
        return relu(self)

    # Arithmetic, reductions, shape ops, indexing and operators are
    # installed from grilly._ops.


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
    _check_factory_args(dtype, device)
    if isinstance(data, Tensor):  # torch copies here too, detached
        return _leaf(_from_array(data._host()), requires_grad)
    if isinstance(data, _np.ndarray):
        if data.dtype != _np.float32 and dtype is None:
            raise _not_yet(f"tensors of numpy dtype {data.dtype} (only float32 so far)")
        return _leaf(_from_array(data), requires_grad)
    arr = _np.asarray(data)
    # torch infers float32 from Python floats and int64/bool from ints/bools.
    if arr.dtype.kind == "f" or (dtype is float32 and arr.dtype.kind in "iub"):
        return _leaf(_from_array(arr.astype(_np.float32)), requires_grad)
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
    _check_factory_args(dtype, device)
    return _leaf(Tensor._wrap(_C.Tensor.empty(list(_size_args(size)))), requires_grad)


def full(size, fill_value, *, dtype=None, device=None, requires_grad=False) -> Tensor:
    _check_factory_args(dtype, device)
    if isinstance(fill_value, _builtins.bool) or not isinstance(fill_value, (_builtins.int, _builtins.float)):
        raise _not_yet(f"full with a {type(fill_value).__name__} fill value (only float32 tensors so far)")
    if isinstance(fill_value, _builtins.int) and dtype is None:
        raise _not_yet("full with an int fill value (torch makes an int64 tensor)")
    t = Tensor._wrap(_C.Tensor.full(list(_size_args((size,))), _builtins.float(fill_value)))
    return _leaf(t, requires_grad)


def zeros(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    _check_factory_args(dtype, device)
    return _leaf(Tensor._wrap(_C.Tensor.full(list(_size_args(size)), 0.0)), requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False) -> Tensor:
    _check_factory_args(dtype, device)
    return _leaf(Tensor._wrap(_C.Tensor.full(list(_size_args(size)), 1.0)), requires_grad)


# ── ops ──────────────────────────────────────────────────────────────────

def relu(input: Tensor) -> Tensor:
    return Tensor._wrap(_C.relu(_require_tensor("input", input)._t))


# ── random ───────────────────────────────────────────────────────────────

from .random import Generator, default_generator, initial_seed, manual_seed, seed  # noqa: E402


def _generator(generator) -> Generator:
    g = default_generator if generator is None else generator
    if not isinstance(g, Generator):
        raise TypeError(f"generator must be a grilly.Generator, not {type(g).__name__}")
    return g


def _random(kind: str, size, generator, dtype_, device, requires_grad, out) -> Tensor:
    if out is not None:
        raise _not_yet("the out= argument")
    _check_factory_args(dtype_, device)
    t = Tensor._wrap(getattr(_generator(generator)._g, kind)(list(_size_args(size))))
    return _leaf(t, requires_grad)


def randn(*size, generator=None, out=None, dtype=None, layout=None, device=None,
          requires_grad=False, pin_memory=False) -> Tensor:
    """Standard-normal samples; deterministic per seed, not torch's values."""
    return _random("normal", size, generator, dtype, device, requires_grad, out)


def rand(*size, generator=None, out=None, dtype=None, layout=None, device=None,
         requires_grad=False, pin_memory=False) -> Tensor:
    """Samples uniform in [0, 1); deterministic per seed, not torch's values."""
    return _random("uniform", size, generator, dtype, device, requires_grad, out)


def randn_like(input: Tensor, *, dtype=None, layout=None, device=None,
               requires_grad=False, memory_format=None) -> Tensor:
    return randn(*_require_tensor("input", input).shape, dtype=dtype, device=device,
                 requires_grad=requires_grad)


def rand_like(input: Tensor, *, dtype=None, layout=None, device=None,
              requires_grad=False, memory_format=None) -> Tensor:
    return rand(*_require_tensor("input", input).shape, dtype=dtype, device=device,
                requires_grad=requires_grad)


def randint(*args, **kwargs):
    raise NotImplementedError(
        "grilly: randint is not implemented yet (it returns int64 tensors, "
        "and integer dtypes are not implemented yet)"
    )


# ── tensor operations ────────────────────────────────────────────────────

from . import _ops  # noqa: E402  (needs Tensor defined)
from ._ops import (  # noqa: E402,F401
    abs, add, allclose, amax, amin, bmm, cat, chunk, clamp, clamp_max, clamp_min, clip,
    concat, concatenate, contiguous, div, divide, empty_like, exp, expand, flatten,
    full_like, log, log_softmax, matmul, max, maximum, mean, min, minimum, mm, mul,
    multiply, narrow, neg, negative, ones_like, permute, pow, rsqrt, select, sigmoid,
    sign, softmax, split, sqrt, square, squeeze, stack, sub, subtract, sum, swapaxes,
    swapdims, t, tanh, transpose, true_divide, unsqueeze, zeros_like,
)

_ops._install()
__all__ += [n for n in dir(_ops) if not n.startswith("_") and callable(getattr(_ops, n))
            and n not in __all__ and n not in ("Tensor",)]

# ── autograd ─────────────────────────────────────────────────────────────

from .autograd import enable_grad, is_grad_enabled, no_grad, set_grad_enabled  # noqa: E402

from . import nn  # noqa: E402  (needs Tensor defined)
from . import utils  # noqa: E402
from . import optim  # noqa: E402
from . import cuda  # noqa: E402
