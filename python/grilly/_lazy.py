"""``torch.compile``'s lazy capture (docs/fusion.md, "Layers", item 2):
inside a compiled callable, float32 elementwise ops build a ``LazyTensor``
expression instead of launching a kernel.

``LazyTensor`` is a ``Tensor`` subclass, so no other file needs a hook for
it: the first read of ``._t`` — a matmul, a reduction, indexing, ``.item()``,
``.numpy()``, printing, an in-place op, or the compiled callable's own
return value — walks the expression once into one
``grilly_core::fusion::Graph`` and runs it as a single fused kernel
(``_C.fused``). Real tensors used more than once, and shared
sub-expressions, become one input or node; once a ``LazyTensor`` is
materialized, anything that still references it treats the cached result
as an input, not more work to fuse.
"""

from __future__ import annotations

import math
import threading

from . import Size, Tensor, _C, float32

__all__ = ["LazyTensor", "materialize"]

_local = threading.local()

# Largest expression kept lazy, in ops (a DAG's shared parts count once per
# use, so this over-counts): past it, an operand is materialized before it is
# used again. Bounds recursion in materialize(), generated-shader size
# and compile time, and stops expressions like `x = x + x` in a loop from
# doubling each step.
MAX_NODES = 64


class _Stats:
    __slots__ = ("materializations", "node_counts")

    def __init__(self) -> None:
        self.materializations = 0
        self.node_counts: list = []


_stats = _Stats()


def stats() -> dict:
    """``{"materializations": int, "node_counts": [int, ...]}`` since the
    last :func:`reset_stats` (``grilly.compiler._stats``)."""
    return {"materializations": _stats.materializations, "node_counts": list(_stats.node_counts)}


def reset_stats() -> None:
    _stats.materializations = 0
    _stats.node_counts = []


# ── capture state (per thread, like torch's grad mode) ─────────────────────

def _depth(name: str) -> int:
    return getattr(_local, name, 0)


def push_compiling() -> None:
    _local.compiling = _depth("compiling") + 1


def pop_compiling() -> None:
    _local.compiling = _depth("compiling") - 1


def push_capture() -> None:
    _local.capture = _depth("capture") + 1


def pop_capture() -> None:
    _local.capture = _depth("capture") - 1


def capture_depth() -> int:
    return _depth("capture")


def is_compiling() -> bool:
    """True inside a callable made by ``grilly.compile`` (regardless of
    backend); ``torch.compiler.is_compiling()``."""
    return _depth("compiling") > 0


def is_capturing() -> bool:
    """True when float elementwise ops should build a lazy graph: inside a
    compiled callable whose backend fuses (not "eager"/"aot_eager")."""
    return _depth("capture") > 0


def should_capture_unary(x: Tensor) -> bool:
    return is_capturing() and x.dtype is float32 and not (_C.is_grad_enabled() and x.requires_grad)


def should_capture_binary(a: Tensor, b: Tensor) -> bool:
    if not is_capturing() or a.dtype is not float32 or b.dtype is not float32:
        return False
    return not (_C.is_grad_enabled() and (a.requires_grad or b.requires_grad))


def _broadcast_shape(a: tuple, b: tuple) -> tuple:
    """torch's broadcasting rule, without launching anything: aligned from
    the right, each dim pair must be equal or one of them 1."""
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + tuple(a)
    b = (1,) * (n - len(b)) + tuple(b)
    out = []
    for da, db in zip(a, b):
        if da != db and da != 1 and db != 1:
            raise RuntimeError(
                f"The size of tensor a ({da}) must match the size of tensor b ({db}) "
                "at a non-singleton dimension"
            )
        out.append(db if da == 1 else da)
    return tuple(out)


# ── the lazy tensor itself ──────────────────────────────────────────────────

class LazyTensor(Tensor):
    """An uncomputed float32 elementwise result. Only ``_unary``/``_binary``
    (python/grilly/_ops.py) construct one, and only while capturing."""

    __slots__ = ("_op", "_x", "_y", "_a", "_b", "_bshape", "_handle", "_vx", "_vy", "_size")

    @classmethod
    def _unary(cls, x, op: str, a: float, b: float) -> "LazyTensor":
        t = object.__new__(cls)
        t._op, t._x, t._y = op, x, None
        t._a, t._b = float(a), float(b)
        t._bshape = tuple(x.shape)
        t._handle = None
        t._grad = None
        t._vx, sx = _operand(x)
        t._vy, t._size = None, 1 + sx
        return t

    @classmethod
    def _binary(cls, op: str, x, y) -> "LazyTensor":
        t = object.__new__(cls)
        t._op, t._x, t._y = op, x, y
        t._a = t._b = 0.0
        t._bshape = _broadcast_shape(tuple(x.shape), tuple(y.shape))
        t._handle = None
        t._grad = None
        (t._vx, sx), (t._vy, sy) = _operand(x), _operand(y)
        t._size = 1 + sx + sy
        return t

    # -- answer without materializing --
    @property
    def shape(self) -> Size:
        return Size(self._bshape)

    def dim(self) -> int:
        return len(self._bshape)

    def numel(self) -> int:
        return math.prod(self._bshape)

    @property
    def dtype(self):
        return float32

    @property
    def device(self):
        from . import _DEVICE
        return _DEVICE

    @property
    def requires_grad(self) -> bool:
        return False

    @property
    def is_leaf(self) -> bool:
        return True  # a tensor that doesn't require grad is always a leaf, as in torch

    @property
    def grad_fn(self):
        return None

    # -- materialize on first real use --
    @property
    def _t(self):
        if self._handle is None:
            self._handle = materialize(self)
        return self._handle


def _pending(t) -> bool:
    return isinstance(t, LazyTensor) and t._handle is None


def _operand(t):
    """(version, size) of an operand of a new node. A real operand's version
    counter is recorded, so an in-place write to it before the node runs is
    caught (materialize() raises rather than read the new values). A lazy
    operand past MAX_NODES is materialized first and becomes a real input."""
    if _pending(t) and t._size >= MAX_NODES:
        t._t  # noqa: B018 - materializes
    if _pending(t):
        return None, t._size
    return t._t.version, 0


def materialize(root: LazyTensor) -> Tensor:
    """Walk ``root``'s expression into one fusion graph and run it as a
    single kernel, caching the result on ``root``. An operand that was
    already materialized independently — something else forced it, e.g. a
    matmul mid-chain — is used as a real input instead of being recomputed;
    an operand still lazy is inlined into this same graph, so a chain that
    was never forced early becomes one kernel end to end."""
    if root._handle is not None:
        return root._handle
    nodes: list = []
    real_inputs: list = []
    params: list = []
    node_of: dict = {}

    def emit_param(value: float) -> int:
        idx = len(nodes)
        nodes.append(("param", len(params)))
        params.append(float(value))
        return idx

    def emit(operand) -> int:
        key = id(operand)
        if key in node_of:
            return node_of[key]
        if isinstance(operand, LazyTensor) and operand._handle is None:
            for real, seen in ((operand._x, operand._vx), (operand._y, operand._vy)):
                if seen is not None and real._t.version != seen:
                    raise RuntimeError(
                        "a tensor used in a compiled (fused) computation was modified in place before that "
                        "computation ran; grilly defers elementwise work inside grilly.compile until its "
                        "result is used. Use the result before the in-place write, or write out of place.")
            x_idx = emit(operand._x)
            if operand._y is not None:
                y_idx = emit(operand._y)  # must run before `idx` is fixed: it may append nodes
                idx = len(nodes)
                nodes.append(("binary", operand._op, x_idx, y_idx))
            else:
                a_idx, b_idx = emit_param(operand._a), emit_param(operand._b)
                idx = len(nodes)
                nodes.append(("unary", operand._op, x_idx, a_idx, b_idx))
        else:
            idx = len(nodes)
            nodes.append(("input", len(real_inputs)))
            real_inputs.append(operand._t)
        node_of[key] = idx
        return idx

    out_idx = emit(root)
    handles = _C.fused(nodes, [out_idx], real_inputs, params, list(root._bshape))
    root._handle = handles[0]
    _stats.materializations += 1
    _stats.node_counts.append(len(nodes))
    return root._handle
