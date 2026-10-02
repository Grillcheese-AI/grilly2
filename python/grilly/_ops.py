"""torch's tensor operations on grilly's kernels: argument conventions
(negative dims, Python numbers as operands, ``dim=None``) are handled here;
the math and its gradients are in the Rust core. Each function is exported
at grilly's top level, and most are installed as ``Tensor`` methods."""

from __future__ import annotations

import builtins
import math

import numpy as np

from . import Tensor, _C, _lazy, _not_yet, _require_tensor

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
    x = _require_tensor("input", x)
    a, b = builtins.float(a), builtins.float(b)
    if _lazy.should_capture_unary(x):
        return _lazy.LazyTensor._unary(x, op, a, b)
    return _w(_C.unary(x._t, op, a, b))


def _binary(a, b, op) -> Tensor:
    if _lazy.should_capture_binary(a, b):
        return _lazy.LazyTensor._binary(op, a, b)
    return _w(_C.binary(a._t, b._t, op))


# ── type promotion ───────────────────────────────────────────────────────

def _dt():
    from . import bool as b_, float32, int32, int64
    return b_, int32, int64, float32


def _category(dt) -> builtins.int:
    b_, i32, i64, f32 = _dt()
    return 0 if dt is b_ else (2 if dt is f32 else 1)


def _promote(a, b):
    b_, i32, i64, f32 = _dt()
    order = {b_: 0, i32: 1, i64: 2, f32: 3}
    return a if order[a] >= order[b] else b


def _number_dtype(v):
    b_, i32, i64, f32 = _dt()
    if isinstance(v, (builtins.bool, np.bool_)):
        return b_
    if isinstance(v, (builtins.int, np.integer)):
        return i64
    return f32


def result_type(*operands):
    """torch's promotion: dimensioned tensors, then 0-d tensors, then Python
    numbers; a later tier only matters if it is of a higher category (bool <
    int < float), and a Python number contributes its category's default."""
    tiers = [
        [o.dtype for o in operands if isinstance(o, Tensor) and o.dim() > 0],
        [o.dtype for o in operands if isinstance(o, Tensor) and o.dim() == 0],
        [_number_dtype(o) for o in operands if _is_number(o)],
    ]
    result = None
    for tier in tiers:
        if not tier:
            continue
        t = tier[0]
        for d in tier[1:]:
            t = _promote(t, d)
        if result is None or _category(t) > _category(result):
            result = t
    return result


def _as(v, dt) -> Tensor:
    """A tensor of dtype `dt`: a cast tensor, or a 0-d tensor for a number."""
    if _is_number(v):
        from . import _code
        return _w(_C.Tensor.full([], builtins.float(v), _code(dt)))
    t = _require_tensor("input", v)
    return t if t.dtype is dt else t.to(dt)


def _is_float(dt) -> builtins.bool:
    return _category(dt) == 2


# ── arithmetic ───────────────────────────────────────────────────────────

def _arith(input, other, float_op, int_op, scalar=None) -> Tensor:
    """`float_op`/`int_op` on the promoted operands. `scalar(tensor, number)`,
    when given, is the float fast path for a Python number operand."""
    rt = result_type(input, other)
    if rt is None:
        raise TypeError("expected at least one Tensor operand")
    if _is_float(rt):
        if scalar is not None:
            if _is_number(other):
                return scalar(_as(input, rt), other, False)
            if _is_number(input):
                return scalar(_as(other, rt), input, True)
        return _binary(_as(input, rt), _as(other, rt), float_op)
    from . import _code
    return _w(_C.integer(_as(input, rt)._t, _as(other, rt)._t, int_op, _code(rt)))


def add(input, other, *, alpha=1) -> Tensor:
    if alpha != 1:
        other = mul(other, alpha)
    return _arith(input, other, "add", "add",
                  lambda t, v, _: _unary(t, "add_scalar", v))


def sub(input, other, *, alpha=1) -> Tensor:
    if alpha != 1:
        other = mul(other, alpha)
    b_ = _dt()[0]
    if builtins.all(isinstance(x, Tensor) and x.dtype is b_ for x in (input, other)):
        raise RuntimeError("Subtraction, the `-` operator, with two bool tensors is not supported. "
                           "Use the `^` or `logical_xor()` operator instead.")
    return _arith(input, other, "sub", "sub",
                  lambda t, v, rev: _unary(t, "rsub_scalar", v) if rev else _unary(t, "add_scalar", -v))


def mul(input, other) -> Tensor:
    return _arith(input, other, "mul", "mul",
                  lambda t, v, _: _unary(t, "mul_scalar", v))


def _int_division(input, other, op, name) -> Tensor:
    """Integer floor/trunc division or remainder, exact, with torch's signs.

    Float operands take the composed route below: the integer kernels
    divide integers, and a float `%` is not a different rounding rule but
    the same formula over a different dtype.
    """
    rt = result_type(input, other)
    if rt is None:
        raise TypeError("expected at least one Tensor operand")
    if _is_float(rt):
        return _float_division(input, other, op, name)
    return _arith(input, other, None, op)


def _float_division(input, other, op, name) -> Tensor:
    """``floor_divide``/``trunc_divide``/``remainder``/``fmod`` on floats.

    Composed from divide-and-round rather than given a kernel, because a
    float ``%`` is rare enough that a handful of dispatches is cheaper
    than a shader to maintain. The one use that matters — a ring-buffer
    slot, ``pos % window``, kept in a float32 counter so a decode step can
    be captured — is a scalar.

    **The correction is not optional.** ``input - floor(input / other) *
    other`` is the formula torch documents, and on its own it is wrong:
    the divide rounds, so ``floor`` can come out one too high or one too
    low, and the remainder then falls outside ``[0, other)`` — measured
    disagreeing with torch at operands as small as 2**12 against 3, and
    with a *negative* result, which is not a rounding difference but a
    wrong answer. So the quotient's off-by-one is repaired explicitly:
    a remainder whose sign disagrees with the divisor gains one divisor,
    and one that reaches the divisor's magnitude loses one. Both repairs
    are exact (Sterbenz), and they are the same two lines torch's own
    remainder applies on top of an exact ``fmod``.

    What remains is the composition's real limit: ``floor(input/other) *
    other`` loses low bits once that product passes the mantissa, and no
    repair recovers them. Measured against torch over 8192 random values
    per decade: **bit-exact while the dividend is within 2**24**, which
    is the regime a float32 position counter lives in, and bit-exact all
    the way to 2**39 when the divisor is a power of two, because then the
    product is exact whatever the quotient. Outside that it is wrong
    rather than imprecise — 2 at a dividend of 2**25 against 3, growing
    with the dividend. ``tests/python/test_float_mod.py`` holds the
    measurement.
    """
    q = div(input, other)
    if op == "floor_divide":
        return floor(q)
    if op == "trunc_divide":
        return trunc(q)
    if op == "fmod":
        # fmod takes the dividend's sign, remainder the divisor's, so one
        # implementation serves both: give remainder a divisor already
        # signed like the dividend.
        mag = _magnitude(other)
        signed = where(_is_negative(input), neg(mag) if isinstance(mag, Tensor) else -mag, mag)
        return _float_division(input, signed, "remainder", name)
    r = sub(input, mul(floor(q), other))
    wrong_sign = logical_and(ne(r, 0.0), _sign_differs(r, other))
    r = where(wrong_sign, add(r, other), r)
    return where(ge(abs(r), _magnitude(other)), sub(r, other), r)


def _magnitude(x):
    """``abs`` over a tensor or a number, staying whichever it was."""
    return abs(x) if isinstance(x, Tensor) else builtins.abs(x)


def _is_negative(x):
    """``x < 0`` as a tensor, or as a Python bool for a number."""
    return lt(x, 0.0) if isinstance(x, Tensor) else x < 0


def _sign_differs(a, b) -> Tensor:
    """``(a < 0) != (b < 0)``, with either side allowed to be a number."""
    a_neg, b_neg = _is_negative(a), _is_negative(b)
    if isinstance(a_neg, bool):
        a_neg, b_neg = b_neg, a_neg
    if b_neg is True:
        return logical_not(a_neg)
    if b_neg is False:
        return a_neg
    return logical_xor(a_neg, b_neg)


def floor_divide(input, other) -> Tensor:
    return _int_division(input, other, "floor_divide", "floor_divide")


def remainder(input, other) -> Tensor:
    """``input - floor(input / other) * other``: the result takes the
    divisor's sign, as Python's ``%`` does."""
    return _int_division(input, other, "remainder", "remainder")


def fmod(input, other) -> Tensor:
    """``input - trunc(input / other) * other``: the result takes the
    *dividend's* sign, as C's ``fmod`` does. The difference from
    :func:`remainder` shows only when the operands' signs disagree."""
    if _is_float(result_type(input, other) or _dt()[3]):
        return _float_division(input, other, "fmod", "fmod")
    # integers have a truncating-divide kernel but no truncating remainder,
    # and the identity is exact over integers, so compose it
    return sub(input, mul(_int_division(input, other, "trunc_divide", "fmod"), other))


def _rounded(input, op) -> Tensor:
    x = _require_tensor("input", input)
    return _unary(x, op) if _is_float(x.dtype) else x.clone()


def floor(input) -> Tensor:
    return _rounded(input, "floor")


def ceil(input) -> Tensor:
    return _rounded(input, "ceil")


def trunc(input) -> Tensor:
    return _rounded(input, "trunc")


def round(input, *, decimals=0) -> Tensor:  # noqa: A001
    """To the nearest integer, halves to even (torch.round)."""
    if decimals:
        raise _not_yet("round(decimals=...)")
    return _rounded(input, "round")


def div(input, other, *, rounding_mode=None) -> Tensor:
    if rounding_mode == "floor":
        return floor_divide(input, other)
    if rounding_mode == "trunc":
        return _int_division(input, other, "trunc_divide", "div(rounding_mode='trunc')")
    if rounding_mode is not None:
        raise ValueError(f"div expected rounding_mode to be one of None, 'trunc', or 'floor' but found '{rounding_mode}'")
    f32 = _dt()[3]
    if _is_number(input):  # true division is always floating
        return _unary(_as(other, f32), "rdiv_scalar", input)
    return _binary(_as(input, f32), _as(other, f32), "div")


true_divide = divide = div
multiply = mul
subtract = sub


def pow(input, exponent) -> Tensor:
    rt = result_type(input, exponent)
    if not _is_float(rt):
        raise _not_yet("integer pow")
    if _is_number(exponent):
        return _unary(_as(input, rt), "pow_scalar", exponent)
    return _binary(_as(input, rt), _as(exponent, rt), "pow")


def maximum(input, other) -> Tensor:
    _require_tensor("input", input)
    _require_tensor("other", other)
    return _arith(input, other, "maximum", "maximum")


def minimum(input, other) -> Tensor:
    _require_tensor("input", input)
    _require_tensor("other", other)
    return _arith(input, other, "minimum", "minimum")


def neg(input) -> Tensor:
    x = _require_tensor("input", input)
    if x.dtype is _dt()[0]:
        raise RuntimeError("Negation, the `-` operator, on a bool tensor is not supported. "
                           "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.")
    if _is_float(x.dtype):
        return _unary(x, "neg")
    return sub(0, x)


negative = neg


def abs(input) -> Tensor:  # noqa: A001 - torch.abs
    x = _require_tensor("input", input)
    if _is_float(x.dtype):
        return _unary(x, "abs")
    return x if x.dtype is _dt()[0] else maximum(x, neg(x))


def sign(input) -> Tensor:
    x = _require_tensor("input", input)
    if _is_float(x.dtype):
        return _unary(x, "sign")
    return sub(_compare(x, 0, "gt").to(x.dtype), _compare(x, 0, "lt").to(x.dtype))


def _float_unary(name):
    """Transcendental ops: integer inputs are computed in float32, as torch does."""
    def op(input) -> Tensor:
        x = _require_tensor("input", input)
        return _unary(x if _is_float(x.dtype) else x.to(_dt()[3]), name)
    op.__name__ = name
    return op


exp = _float_unary("exp")
log = _float_unary("log")
sqrt = _float_unary("sqrt")
rsqrt = _float_unary("rsqrt")
tanh = _float_unary("tanh")
sigmoid = _float_unary("sigmoid")
sin = _float_unary("sin")
cos = _float_unary("cos")


def square(input) -> Tensor:
    x = _require_tensor("input", input)
    return mul(x, x) if not _is_float(x.dtype) else _unary(x, "pow_scalar", 2.0)


def clamp(input, min=None, max=None) -> Tensor:  # noqa: A002 - torch's names
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")
    x = _require_tensor("input", input)
    if isinstance(min, Tensor) or isinstance(max, Tensor) or not _is_float(x.dtype):
        out = x
        if min is not None:
            out = maximum(out, _as(min, x.dtype))
        if max is not None:
            out = minimum(out, _as(max, x.dtype))
        return out
    lo = -math.inf if min is None else min
    hi = math.inf if max is None else max
    return _unary(x, "clamp", lo, hi)


clip = clamp


def clamp_min(input, min) -> Tensor:  # noqa: A002
    return clamp(input, min=min)


def clamp_max(input, max) -> Tensor:  # noqa: A002
    return clamp(input, max=max)


def outer(input, vec2) -> Tensor:
    a, b = _require_tensor("input", input), _require_tensor("vec2", vec2)
    if a.dim() != 1 or b.dim() != 1:
        raise RuntimeError("outer: expected 1-D tensors")
    return mul(a.reshape(-1, 1), b.reshape(1, -1))


ger = outer


# ── comparisons, logic and selection ────────────────────────────────────

def _compare(a, b, op) -> Tensor:
    rt = result_type(a, b)
    return _w(_C.compare(_as(a, rt)._t, _as(b, rt)._t, op))


def eq(input, other) -> Tensor:
    return _compare(input, other, "eq")


def ne(input, other) -> Tensor:
    return _compare(input, other, "ne")


def gt(input, other) -> Tensor:
    return _compare(input, other, "gt")


def ge(input, other) -> Tensor:
    return _compare(input, other, "ge")


def lt(input, other) -> Tensor:
    return _compare(input, other, "lt")


def le(input, other) -> Tensor:
    return _compare(input, other, "le")


greater, greater_equal, less, less_equal, not_equal = gt, ge, lt, le, ne


def _bitwise(a, b, op) -> Tensor:
    rt = result_type(a, b)
    if _is_float(rt):
        raise TypeError(f"bitwise {op} is only supported for integer and boolean tensors")
    from . import _code
    return _w(_C.integer(_as(a, rt)._t, _as(b, rt)._t, op, _code(rt)))


def bitwise_and(input, other) -> Tensor:
    return _bitwise(input, other, "and")


def bitwise_or(input, other) -> Tensor:
    return _bitwise(input, other, "or")


def bitwise_xor(input, other) -> Tensor:
    return _bitwise(input, other, "xor")


def bitwise_not(input) -> Tensor:
    x = _require_tensor("input", input)
    if _is_float(x.dtype):
        raise TypeError("bitwise_not is only supported for integer and boolean tensors")
    return _bitwise(x, True if x.dtype is _dt()[0] else -1, "xor")


def _as_bool(x) -> Tensor:
    return _require_tensor("input", x).to(_dt()[0])


def logical_and(input, other) -> Tensor:
    return _bitwise(_as_bool(input), _as_bool(other), "and")


def logical_or(input, other) -> Tensor:
    return _bitwise(_as_bool(input), _as_bool(other), "or")


def logical_xor(input, other) -> Tensor:
    return _bitwise(_as_bool(input), _as_bool(other), "xor")


def logical_not(input) -> Tensor:
    return _bitwise(_as_bool(input), True, "xor")


def isfinite(input) -> Tensor:
    return _w(_C.predicate(_require_tensor("input", input)._t, "isfinite"))


def isinf(input) -> Tensor:
    return _w(_C.predicate(_require_tensor("input", input)._t, "isinf"))


def isnan(input) -> Tensor:
    return _w(_C.predicate(_require_tensor("input", input)._t, "isnan"))


def where(condition, input=None, other=None) -> Tensor:
    if input is None and other is None:
        raise _not_yet("where(condition) (it returns int64 index tensors of nonzero entries)")
    cond = _require_tensor("condition", condition)
    if cond.dtype is not _dt()[0]:
        raise RuntimeError(f"where expected condition to be a boolean tensor, but got a tensor with dtype {cond.dtype!r}")
    rt = result_type(input, other)
    return _w(_C.where_(cond._t, _as(input, rt)._t, _as(other, rt)._t))


def masked_fill(input, mask, value) -> Tensor:
    x = _require_tensor("input", input)
    return where(_require_tensor("mask", mask), _as(value, x.dtype), x)


def equal(input, other) -> builtins.bool:
    a, b = _require_tensor("input", input), _require_tensor("other", other)
    return tuple(a.shape) == tuple(b.shape) and builtins.bool(all_(eq(a, b)).item())


def all_(input, dim=None, keepdim=False) -> Tensor:
    x = _as_bool(input)
    return _w(_C.integer_reduce(x._t, _dims(dim, x.dim()), "min", builtins.bool(keepdim))).to(_dt()[0])


def any_(input, dim=None, keepdim=False) -> Tensor:
    x = _as_bool(input)
    return _w(_C.integer_reduce(x._t, _dims(dim, x.dim()), "max", builtins.bool(keepdim))).to(_dt()[0])


# ── reductions ───────────────────────────────────────────────────────────

def _reduce_dtype(x, dtype):
    if dtype is not None:
        from . import _code
        _code(dtype)
        return x if x.dtype is dtype else x.to(dtype)
    return x


def sum(input, dim=None, keepdim=False, *, dtype=None) -> Tensor:  # noqa: A001
    x = _reduce_dtype(_require_tensor("input", input), dtype)
    if not _is_float(x.dtype):  # torch sums integers and bools as int64
        return _w(_C.integer_reduce(x._t, _dims(dim, x.dim()), "sum", builtins.bool(keepdim)))
    return _w(_C.sum(x._t, _dims(dim, x.dim()), builtins.bool(keepdim)))


def mean(input, dim=None, keepdim=False, *, dtype=None) -> Tensor:
    x = _reduce_dtype(_require_tensor("input", input), dtype)
    if not _is_float(x.dtype):
        raise RuntimeError(f"mean(): could not infer output dtype. Input dtype must be either a floating "
                           f"point or complex dtype. Got: {repr(x.dtype).split('.')[-1].capitalize()}")
    return _w(_C.mean(x._t, _dims(dim, x.dim()), builtins.bool(keepdim)))


def _extreme(input, dim, keepdim, is_max) -> Tensor:
    x = _require_tensor("input", input)
    if _is_float(x.dtype):
        return _w(_C.amax(x._t, _dims(dim, x.dim()), builtins.bool(keepdim), is_max))
    r = _w(_C.integer_reduce(x._t, _dims(dim, x.dim()), "max" if is_max else "min", builtins.bool(keepdim)))
    return r.to(x.dtype)


def amax(input, dim=(), keepdim=False) -> Tensor:
    return _extreme(input, dim, keepdim, True)


def amin(input, dim=(), keepdim=False) -> Tensor:
    return _extreme(input, dim, keepdim, False)


def _zero_index(x: Tensor, dim) -> Tensor:
    """The only index of a 0-d tensor (dim must be 0 or -1, as in torch)."""
    _dim(dim, 1)
    from . import zeros
    return zeros((), dtype=_dt()[2])


def _max_dim(x: Tensor, dim, keepdim, is_max):
    from . import return_types
    cls = return_types.max if is_max else return_types.min
    if x.dim() == 0:
        return cls(x, _zero_index(x, dim))
    v, i = _C.max_dim(x._t, _dim(dim, x.dim()), builtins.bool(keepdim), is_max)
    return cls(_w(v), _w(i))


def max(input, dim=None, keepdim=False):  # noqa: A001
    """``max(x)``: the largest value; ``max(x, dim)``: (values, indices),
    the first largest along ``dim``; ``max(a, b)``: elementwise maximum."""
    if isinstance(dim, Tensor):
        return maximum(input, dim)  # torch.max(a, b)
    if dim is None:
        return amax(input)
    return _max_dim(_require_tensor("input", input), dim, keepdim, True)


def min(input, dim=None, keepdim=False):  # noqa: A001
    if isinstance(dim, Tensor):
        return minimum(input, dim)
    if dim is None:
        return amin(input)
    return _max_dim(_require_tensor("input", input), dim, keepdim, False)


def _arg(input, dim, keepdim, is_max) -> Tensor:
    x = _require_tensor("input", input)
    if dim is None:
        if x.numel() == 0:
            raise IndexError(f"{'argmax' if is_max else 'argmin'}(): Expected reduction dim to be specified "
                             "for input.numel() == 0.")
        flat = x.reshape(-1)
        r = _w(_C.argmax(flat._t, 0, False, is_max))
        return r.reshape([1] * x.dim()) if keepdim else r
    if x.dim() == 0:
        return _zero_index(x, dim)
    return _w(_C.argmax(x._t, _dim(dim, x.dim()), builtins.bool(keepdim), is_max))


def argmax(input, dim=None, keepdim=False) -> Tensor:
    """The index of the first largest value (NaN counts as largest), of the
    flattened tensor when ``dim`` is None."""
    return _arg(input, dim, keepdim, True)


def argmin(input, dim=None, keepdim=False) -> Tensor:
    return _arg(input, dim, keepdim, False)


def _sort_dim(x: Tensor, dim):
    """(dim, its size); a 0-d tensor sorts as one element along dim 0."""
    d = _dim(dim, builtins.max(x.dim(), 1))
    return d, (x.shape[d] if x.dim() else 1)


def _sorted(x: Tensor, dim, descending, take, name):
    v, i = _C.sorted(x._t, _sort_dim(x, dim)[0], builtins.bool(descending), take, name)
    return _w(v), _w(i)


def sort(input, dim=-1, descending=False, stable=False):
    """(values, indices) along ``dim``. Equal values keep their original
    order (so ``stable`` is always honoured) and NaN sorts as the largest."""
    from . import return_types
    x = _require_tensor("input", input)
    return return_types.sort(*_sorted(x, dim, descending, _sort_dim(x, dim)[1], "sort"))


def argsort(input, dim=-1, descending=False, stable=False) -> Tensor:
    return sort(input, dim, descending, stable).indices


def topk(input, k, dim=-1, largest=True, sorted=True):  # noqa: A002
    """The ``k`` largest (or smallest) values along ``dim`` and their
    indices, always in sorted order; ties go to the lower index."""
    from . import return_types
    x = _require_tensor("input", input)
    if not 0 <= k <= _sort_dim(x, dim)[1]:
        raise RuntimeError("selected index k out of range")
    return return_types.topk(*_sorted(x, dim, largest, builtins.int(k), "topk"))


def _float_input(x, op):
    x = _require_tensor("input", x)
    if not _is_float(x.dtype):
        raise RuntimeError(f"\"{op}\" not implemented for '{repr(x.dtype).split('.')[-1].capitalize()}'")
    return x


def softmax(input, dim, dtype=None) -> Tensor:
    x = _float_input(_reduce_dtype(_require_tensor("input", input), dtype), "softmax")
    return _w(_C.softmax(x._t, _dim(dim, x.dim())))


def log_softmax(input, dim, dtype=None) -> Tensor:
    x = _float_input(_reduce_dtype(_require_tensor("input", input), dtype), "log_softmax")
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
    """``x.transpose(dim0, dim1)``.

    A swap of the last two dims is **deferred**: grilly has no strides, so
    building it means copying the whole tensor, and the one thing that
    usually consumes it — a matmul — can read the original in place
    instead (python/grilly/_transposed.py). Anything else materializes it
    on first use.
    """
    from ._transposed import lazy_transpose

    x = _require_tensor("input", input)
    return lazy_transpose(x, _dim(dim0, x.dim()), _dim(dim1, x.dim()))


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
    """``input @ other``.

    When ``other`` is a deferred transpose this reads its base with a
    transpose flag rather than building the copy, which is what torch does
    by passing one to BLAS. ``x @ w.t()`` is then exactly ``linear(x, w)``.
    """
    from ._transposed import transposed_base

    a = _require_tensor("input", input)
    b = _require_tensor("other", other)
    base = transposed_base(b)
    if base is not None and a.dim() >= 1 and base.dim() >= 2:
        return _w(_C.matmul_t(a._t, base._t))
    return _w(_C.matmul(a._t, b._t))


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
    items = idx if isinstance(idx, tuple) else (idx,)
    if builtins.any(_is_advanced(i) for i in items):
        return _advanced(x, items)
    offset, strides, shape, only_ints = _parse_index(x, idx)
    return _w(_C.strided_read(x._t, offset, strides, shape, only_ints))


def _setitem(x: Tensor, idx, value) -> None:
    items = idx if isinstance(idx, tuple) else (idx,)
    if builtins.any(_is_advanced(i) for i in items):
        return _advanced_setitem(x, items, value)
    offset, strides, shape, _ = _parse_index(x, idx)
    if _is_number(value):
        from . import _code
        v = _w(_C.Tensor.full([], builtins.float(value), _code(x.dtype)))
    else:
        v = _require_tensor("value", value)
    _C.write_region(x._t, offset, strides, shape, v._t)


# ── indexing with index tensors ─────────────────────────────────────────

def gather(input, dim, index, *, sparse_grad=False) -> Tensor:
    if sparse_grad:
        raise _not_yet("gather(sparse_grad=True)")
    x, i = _require_tensor("input", input), _require_tensor("index", index)
    return _w(_C.gather(x._t, _dim(dim, x.dim()), i._t))


def index_select(input, dim, index) -> Tensor:
    x, i = _require_tensor("input", input), _require_tensor("index", index)
    return _w(_C.index_select(x._t, _dim(dim, x.dim()), i._t))


def _expand_index(x: Tensor, dim: int, index: Tensor, source: Tensor, name: str) -> Tensor:
    """A 1-D ``index`` over ``dim`` as the full-shaped index ``scatter_``
    wants: the same value repeated across every other dimension."""
    if index.dim() != 1:
        raise RuntimeError(f"{name}(): Index is supposed to be a vector, got {index.dim()}D")
    if source.dim() != x.dim():
        raise RuntimeError(
            f"{name}(): Source/destination tensor must have same number of dimensions, "
            f"got {source.dim()} and {x.dim()}"
        )
    if source.shape[dim] != index.numel():
        raise RuntimeError(
            f"{name}(): Number of indices ({index.numel()}) should be equal to "
            f"source.shape[{dim}] ({source.shape[dim]})"
        )
    shape = [1] * x.dim()
    shape[dim] = index.numel()
    return index.reshape(shape).expand(*source.shape)


def index_copy_(input, dim, index, source) -> Tensor:
    """``self[index[i], ...] = source[i, ...]`` along ``dim``, in place.

    A ring buffer's write. Stays on the device and inside a capture, and
    is **one dispatch**: the kernel reads the 1-D index by the ``dim``
    coordinate rather than being handed a broadcast of it, which used to
    cost a reshape and an expand first. Bounds are validated on the device
    (``grilly_core::fault``).
    """
    x = _require_tensor("input", input)
    i, src = _require_tensor("index", index), _require_tensor("source", source)
    d = _dim(dim, x.dim())
    x._t = _C.index_copy_(x._t, d, i._t, src._t)
    return x


def index_add_(input, dim, index, source, *, alpha=1) -> Tensor:
    """``self[index[i], ...] += alpha * source[i, ...]`` along ``dim``.

    Accumulating, so repeated indices all land — which is the difference
    from ``index_copy_`` and the reason a histogram or a scatter-gather
    uses it.
    """
    x = _require_tensor("input", input)
    i, src = _require_tensor("index", index), _require_tensor("source", source)
    d = _dim(dim, x.dim())
    if alpha != 1:
        src = src * alpha
    return x.add_(_w(_C.index_add_into_zeros(src._t, d, i._t, list(x.shape))))


def index_fill_(input, dim, index, value) -> Tensor:
    """``self[index[i], ...] = value`` along ``dim``, in place."""
    from . import full

    x = _require_tensor("input", input)
    i = _require_tensor("index", index)
    d = _dim(dim, x.dim())
    shape = list(x.shape)
    shape[d] = i.numel()
    return index_copy_(x, d, i, full(builtins.tuple(shape), value, dtype=x.dtype))


def index_copy(input, dim, index, source) -> Tensor:
    """:func:`index_copy_` on a copy."""
    return index_copy_(_require_tensor("input", input).clone(), dim, index, source)


def index_add(input, dim, index, source, *, alpha=1) -> Tensor:
    """:func:`index_add_` on a copy."""
    return index_add_(_require_tensor("input", input).clone(), dim, index, source, alpha=alpha)


def index_fill(input, dim, index, value) -> Tensor:
    """:func:`index_fill_` on a copy."""
    return index_fill_(_require_tensor("input", input).clone(), dim, index, value)


def _scatter_args(x: Tensor, index, src, reduce):
    if reduce is not None:
        raise _not_yet(f"scatter(reduce={reduce!r})")
    i = _require_tensor("index", index)
    if _is_number(src):
        from . import full
        src = full(builtins.tuple(i.shape), src, dtype=x.dtype)
    return i, _require_tensor("src", src)


def scatter(input, dim, index, src=None, *, value=None, reduce=None) -> Tensor:
    """A copy of ``input`` with ``src`` (a tensor or a number) written at
    ``index`` along ``dim``. Duplicate indices leave which write wins
    unspecified, as in torch."""
    x = _require_tensor("input", input)
    i, v = _scatter_args(x, index, value if src is None else src, reduce)
    return _w(_C.scatter(x._t, _dim(dim, x.dim()), i._t, v._t))


def _scatter_(self, dim, index, src=None, *, value=None, reduce=None) -> Tensor:
    i, v = _scatter_args(self, index, value if src is None else src, reduce)
    self._t = _C.scatter_(self._t, _dim(dim, self.dim()), i._t, v._t)
    return self


def cumsum(input, dim, *, dtype=None) -> Tensor:
    """Running sums along ``dim``: float32 stays float32; integers and bools
    give int64, as in torch."""
    x = _require_tensor("input", input)
    if dtype is not None:
        x = x.to(dtype)
    if x.dim() == 0:
        _dim(dim, 1)
        return _w(_C.cumsum(x.reshape(1)._t, 0)).reshape(())
    return _w(_C.cumsum(x._t, _dim(dim, x.dim())))


def logcumsumexp(input, dim) -> Tensor:
    """``log(cumsum(exp(input)))`` along ``dim``, computed stably."""
    x = _require_tensor("input", input)
    if x.dim() == 0:
        _dim(dim, 1)
        return _w(_C.logcumsumexp(x.reshape(1)._t, 0)).reshape(())
    return _w(_C.logcumsumexp(x._t, _dim(dim, x.dim())))


def use_deterministic_algorithms(mode, *, warn_only=False) -> None:
    """When on, ops that would sum with atomics (in hardware order, e.g.
    embedding and gather backward) use fixed-order kernels instead."""
    _C.set_deterministic(builtins.bool(mode))


def are_deterministic_algorithms_enabled() -> builtins.bool:
    return _C.deterministic()


def _is_advanced(item) -> builtins.bool:
    return isinstance(item, (Tensor, list, np.ndarray))


def _index_tensor(item) -> Tensor:
    if isinstance(item, Tensor):
        return item
    from . import tensor
    return tensor(np.asarray(item))


def _mask_indices(x: Tensor, mask: Tensor):
    """Flat positions of a bool mask over x's leading dims (a host sync, as in torch)."""
    if tuple(mask.shape) != tuple(x.shape[:mask.dim()]):
        raise IndexError(f"The shape of the mask {list(mask.shape)} at index 0 does not match the shape of "
                         f"the indexed tensor {list(x.shape)} at index 0")
    from . import tensor
    return tensor(np.flatnonzero(mask.numpy()).astype(np.int64)), mask.dim()


def _expand_ellipsis(x: Tensor, items):
    if builtins.any(i is None for i in items):
        raise _not_yet("advanced indexing combined with None")
    n = builtins.sum(1 for i in items if i is Ellipsis)
    if n > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    used = builtins.sum(_index_tensor(i).dim() if _is_advanced(i) and _index_tensor(i).dtype is _dt()[0] else 1
                        for i in items if i is not Ellipsis)
    if used > x.dim():
        raise IndexError(f"too many indices for tensor of dimension {x.dim()}")
    out = []
    for i in items:
        out.extend([slice(None)] * (x.dim() - used) if i is Ellipsis else [i])
    return out


def _advanced(x: Tensor, items) -> Tensor:
    """torch's advanced indexing: ints, int tensors (or lists) and slices,
    or one bool mask over the leading dims. Ints and slices are applied
    first, as basic indexing (as in torch, unlike numpy, an int is not
    broadcast with the index tensors); the indexed dims are moved to the front, linearized
    into one index, and the broadcast index dims are put back where torch
    puts them: in place if the indexed dims were adjacent, in front if not."""
    b_ = _dt()[0]
    items = _expand_ellipsis(x, items)
    for i in items:
        if _is_advanced(i) and _index_tensor(i).is_floating_point():
            raise IndexError("tensors used as indices must be long, int, byte or bool tensors")
    masks = [k for k, i in enumerate(items) if _is_advanced(i) and _index_tensor(i).dtype is b_]
    if masks:
        full = builtins.all(isinstance(i, slice) and i == slice(None) for i in items[1:])
        if masks != [0] or not full:
            raise _not_yet("bool masks combined with other indices")
        flat, nd = _mask_indices(x, _index_tensor(items[0]))
        lead = math.prod(x.shape[:nd])
        return _w(_C.index_tensor(x.reshape([lead] + list(x.shape[nd:]))._t, 0, flat._t))
    items += [slice(None)] * (x.dim() - len(items))
    basic = lambda i: isinstance(i, (slice, builtins.int))  # noqa: E731
    if builtins.any(basic(i) and i != slice(None) for i in items):
        # ints and slices first, as torch does: an int drops its dim
        x = x[tuple(i if basic(i) else slice(None) for i in items)]
        items = [slice(None) if isinstance(i, slice) else i for i in items if not isinstance(i, builtins.int)]
    adv = [k for k, i in enumerate(items) if not isinstance(i, slice)]
    rest = [k for k in range(x.dim()) if k not in adv]
    if len(adv) == 1:
        return _w(_C.index_tensor(x._t, adv[0], _index_tensor(items[adv[0]])._t))
    idx = [_index_tensor(items[k]) for k in adv]
    flat, stride = None, 1
    for k, t in builtins.reversed(builtins.list(builtins.zip(adv, idx))):
        size = x.shape[k]
        if builtins.bool(((t >= size) | (t < -size)).any()):
            bad = t.max().item() if builtins.bool((t >= size).any()) else t.min().item()
            raise IndexError(f"index {bad} is out of bounds for dimension {k} with size {size}")
        term = where(t < 0, t + size, t) * stride
        flat = term if flat is None else flat + term
        stride *= size
    moved = x if adv + rest == builtins.list(range(x.dim())) else x.permute(*(adv + rest))
    lead = math.prod(x.shape[k] for k in adv)
    picked = _w(_C.index_tensor(moved.reshape([lead] + [x.shape[k] for k in rest])._t, 0, flat._t))
    p, nb = adv[0], flat.dim()
    if p == 0 or adv != builtins.list(range(p, adv[-1] + 1)):
        return picked
    order = builtins.list(range(nb, nb + p)) + builtins.list(range(nb)) + builtins.list(range(nb + p, picked.dim()))
    return picked.permute(*order)


def _numpy_key(item):
    if isinstance(item, Tensor):
        return item.numpy()
    return np.asarray(item) if isinstance(item, list) else item


def _advanced_setitem(x: Tensor, items, value) -> None:
    """A bool mask over the leading dims is a GPU select. Other advanced
    keys (int tensors, lists, mixed with ints and slices) go through the
    host: numpy's assignment semantics are torch's for these keys, and this
    is setup code, not a training path."""
    b_ = _dt()[0]
    if len(items) == 1 and _is_advanced(items[0]):
        mask = _index_tensor(items[0])
        if mask.dtype is b_ and tuple(mask.shape) == tuple(x.shape[:mask.dim()]):
            m = mask.reshape(list(mask.shape) + [1] * (x.dim() - mask.dim()))
            x.copy_(where(m, _as(value, x.dtype), x))
            return
    if len(items) == 1 and _is_advanced(items[0]):
        key = _index_tensor(items[0])
        if key.dtype is not b_ and not key.is_floating_point() and key.dim() == 1 and x.dim() >= 1:
            # x[rows] = value along dim 0: a ring buffer's write, and the one
            # advanced key worth keeping on the device. Everything else below
            # is setup code and goes through numpy.
            shape = [key.numel()] + list(x.shape[1:])
            src = _as(value, x.dtype)
            if not isinstance(src, Tensor) or builtins.tuple(src.shape) != builtins.tuple(shape):
                from . import empty
                src = empty(*shape, dtype=x.dtype).copy_(_as(value, x.dtype))
            index_copy_(x, 0, key, src)
            return
    for i in items:
        if _is_advanced(i) and _index_tensor(i).is_floating_point():
            raise IndexError("tensors used as indices must be long, int, byte or bool tensors")
    host = x.detach().numpy().copy()
    try:
        host[tuple(_numpy_key(i) for i in items)] = _numpy_key(value)
    except IndexError as e:
        raise IndexError(str(e)) from None
    from . import tensor
    x.copy_(tensor(host))


# ── installation as Tensor methods ───────────────────────────────────────

def _install() -> None:
    T = Tensor
    for name in ("index_copy_", "index_add_", "index_fill_", "index_copy", "index_add", "index_fill",
                 "sin", "cos", "add", "sub", "mul", "div", "pow", "neg", "abs", "sign", "exp", "log",
                 "sqrt", "rsqrt", "tanh", "sigmoid", "square", "clamp", "clip", "clamp_min",
                 "clamp_max", "sum", "mean", "amax", "amin", "max", "min", "argmax", "argmin", "sort",
                 "argsort", "topk", "cumsum", "logcumsumexp", "scatter", "floor_divide", "remainder", "fmod",
                 "floor", "ceil", "trunc", "round", "softmax",
                 "log_softmax", "transpose", "swapaxes", "t", "permute", "unsqueeze",
                 "squeeze", "flatten", "expand", "narrow", "select", "chunk", "split",
                 "matmul", "mm", "bmm", "contiguous", "maximum", "minimum", "true_divide",
                 "multiply", "subtract", "negative", "divide"):
        setattr(T, name, globals()[name])
    T.scatter_ = _scatter_
    T.fmod = fmod
    T.__mod__ = remainder
    T.__rmod__ = lambda self, other: remainder(other, self)
    T.__floordiv__ = floor_divide
    T.__rfloordiv__ = lambda self, other: floor_divide(other, self)
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
    T.__eq__, T.__ne__, T.__lt__, T.__le__, T.__gt__, T.__ge__ = eq, ne, lt, le, gt, ge
    T.__hash__ = object.__hash__  # identity, as torch's
    T.__and__ = T.__rand__ = bitwise_and
    T.__or__ = T.__ror__ = bitwise_or
    T.__xor__ = T.__rxor__ = bitwise_xor
    T.__invert__ = bitwise_not
    for name in ("eq", "ne", "gt", "ge", "lt", "le", "greater", "greater_equal", "less", "less_equal",
                 "not_equal", "logical_and", "logical_or", "logical_xor", "logical_not",
                 "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "isfinite", "isinf",
                 "isnan", "masked_fill", "equal", "outer", "ger", "where", "gather", "index_select"):
        setattr(T, name, globals()[name])
    T.all = all_
    T.any = any_
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
