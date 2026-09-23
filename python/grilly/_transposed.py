"""``x.t()`` without copying, so that ``x @ w.t()`` can be ``linear(x, w)``.

grilly's tensors are contiguous: it has no strides, so where torch returns
a view of `w` with two dimensions swapped, grilly has to write the elements
out in the new order. For a matmul that copy is pure loss — the matmul is
about to read every element once anyway, and the copy reads *and writes*
them first.

It is worse than pure loss when the matrix is wide. Transposing a
(127996, 512) weight reads it with a stride of 2 KB across 127996
concurrent streams, so every cache line is fetched for one word of it.
Measured on an RX 6750 XT: **6.1 GB/s**, against 196 GB/s for the same
data transposed the other way round, where there are only 512 streams. On
CubbyLLM's retrieval head that turned a 1.4 ms matmul into 90 ms, and one
op into 92% of a token.

So ``t()`` and ``transpose()`` return a :class:`_Transposed` when the swap
is of the last two dimensions: a tensor that has not been built yet,
holding the base and the shape it would have. Matmul consumes it directly
through ``_C.matmul_t``, which reads the base in place — what torch does
by handing BLAS a transpose flag.

**Anything else materializes it**, and that is why this is a subclass:
every method it does not override reads ``self._t``, which is deliberately
unset, so ``__getattr__`` fires and builds the copy. A path that forgets
about laziness gets the old behaviour rather than a wrong answer. Only
``shape``, ``dim``, ``numel``, ``dtype`` and ``requires_grad`` are answered
without it, because they are questions about the shape rather than the
data, and matmul asks them.

``permute`` stays eager even when its permutation happens to be a swap of
the last two dims: it is a rarer spelling, and one route through this is
enough to keep correct.
"""

from __future__ import annotations

import builtins as _builtins
import math as _math

from . import Size, Tensor, _C

__all__ = ["lazy_transpose"]


class _Transposed(Tensor):
    """A transpose that has not been paid for yet.

    ``_t`` is left unset; touching it builds the copy exactly once.
    """

    __slots__ = ("_base", "_dims", "_shape", "_grad_when_made")

    @classmethod
    def of(cls, base: Tensor, d0: _builtins.int, d1: _builtins.int) -> "_Transposed":
        t = object.__new__(cls)
        t._grad = None
        t._base = base
        t._dims = (d0, d1)
        shape = list(base.shape)
        shape[d0], shape[d1] = shape[d1], shape[d0]
        t._shape = tuple(shape)
        # Whether the eager transpose would have recorded an autograd node.
        # Materializing later under no_grad must not quietly drop it.
        from . import is_grad_enabled

        t._grad_when_made = is_grad_enabled() and base.requires_grad
        return t

    # ── the one thing that is not paid for ───────────────────────────────

    def __getattr__(self, name: str):
        if name != "_t":
            raise AttributeError(name)
        from . import enable_grad, is_grad_enabled

        if self._grad_when_made and not is_grad_enabled():
            with enable_grad():
                handle = _C.transpose(self._base._t, *self._dims)
        else:
            handle = _C.transpose(self._base._t, *self._dims)
        self._t = handle
        return handle

    @property
    def materialized(self) -> _builtins.bool:
        """Whether the copy has been built. For tests, not for callers."""
        try:
            object.__getattribute__(self, "_t")
        except AttributeError:
            return False
        return True

    # ── questions about the shape, which need no data ────────────────────

    @property
    def shape(self) -> Size:
        return Size(self._shape)

    def dim(self) -> _builtins.int:
        return len(self._shape)

    ndim = property(dim)

    def numel(self) -> _builtins.int:
        return _math.prod(self._shape)

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def requires_grad(self) -> _builtins.bool:
        return self._base.requires_grad

    @requires_grad.setter
    def requires_grad(self, value) -> None:
        Tensor.requires_grad.__set__(self, value)      # materializes, as it must

    def __repr__(self) -> str:
        if self.materialized:
            return Tensor.__repr__(self)
        return f"tensor(<transpose of {tuple(self._base.shape)}, not yet materialized>)"


def lazy_transpose(x: Tensor, d0: _builtins.int, d1: _builtins.int) -> Tensor:
    """``x.transpose(d0, d1)``, deferred when it is the last two dims.

    Only that case, because it is the only one matmul can absorb: every
    other permutation has to be written out whatever reads it next.
    """
    n = x.dim()
    if n < 2 or d0 == d1 or {d0, d1} != {n - 2, n - 1}:
        return Tensor._wrap(_C.transpose(x._t, d0, d1))
    base = transposed_base(x)
    if base is not None:
        return base            # two transposes of the same pair cancel
    return _Transposed.of(x, d0, d1)


def transposed_base(x) -> "Tensor | None":
    """The tensor ``x`` is the last-two-dims transpose of, if it is one and
    has not been materialized. ``None`` otherwise — including once it has
    been built, since then reading it costs nothing extra."""
    if type(x) is _Transposed and not x.materialized:
        return x._base
    return None
