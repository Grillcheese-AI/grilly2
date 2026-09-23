"""Sparse block-code VSA (IBM NVSA), on the GPU.

A hypervector is ``(..., k, l)`` float32: k blocks of l positions. A
**discrete** code is one-hot per block; a **continuous** one is a
distribution over each block's positions. Binding is per-block circular
convolution, which keeps that structure exactly — bind two one-hot blocks
and the result is one-hot at the sum of their positions, mod l — so binding
is invertible and a bound code is still a code.

    import grilly
    from grilly import vsa

    role = vsa.random_discrete(80, 128)          # (80, 128), one hot per block
    filler = vsa.random_discrete(80, 128)
    bound = vsa.bind(role, filler)
    assert vsa.similarity(vsa.unbind(bound, role), filler) == 1.0

Everything takes and returns ``grilly.Tensor``. ``grilly.from_numpy`` and
``.numpy()`` cross the boundary; the deprecated ``grilly.experimental.vsa``
aliases do it for you, with v1's numpy-in, numpy-out signatures.

**This is not the packed-binary algebra.** ``grilly.vsa.packed`` holds that
one — bits, XOR, Hamming — and the two are different algebras over
different dtypes, not two implementations of one thing. Never let a
capability check for one decide the other (CLAUDE.md §8.3).
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _C, _require_tensor

from . import packed  # noqa: F401  -- the other algebra, deliberately its own namespace

__all__ = [
    "bind", "bundle", "codebook", "cyclic_shift", "discretize", "from_flat",
    "packed", "random_continuous", "random_discrete", "similarity",
    "similarity_batch", "to_flat", "unbind", "zero",
]


def _blocks(a: Tensor, what: str):
    if a.dim() < 2:
        raise ValueError(f"{what}: a block code is (..., k, l), got shape {tuple(a.shape)}")
    return a.shape[-2], a.shape[-1]


def bind(a, b) -> Tensor:
    """Per-block circular convolution — association.

    ``a`` and ``b`` are ``(..., k, l)``; one may be a single code, which is
    broadcast over the other's batch. For one-hot codes the result is an
    exact one-hot, so ``unbind`` recovers either factor exactly.
    """
    return Tensor._wrap(_C.blockcode_bind(_require_tensor("a", a)._t, _require_tensor("b", b)._t, False))


def unbind(composite, key) -> Tensor:
    """Per-block circular correlation: ``unbind(bind(x, key), key) == x``."""
    return Tensor._wrap(
        _C.blockcode_bind(_require_tensor("composite", composite)._t, _require_tensor("key", key)._t, True)
    )


def bundle(vectors, normalize: _builtins.bool = True) -> Tensor:
    """Superposition — the sum of the codes, each block renormalized to
    sum to 1.

    ``vectors`` is a sequence of ``(..., k, l)`` codes, or one
    ``(n, ..., k, l)`` tensor to sum over its first dimension. The result is
    continuous: bundling one-hot codes gives a distribution, which is the
    point of bundling. A block that came out entirely zero has no
    distribution to normalize to, so it is left alone rather than turned
    into NaN.
    """
    if isinstance(vectors, Tensor):
        if vectors.dim() < 3:
            raise ValueError(f"bundle: a stacked bundle is (n, ..., k, l), got {tuple(vectors.shape)}")
        total = vectors.sum(0)
    else:
        items = [_require_tensor("vectors", v) for v in vectors]
        if not items:
            raise ValueError("cannot bundle an empty sequence")
        total = items[0]
        for item in items[1:]:
            total = total + item
    if not normalize:
        return total
    sums = total.sum(-1, keepdim=True)
    return total / _g.where(sums == 0.0, _g.full(tuple(sums.shape), 1.0), sums)


def similarity(a, b) -> _builtins.float:
    """IBM NVSA similarity, ``sum(a * b) / k``.

    For one-hot codes this is the fraction of blocks whose hot positions
    agree: 1.0 for identical codes, about ``1 / l`` for unrelated ones.
    """
    a, b = _require_tensor("a", a), _require_tensor("b", b)
    k, _ = _blocks(a, "similarity")
    return _builtins.float((a * b).sum().item()) / k


def similarity_batch(query, codebook) -> Tensor:
    """``similarity`` of one ``(k, l)`` query against every entry of an
    ``(n, k, l)`` codebook, as an ``(n,)`` tensor. One matmul."""
    query, codebook = _require_tensor("query", query), _require_tensor("codebook", codebook)
    k, l = _blocks(query, "similarity_batch")
    if codebook.dim() != 3 or tuple(codebook.shape[-2:]) != (k, l):
        raise ValueError(f"similarity_batch: codebook must be (n, {k}, {l}), got {tuple(codebook.shape)}")
    return (codebook.reshape(codebook.shape[0], k * l) @ query.reshape(k * l)) / k


def random_discrete(k: _builtins.int, l: _builtins.int, *, generator=None, n=None) -> Tensor:
    """A random discrete code: one hot position per block, uniform.

    ``n`` makes it ``(n, k, l)`` rather than ``(k, l)``.
    """
    shape = (k,) if n is None else (_builtins.int(n), k)
    hot = _g.randint(0, l, shape, generator=generator)
    return _g.zeros(*shape, l).scatter(-1, hot.unsqueeze(-1), 1.0)


def random_continuous(k: _builtins.int, l: _builtins.int, *, generator=None, n=None) -> Tensor:
    """A random continuous code: each block a distribution over positions."""
    shape = (k, l) if n is None else (_builtins.int(n), k, l)
    v = _g.rand(*shape, generator=generator)
    return v / v.sum(-1, keepdim=True)


def codebook(n: _builtins.int, k: _builtins.int, l: _builtins.int, *, generator=None,
             orthogonal: _builtins.bool = False) -> Tensor:
    """``n`` discrete codes, ``(n, k, l)``.

    ``orthogonal=True`` builds them by successive binding
    (``cb[i] = bind(cb[i - 1], cb[1])``), an algebraically structured set for
    **resonator factorization**. For role/filler binding keep the default:
    the structured set puts systematic unbind crosstalk on a neighbour,
    where independent atoms recover cleanly.
    """
    entries = random_discrete(k, l, generator=generator, n=n)
    if not orthogonal or n < 3:
        return entries
    rows = [entries[0], entries[1]]
    for _ in range(2, n):
        rows.append(bind(rows[-1], entries[1]))
    return _g.stack(rows)


def zero(k: _builtins.int, l: _builtins.int) -> Tensor:
    """The identity: position 0 hot in every block, so ``bind(x, zero) == x``."""
    hot = _g.zeros(k, 1, dtype=_g.int64)
    return _g.zeros(k, l).scatter(-1, hot, 1.0)


def cyclic_shift(a, n: _builtins.int) -> Tensor:
    """Rotate the code by ``n`` **blocks** — which block sits at which
    position. VSA's protection operator: a sequence position gets its own
    shift, so the same filler at two positions is two different codes.

    v1's ``BlockCodeOps.cyclic_shift``, which is ``np.roll(a, n, axis=-2)``.
    """
    a = _require_tensor("a", a)
    k, _ = _blocks(a, "cyclic_shift")
    n = _builtins.int(n) % k
    if n == 0:
        return a.clone()
    return _g.cat([a[..., k - n:, :], a[..., : k - n, :]], dim=-2)


def discretize(a) -> Tensor:
    """The nearest discrete code: each block's largest position, one-hot."""
    a = _require_tensor("a", a)
    return _g.zeros(*a.shape).scatter(-1, a.argmax(-1, keepdim=True), 1.0)


def to_flat(a) -> Tensor:
    """``(..., k, l)`` seen as ``(..., k * l)``."""
    a = _require_tensor("a", a)
    k, l = _blocks(a, "to_flat")
    return a.reshape(*a.shape[:-2], k * l)


def from_flat(v, k: _builtins.int) -> Tensor:
    """``(..., k * l)`` seen as ``(..., k, l)``."""
    v = _require_tensor("v", v)
    width = v.shape[-1]
    if width % k:
        raise ValueError(f"from_flat: {width} is not divisible by k = {k}")
    return v.reshape(*v.shape[:-1], k, width // k)
