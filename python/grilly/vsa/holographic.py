"""Holographic reduced representations: dense float32, circular convolution.

Plate's HRR. A hypervector is a dense real vector, binding is circular
convolution over the whole thing, and unbinding is circular correlation —
which recovers a factor *approximately*, unlike the bipolar and block-code
algebras where it is exact. What HRR buys for that is that a bound vector
keeps more of what went into it.

    from grilly.vsa import holographic as hrr

    a, b = hrr.random(1024), hrr.random(1024)
    bound = hrr.bind(a, b)
    assert hrr.similarity(hrr.unbind(bound, b), a) > 0.9    # approximate

**No kernel of its own.** HRR binding is exactly what ``grilly.vsa`` does
with one block: a block code of ``k = 1, l = dim`` *is* a dense vector
under circular convolution. So this module reshapes and calls the same
kernel.

**The cost, stated plainly.** That kernel does the convolution directly,
which is O(d²) where an FFT pair is O(d log d). Measured on an RX 6750 XT
against numpy's FFT on the same vectors, as a ratio (above 1.0 this is
faster):

| dim | batch 1 | batch 16 | batch 64 |
|---:|---:|---:|---:|
| 1024 | 0.85 | 3.80 | 5.83 |
| 4096 | 0.40 | 1.36 | 3.01 |
| 10240 | 0.89 | 0.82 | 1.69 |

So it is a clear win below 4096, roughly a wash at 10240, and **the wrong
algorithm above that** — the quadratic term takes over and a GPU FFT would
be the answer. It is not written yet because nothing here needs it; if you
find yourself binding vectors wider than ~16k, that is the reason it got
slow, and it is a missing kernel rather than a property of the approach.

Accuracy against a float64 FFT is 1e-6 to 5e-6 relative, a float32 sum of
d products.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _require_tensor
from . import bind as _block_bind, unbind as _block_unbind

__all__ = ["bind", "bundle", "random", "similarity", "similarity_batch", "unbind"]


def _as_blocks(x, what: str) -> Tensor:
    """A dense ``(..., dim)`` vector as the one-block code it is."""
    x = _require_tensor(what, x)
    if x.dim() < 1 or x.shape[-1] == 0:
        raise ValueError(f"{what}: an HRR vector is (..., dim), got {tuple(x.shape)}")
    return x.reshape(*x.shape[:-1], 1, x.shape[-1])


def _flat(t: Tensor) -> Tensor:
    return t.reshape(*t.shape[:-2], t.shape[-1])


def bind(a, b) -> Tensor:
    """Circular convolution — association."""
    return _flat(_block_bind(_as_blocks(a, "a"), _as_blocks(b, "b")))


def unbind(composite, key) -> Tensor:
    """Circular correlation — an *approximate* inverse of ``bind``.

    The recovered vector is the original plus crosstalk from everything else
    the composite holds, so read it by ranking a codebook against it
    (``similarity_batch``) rather than by comparing it to one vector.
    """
    return _flat(_block_unbind(_as_blocks(composite, "composite"), _as_blocks(key, "key")))


def random(dim: _builtins.int, *, generator=None, n=None) -> Tensor:
    """A random unit vector — HRR's atoms are normal and normalized, which
    is what keeps a convolution from growing or shrinking the magnitude."""
    shape = (dim,) if n is None else (_builtins.int(n), dim)
    v = _g.randn(*shape, generator=generator)
    return v / v.pow(2).sum(-1, keepdim=True).sqrt()


def bundle(vectors, normalize: _builtins.bool = True) -> Tensor:
    """Sum the vectors, normalized back to unit length — superposition."""
    if isinstance(vectors, Tensor):
        total = _require_tensor("vectors", vectors).sum(0)
    else:
        items = [_require_tensor("vectors", v) for v in vectors]
        if not items:
            raise ValueError("cannot bundle an empty sequence")
        total = items[0]
        for item in items[1:]:
            total = total + item
    if not normalize:
        return total
    norm = total.pow(2).sum(-1, keepdim=True).sqrt()
    return total / _g.where(norm == 0.0, _g.full(tuple(norm.shape), 1.0), norm)


def similarity(a, b) -> _builtins.float:
    """Cosine similarity: +1 identical, **0 unrelated**, −1 opposite.

    A zero-length vector is 0.0 rather than a NaN, which is v1's rule.
    """
    a, b = _require_tensor("a", a), _require_tensor("b", b)
    na = _builtins.float(a.pow(2).sum().item()) ** 0.5
    nb = _builtins.float(b.pow(2).sum().item()) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _builtins.float((a * b).sum().item()) / (na * nb)


def similarity_batch(query, codebook) -> Tensor:
    """Cosine similarity against every row of a ``(n, dim)`` codebook."""
    query, codebook = _require_tensor("query", query), _require_tensor("codebook", codebook)
    dim = query.shape[-1]
    if codebook.dim() != 2 or codebook.shape[1] != dim:
        raise ValueError(f"similarity_batch: codebook must be (n, {dim}), got {tuple(codebook.shape)}")
    nq = _builtins.float(query.pow(2).sum().item()) ** 0.5
    if nq == 0.0:
        return _g.zeros(codebook.shape[0])
    norms = codebook.pow(2).sum(-1).sqrt()
    norms = _g.where(norms == 0.0, _g.full(tuple(norms.shape), 1.0), norms)
    return (codebook @ query.reshape(dim)) / (norms * nq)
