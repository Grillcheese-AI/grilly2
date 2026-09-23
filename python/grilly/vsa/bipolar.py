"""Bipolar hypervectors: dense ±1, float32.

The third VSA algebra here, and the one with no kernels of its own —
deliberately, because it turned out not to need any:

- **bind** is element-wise multiplication, which makes it its own inverse
  (``x * x == 1``), so ``unbind`` is the same operation;
- **bundle** is a sum, then ``sign`` — a majority vote;
- **similarity** is ``dot / dim``, which for ±1 vectors runs −1 to +1, with
  **0 meaning unrelated**. Not 0.5, which is the packed algebra's baseline,
  and not 1/l, which is a block code's.

**Bipolar and packed are the same vectors in two representations**, and
``pack``/``unpack`` move between them exactly. The relation that holds
whatever you do:

    bipolar.similarity(a, b) == 2 * packed.similarity(pack(a), pack(b)) - 1

— so ranking a codebook gives the *same order* either way. That is the
relation that matters, because packing is for storage and search: 32x
less memory, and a popcount instead of a dot product.

**Binding is where they part.** ``pack`` writes ``+1 → bit 1``, which is
what v1's ``vsa_bitpack`` wrote and therefore what a stored file holds.
Under that convention an element-wise product is XNOR of the bits, while
``packed.bind`` is XOR, so the two give complementary answers: do not mix
them mid-expression. Bind in one representation, then pack.

Dense is also what you want while values are still being combined: an
un-normalized bundle keeps the *counts*, which a bit string cannot hold.

This module has no kernels. Multiplication, a sum, a sign and a dot
product are ops the framework already has, and that is the whole
implementation.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _require_tensor
from . import packed as _packed

__all__ = [
    "bind", "bundle", "from_key", "pack", "random", "similarity",
    "similarity_batch", "unbind", "unpack",
]


def _require_bipolar(x, what: str) -> Tensor:
    x = _require_tensor(what, x)
    if x.dtype != _g.float32:
        raise ValueError(f"{what}: bipolar hypervectors are float32 ±1, got {x.dtype}")
    return x


def random(dim: _builtins.int, *, generator=None, n=None) -> Tensor:
    """A random bipolar hypervector: each element +1 or −1, even odds."""
    shape = (dim,) if n is None else (_builtins.int(n), dim)
    return _g.where(_g.rand(*shape, generator=generator) < 0.5, -1.0, 1.0)


def bind(a, b) -> Tensor:
    """Element-wise product — association, and its own inverse."""
    return _require_bipolar(a, "a") * _require_bipolar(b, "b")


def unbind(composite, key) -> Tensor:
    """The same product: ``unbind(bind(x, key), key) == x``, exactly."""
    return bind(composite, key)


def bundle(vectors, normalize: _builtins.bool = True) -> Tensor:
    """Sum the vectors; with ``normalize``, take the sign — a majority vote.

    ``normalize=False`` keeps the sum, which holds how *many* vectors voted
    each way. That is real information and the packed algebra cannot carry
    it, so a bundle that will be added to again should stay un-normalized
    until the end.

    A tie — only an even count can produce one — goes to +1, which is v1's
    rule (it took the sign of ``sum + 1e-8``).
    """
    if isinstance(vectors, Tensor):
        total = _require_bipolar(vectors, "vectors").sum(0)
    else:
        items = [_require_bipolar(v, "vectors") for v in vectors]
        if not items:
            raise ValueError("cannot bundle an empty sequence")
        total = items[0]
        for item in items[1:]:
            total = total + item
    if not normalize:
        return total
    return _g.where(total < 0.0, -1.0, 1.0)      # a tie (0) goes to +1, as v1's did


def similarity(a, b) -> _builtins.float:
    """``dot(a, b) / dim``: +1 identical, **0 unrelated**, −1 opposite."""
    a, b = _require_bipolar(a, "a"), _require_bipolar(b, "b")
    return _builtins.float((a * b).sum().item()) / a.shape[-1]


def similarity_batch(query, codebook) -> Tensor:
    """``similarity`` against every row of a ``(n, dim)`` codebook."""
    query, codebook = _require_bipolar(query, "query"), _require_bipolar(codebook, "codebook")
    if codebook.dim() != 2 or codebook.shape[1] != query.shape[-1]:
        raise ValueError(
            f"similarity_batch: codebook must be (n, {query.shape[-1]}), got {tuple(codebook.shape)}"
        )
    return (codebook @ query.reshape(query.shape[-1])) / query.shape[-1]


def from_key(key: str, dim: _builtins.int, *, domain: str = "grilly.bipolar") -> Tensor:
    """The bipolar vector for ``key``, from a hash stream rather than an RNG.

    The same key gives the same vector in every process and every version —
    which is what a symbol's hypervector has to do if it is stored
    (``grilly.vsa.hashing``).
    """
    from .hashing import bipolar_from_key

    return _g.from_numpy(bipolar_from_key(key, _builtins.int(dim), domain=domain))


def pack(x) -> Tensor:
    """The same hypervectors as packed bits: ``+1 → 1``, ``−1 → 0``.

    v1's ``vsa_bitpack`` convention, so a packed store written by v1 reads
    back correctly. Ranking survives it exactly (see the module docstring);
    binding does not, because a product is XNOR under this convention and
    ``packed.bind`` is XOR.
    """
    x = _require_bipolar(x, "x")
    return _packed.from_bits((x.numpy() > 0).astype("uint8"))


def unpack(words, dim=None) -> Tensor:
    """The inverse of ``pack``: packed bits back to dense ±1."""
    bits = _packed.to_bits(words, dim)
    return _g.from_numpy((2.0 * bits - 1.0).astype("float32"))
