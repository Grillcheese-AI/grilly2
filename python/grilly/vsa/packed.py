"""Packed binary hypervectors: ``dim`` bits held 32 to an int32 word.

The other VSA algebra, and **not** a variant of the block codes in
``grilly.vsa`` (CLAUDE.md §8.3). Here a hypervector is a bit string, 32x
smaller than one float per dimension:

    from grilly.vsa import packed

    role = packed.random(10240, generator=g)      # (320,) int32 words
    filler = packed.random(10240, generator=g)
    bound = packed.bind(role, filler)
    assert packed.similarity(bound, role) < 0.55  # binding hides both
    assert (packed.bind(bound, role) == filler).all()   # XOR is its own inverse

- **bind** is XOR, so it is its own inverse: ``bind(bind(a, b), b) == a``;
- **bundle** is a bit-wise majority vote — superposition;
- **similarity** is Hamming, ``1 - distance / dim``. 1.0 identical, **0.5
  for two unrelated vectors**, 0.0 for a complement. 0.5 is the number to
  compare against, not 0;
- **permute** rotates the bits, giving a sequence position its own role.

Every operation is integer, so every result is exact. There is no tolerance
here, and a test that wants one has found a bug.
"""

from __future__ import annotations

import builtins as _builtins

import numpy as _np

import grilly as _g
from .. import Tensor, _C, _require_tensor

__all__ = [
    "bind", "bundle", "from_bits", "permute", "random", "similarity",
    "similarity_batch", "to_bits", "topk", "unbind", "words_for",
]


def words_for(dim: _builtins.int) -> _builtins.int:
    """Words a ``dim``-bit hypervector needs. ``dim`` must be a multiple of
    32: a partly used last word would make two vectors of the same dim
    compare differently depending on what the padding held."""
    dim = _builtins.int(dim)
    if dim <= 0 or dim % 32:
        raise ValueError(f"a packed hypervector's dim must be a positive multiple of 32, got {dim}")
    return dim // 32


def _require_words(x, what: str) -> Tensor:
    x = _require_tensor(what, x)
    if x.dtype != _g.int32:
        raise ValueError(f"{what}: packed hypervectors are int32 words, got {x.dtype}")
    return x


def random(dim: _builtins.int, *, generator=None, n=None) -> Tensor:
    """A random packed hypervector: every bit an independent coin flip.

    ``n`` makes it ``(n, words)`` rather than ``(words,)``.
    """
    words = words_for(dim)
    shape = (words,) if n is None else (_builtins.int(n), words)
    # randint over the full int32 range gives 32 independent bits a word
    return _g.randint(-(2 ** 31), 2 ** 31, shape, generator=generator, dtype=_g.int32)


def bind(a, b) -> Tensor:
    """XOR — association. Its own inverse, so ``unbind`` is this again."""
    return _g.bitwise_xor(_require_words(a, "a"), _require_words(b, "b"))


def unbind(composite, key) -> Tensor:
    """XOR again: ``unbind(bind(x, key), key) == x``, exactly."""
    return bind(composite, key)


def bundle(vectors) -> Tensor:
    """Bit-wise majority vote over ``(n, words)`` — superposition.

    A tie, which only an even ``n`` can produce, goes to 0 (v1's rule). The
    result is similar to every vector bundled into it and unrelated to
    everything else, which is what makes it a set.
    """
    if not isinstance(vectors, Tensor):
        vectors = _g.stack([_require_words(v, "vectors") for v in vectors])
    return Tensor._wrap(_C.hdc_bundle(_require_words(vectors, "vectors")._t))


def hamming(query, codebook) -> Tensor:
    """Hamming distance from ``query`` (words,) to every row of ``codebook``
    (n, words), as int32 — the count of differing bits."""
    return Tensor._wrap(
        _C.hdc_hamming(_require_words(query, "query")._t, _require_words(codebook, "codebook")._t)
    )


def similarity(a, b, dim=None) -> _builtins.float:
    """``1 - hamming(a, b) / dim``. 1.0 identical, **0.5 unrelated**, 0.0 a
    complement. ``dim`` defaults to all the bits of the words given."""
    a, b = _require_words(a, "a"), _require_words(b, "b")
    dim = _builtins.int(dim) if dim is not None else a.numel() * 32
    d = hamming(a, b.reshape(1, b.numel()))
    return 1.0 - _builtins.float(d.item()) / dim


def similarity_batch(query, codebook, dim=None) -> Tensor:
    """``similarity`` of one query against every codebook row, as floats."""
    query = _require_words(query, "query")
    dim = _builtins.int(dim) if dim is not None else query.numel() * 32
    return 1.0 - hamming(query, codebook).to(_g.float32) / dim


def topk(query, codebook, k: _builtins.int = 10, dim=None):
    """The ``k`` nearest codebook rows by Hamming distance.

    Returns ``(indices, distances, similarities)``, nearest first — v1's
    ``hamming_topk``, whose dict had those three keys.
    """
    query = _require_words(query, "query")
    dim = _builtins.int(dim) if dim is not None else query.numel() * 32
    distances = hamming(query, _require_words(codebook, "codebook"))
    k = _builtins.min(_builtins.int(k), distances.numel())
    nearest, indices = _g.topk(distances, k, largest=False, sorted=True)
    return indices, nearest, 1.0 - nearest.to(_g.float32) / dim


def permute(x, shift: _builtins.int, dim=None) -> Tensor:
    """Rotate the bits cyclically: ``out bit i = in bit (i - shift) mod dim``.

    VSA's protection operator — the same filler at two sequence positions
    becomes two unrelated hypervectors, reversibly.
    """
    x = _require_words(x, "x")
    dim = _builtins.int(dim) if dim is not None else x.shape[-1] * 32
    return Tensor._wrap(_C.hdc_permute(x._t, dim, _builtins.int(shift)))


def from_bits(bits) -> Tensor:
    """Pack a 0/1 (or boolean) array of ``dim`` bits, last axis first, into
    int32 words — numpy's ``packbits(..., bitorder="little")`` over
    little-endian words, which is the layout v1 used."""
    bits = _np.ascontiguousarray(bits).astype(_np.uint8)
    words_for(bits.shape[-1])
    packed = _np.packbits(bits, axis=-1, bitorder="little")
    return _g.from_numpy(packed.view(_np.int32).copy())


def to_bits(x, dim=None):
    """The inverse of ``from_bits``: a uint8 array of 0/1, ``dim`` wide."""
    x = _require_words(x, "x")
    dim = _builtins.int(dim) if dim is not None else x.shape[-1] * 32
    words = x.numpy().astype(_np.int32)
    return _np.unpackbits(words.view(_np.uint8), axis=-1, bitorder="little")[..., :dim]
