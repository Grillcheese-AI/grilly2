"""Deprecated: v1's ``BlockCodeOps``. Use ``grilly.vsa``.

Every method keeps v1's signature — numpy in, numpy out, ``(..., k, l)``
float32 — and runs on the GPU underneath, which v1's version did not: its
``BlockCodeOps`` was numpy through and through, and the Vulkan tier its
callers detected was a separate, one-hot-only bridge that the algebra never
actually went through (CLAUDE.md §8.3).

So a caller that does nothing but keep importing this gets the GPU. A
caller that moves to ``grilly.vsa`` also stops paying for a numpy round
trip on every call, which at batch 512 is most of the time.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly import vsa
from grilly._v1 import moved

__all__ = ["BlockCodeOps", "EPS"]

EPS = 1e-20


def _in(a) -> "grilly.Tensor":
    return grilly.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _out(t) -> np.ndarray:
    return t.numpy()


class BlockCodeOps:
    """v1's block-code operations, on ``grilly.vsa``.

    Arrays are ``(..., k, l)`` float32: k blocks of l, one-hot per block for
    a discrete code, a distribution per block for a continuous one.
    """

    @staticmethod
    def random_discrete(k: int, l: int, seed: int | None = None) -> np.ndarray:
        moved("BlockCodeOps.random_discrete", "grilly.vsa.random_discrete")
        return _out(vsa.random_discrete(k, l, generator=_generator(seed)))

    @staticmethod
    def random_continuous(k: int, l: int, seed: int | None = None) -> np.ndarray:
        moved("BlockCodeOps.random_continuous", "grilly.vsa.random_continuous")
        return _out(vsa.random_continuous(k, l, generator=_generator(seed)))

    @staticmethod
    def zero_vector(k: int, l: int) -> np.ndarray:
        moved("BlockCodeOps.zero_vector", "grilly.vsa.zero")
        return _out(vsa.zero(k, l))

    @staticmethod
    def codebook_discrete(k: int, l: int, n: int, seed: int | None = None,
                          orthogonal: bool = True) -> np.ndarray:
        moved("BlockCodeOps.codebook_discrete", "grilly.vsa.codebook")
        return _out(vsa.codebook(n, k, l, generator=_generator(seed), orthogonal=orthogonal))

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        moved("BlockCodeOps.bind", "grilly.vsa.bind")
        return _out(vsa.bind(_in(a), _in(b)))

    @staticmethod
    def unbind(composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        moved("BlockCodeOps.unbind", "grilly.vsa.unbind")
        return _out(vsa.unbind(_in(composite), _in(known)))

    @staticmethod
    def bind3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        moved("BlockCodeOps.bind3", "grilly.vsa.bind")
        return _out(vsa.bind(vsa.bind(_in(a), _in(b)), _in(c)))

    @staticmethod
    def unbind3(composite: np.ndarray, known1: np.ndarray, known2: np.ndarray) -> np.ndarray:
        moved("BlockCodeOps.unbind3", "grilly.vsa.unbind")
        return _out(vsa.unbind(vsa.unbind(_in(composite), _in(known1)), _in(known2)))

    @staticmethod
    def bundle(vectors: list, normalize: bool = True) -> np.ndarray:
        moved("BlockCodeOps.bundle", "grilly.vsa.bundle")
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")
        return _out(vsa.bundle([_in(v) for v in vectors], normalize=normalize))

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        moved("BlockCodeOps.similarity", "grilly.vsa.similarity")
        return vsa.similarity(_in(a), _in(b))

    @staticmethod
    def similarity_batch(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        moved("BlockCodeOps.similarity_batch", "grilly.vsa.similarity_batch")
        return _out(vsa.similarity_batch(_in(query), _in(codebook)))

    @staticmethod
    def similarity_zero(a: np.ndarray) -> float:
        moved("BlockCodeOps.similarity_zero", "grilly.vsa.similarity against grilly.vsa.zero")
        k, l = a.shape[-2], a.shape[-1]
        return vsa.similarity(_in(a), vsa.zero(k, l))

    @staticmethod
    def cyclic_shift(a: np.ndarray, n: int) -> np.ndarray:
        moved("BlockCodeOps.cyclic_shift", "grilly.vsa.cyclic_shift")
        return _out(vsa.cyclic_shift(_in(a), n))

    @staticmethod
    def discretize(a: np.ndarray) -> np.ndarray:
        moved("BlockCodeOps.discretize", "grilly.vsa.discretize")
        return _out(vsa.discretize(_in(a)))

    @staticmethod
    def from_flat(v: np.ndarray, k: int) -> np.ndarray:
        moved("BlockCodeOps.from_flat", "grilly.vsa.from_flat")
        return _out(vsa.from_flat(_in(v), k))

    @staticmethod
    def to_flat(a: np.ndarray) -> np.ndarray:
        moved("BlockCodeOps.to_flat", "grilly.vsa.to_flat")
        return _out(vsa.to_flat(_in(a)))


def _generator(seed):
    """v1 took a seed per call; grilly takes a Generator. ``None`` means the
    default generator, which is what v1's ``seed=None`` meant too."""
    return None if seed is None else grilly.Generator().manual_seed(int(seed))
