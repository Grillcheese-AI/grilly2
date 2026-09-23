"""Deprecated: v1's ``BinaryOps`` and ``HolographicOps``.

Use ``grilly.vsa.bipolar`` and ``grilly.vsa.holographic``. Both keep v1's
signatures — numpy in, numpy out, float32 — and run on the GPU, which v1's
did not: these two classes were numpy throughout.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly._v1 import moved
from grilly.vsa import bipolar as _bipolar
from grilly.vsa import holographic as _hrr

__all__ = ["BinaryOps", "HolographicOps"]


def _in(a):
    return grilly.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _generator(seed):
    return None if seed is None else grilly.Generator().manual_seed(int(seed))


class BinaryOps:
    """Bipolar (+1/−1) vectors: bind is a product, bundle a majority vote."""

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        moved("BinaryOps.bind", "grilly.vsa.bipolar.bind")
        return _bipolar.bind(_in(a), _in(b)).numpy()

    @staticmethod
    def bind_batch(a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        moved("BinaryOps.bind_batch", "grilly.vsa.bipolar.bind")
        return _bipolar.bind(_in(a_batch), _in(b_batch)).numpy()

    @staticmethod
    def unbind(composite: np.ndarray, known: np.ndarray) -> np.ndarray:
        moved("BinaryOps.unbind", "grilly.vsa.bipolar.unbind")
        return _bipolar.unbind(_in(composite), _in(known)).numpy()

    @staticmethod
    def bundle(vectors: list, normalize: bool = True) -> np.ndarray:
        moved("BinaryOps.bundle", "grilly.vsa.bipolar.bundle")
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")
        return _bipolar.bundle([_in(v) for v in vectors], normalize=normalize).numpy()

    @staticmethod
    def bundle_batch(vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
        moved("BinaryOps.bundle_batch", "grilly.vsa.bipolar.bundle")
        return _bipolar.bundle(_in(vectors), normalize=normalize).numpy()

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        moved("BinaryOps.similarity", "grilly.vsa.bipolar.similarity")
        return _bipolar.similarity(_in(a), _in(b))

    @staticmethod
    def similarity_batch(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        moved("BinaryOps.similarity_batch", "grilly.vsa.bipolar.similarity_batch")
        return _bipolar.similarity_batch(_in(query), _in(codebook)).numpy()

    @staticmethod
    def random_bipolar(dim: int, seed: int | None = None) -> np.ndarray:
        moved("BinaryOps.random_bipolar", "grilly.vsa.bipolar.random")
        return _bipolar.random(dim, generator=_generator(seed)).numpy()

    @staticmethod
    def hash_to_bipolar(s: str, dim: int) -> np.ndarray:
        moved("BinaryOps.hash_to_bipolar", "grilly.vsa.bipolar.from_key")
        return _bipolar.from_key(s, dim, domain="grilly.vsa.binaryops").numpy()


class HolographicOps:
    """Dense real vectors under circular convolution (Plate's HRR).

    Unbinding is approximate here, unlike ``BinaryOps``: read the result by
    ranking a codebook against it, not by comparing it to one vector.
    """

    @staticmethod
    def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        moved("HolographicOps.convolve", "grilly.vsa.holographic.bind")
        return _hrr.bind(_in(a), _in(b)).numpy()

    @staticmethod
    def convolve_batch(a_batch: np.ndarray, b_batch: np.ndarray) -> np.ndarray:
        moved("HolographicOps.convolve_batch", "grilly.vsa.holographic.bind")
        return _hrr.bind(_in(a_batch), _in(b_batch)).numpy()

    @staticmethod
    def correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        moved("HolographicOps.correlate", "grilly.vsa.holographic.unbind")
        return _hrr.unbind(_in(a), _in(b)).numpy()

    @staticmethod
    def bundle(vectors: list, normalize: bool = True) -> np.ndarray:
        moved("HolographicOps.bundle", "grilly.vsa.holographic.bundle")
        if not vectors:
            raise ValueError("Cannot bundle empty list of vectors")
        return _hrr.bundle([_in(v) for v in vectors], normalize=normalize).numpy()

    @staticmethod
    def bundle_batch(vectors: np.ndarray, normalize: bool = True) -> np.ndarray:
        moved("HolographicOps.bundle_batch", "grilly.vsa.holographic.bundle")
        return _hrr.bundle(_in(vectors), normalize=normalize).numpy()

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        moved("HolographicOps.similarity", "grilly.vsa.holographic.similarity")
        return _hrr.similarity(_in(a), _in(b))

    @staticmethod
    def similarity_batch(query: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        moved("HolographicOps.similarity_batch", "grilly.vsa.holographic.similarity_batch")
        return _hrr.similarity_batch(_in(query), _in(codebook)).numpy()

    @staticmethod
    def random_vector(dim: int, seed: int | None = None) -> np.ndarray:
        moved("HolographicOps.random_vector", "grilly.vsa.holographic.random")
        return _hrr.random(dim, generator=_generator(seed)).numpy()
