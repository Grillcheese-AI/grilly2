"""Deprecated: v1's ``grilly.backend.cvl``. Use ``grilly.cvl``.

Same classes and the same maths, on tensors rather than numpy — the
feature map is an ``(in_dim, n_features)`` matmul, which is the only part
that ever cost anything. These take and return numpy, as v1's did.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly import cvl as _cvl
from grilly._v1 import moved

__all__ = ["CVLValue", "RandomFourierFeatures"]


def _in(a):
    return grilly.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


class RandomFourierFeatures:
    """``F(x) = sqrt(2e/d) * cos(Wx + b)`` (Rahimi and Recht, 2007)."""

    def __init__(self, in_dim: int, n_features: int = 4096, seed: int = 0) -> None:
        moved("grilly.backend.cvl.RandomFourierFeatures", "grilly.cvl.RandomFourierFeatures")
        self._f = _cvl.RandomFourierFeatures(in_dim, n_features, seed)
        self.n_features = self._f.n_features

    @property
    def W(self) -> np.ndarray:
        return self._f.W.numpy()

    @property
    def b(self) -> np.ndarray:
        return self._f.b.numpy()

    def __call__(self, x) -> np.ndarray:
        return self._f(_in(x)).numpy()


class CVLValue:
    """TD-free contrastive value: ``Q = F(phi) . xi / (1 - gamma)``."""

    def __init__(self, gamma: float = 0.95, beta: float = 0.99) -> None:
        moved("grilly.backend.cvl.CVLValue", "grilly.cvl.CVLValue")
        self._v = _cvl.CVLValue(gamma, beta)
        self.gamma, self.beta = self._v.gamma, self._v.beta

    @property
    def xi(self):
        return None if self._v.xi is None else self._v.xi.numpy()

    def observe(self, future_feat, reward: float) -> None:
        self._v.observe(_in(future_feat), reward)

    def value(self, phi_feat) -> float:
        return self._v.value(_in(phi_feat))
