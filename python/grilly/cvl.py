"""Random Fourier features, and the TD-free value readout built on them.

Two things, one of which is much more general than the other.

**RandomFourierFeatures** (Rahimi and Recht, 2007) linearizes an RBF
kernel: ``F(x) = sqrt(2e/d) * cos(Wx + b)`` with ``W ~ N(0, I)`` and
``b ~ U(0, 2pi)`` gives ``E[F(x) . F(y)] = exp(x . y)`` for unit ``x`` and
``y``. So an exponential kernel becomes a dot product in ``d`` dimensions,
and anything that wanted the kernel can use a matmul instead. Useful well
beyond what follows.

**CVLValue** (Mazoure, Eysenbach, Nachum, Tompson — Contrastive Value
Learning, NeurIPS-W 2022) is what v1 used it for: a value estimate with
**no temporal-difference back-up**. Because the features factor the
exponential, the discounted future-state occupancy collapses into a single
reward-weighted vector ``xi`` kept as a moving average, and the value is
one inner product:

    Q(s, a) = 1 / (1 - gamma) * F(phi(s, a)) . xi

read in O(1), no bootstrapping, no target network. It recovers Q up to a
positive constant, which is enough to rank actions and not enough to
compare against a reward scale.

    from grilly import cvl

    features = cvl.RandomFourierFeatures(in_dim=256, n_features=4096, seed=0)
    value = cvl.CVLValue(gamma=0.95)
    value.observe(features(future_state), reward=1.0)
    q = value.value(features(state_action))

v1's ``grilly.backend.cvl``, on tensors rather than numpy — which is the
whole point, since the feature map is a ``(in_dim, n_features)`` matmul and
that is the only expensive part.
"""

from __future__ import annotations

import builtins as _builtins
import math as _math

import grilly as _g
from . import Tensor, _require_tensor

__all__ = ["CVLValue", "RandomFourierFeatures"]


class RandomFourierFeatures:
    """``F(x) = sqrt(2e/d) * cos(Wx + b)``, a fixed random feature map.

    ``W`` and ``b`` are drawn once and never trained — that is what makes
    the kernel approximation hold. ``seed`` fixes them, so two processes
    that agree on it agree on every feature.
    """

    def __init__(self, in_dim: _builtins.int, n_features: _builtins.int = 4096,
                 seed: _builtins.int = 0) -> None:
        generator = _g.Generator().manual_seed(_builtins.int(seed))
        self.in_dim = _builtins.int(in_dim)
        self.n_features = _builtins.int(n_features)
        self.W = _g.randn(self.in_dim, self.n_features, generator=generator)
        self.b = _g.rand(self.n_features, generator=generator) * (2.0 * _math.pi)
        self._scale = _math.sqrt(2.0 * _math.e / self.n_features)

    def __call__(self, x) -> Tensor:
        """``(..., in_dim)`` to ``(..., n_features)``."""
        x = _require_tensor("x", x)
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"RandomFourierFeatures: expected (..., {self.in_dim}), got {tuple(x.shape)}"
            )
        return self._scale * (x @ self.W + self.b).cos()


class CVLValue:
    """A value read forward from a reward-weighted future-feature average.

    ``observe`` folds ``reward * F(future_state)`` into ``xi`` as an
    exponential moving average with decay ``beta``; ``value`` reads
    ``F(phi) . xi / (1 - gamma)``. Nothing bootstraps off anything, so
    there is no divergence to manage — and no absolute scale either, since
    the result is proportional to Q rather than equal to it.
    """

    def __init__(self, gamma: _builtins.float = 0.95, beta: _builtins.float = 0.99) -> None:
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"CVLValue: gamma must be in [0, 1), got {gamma}")
        self.gamma = _builtins.float(gamma)
        self.beta = _builtins.float(beta)
        self._xi: Tensor | None = None

    @property
    def xi(self) -> Tensor | None:
        """The running future-feature vector, or ``None`` before anything
        has been observed."""
        return self._xi

    def observe(self, future_features, reward: _builtins.float) -> None:
        """Fold one observed future state, weighted by its reward, into ``xi``."""
        f = _require_tensor("future_features", future_features) * _builtins.float(reward)
        if self._xi is None:
            self._xi = _g.zeros(*f.shape)
        self._xi = self.beta * self._xi + (1.0 - self.beta) * f

    def value(self, features) -> _builtins.float:
        """``F(phi) . xi / (1 - gamma)``. Zero before anything is observed,
        which is the right answer for a value nothing has informed."""
        if self._xi is None:
            return 0.0
        f = _require_tensor("features", features)
        return _builtins.float((f * self._xi).sum().item()) / (1.0 - self.gamma)
