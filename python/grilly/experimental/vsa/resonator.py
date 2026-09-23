"""Deprecated: v1's ``ResonatorNetwork``. Use ``grilly.vsa.resonator``.

v1's took numpy codebooks and returned ``(estimates, indices, iterations)``;
the new one returns ``(indices, estimates, iterations)``, because the
indices are what a caller almost always wants. This alias keeps v1's order
and its numpy arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import grilly
from grilly._v1 import moved
from grilly.vsa import bipolar as _bipolar
from grilly.vsa import resonator as _resonator

__all__ = ["ResonatorNetwork"]


class ResonatorNetwork:
    """v1's resonator network, on ``grilly.vsa.resonator``."""

    def __init__(self, codebooks: dict, max_iterations: int = 20,
                 convergence_threshold: float = 0.95, vsa_backend: Any | None = None) -> None:
        moved("grilly.experimental.vsa.resonator.ResonatorNetwork",
              "grilly.vsa.resonator.ResonatorNetwork")
        if not codebooks:
            raise ValueError("Codebooks cannot be empty")
        dims = {np.asarray(cb).shape[-1] for cb in codebooks.values()}
        assert len(dims) == 1, "All codebooks must have same dimension"
        self.dim = dims.pop()
        self.codebooks = codebooks
        self.factor_names = list(codebooks)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        # v1 took a backend object to route similarity through; grilly2 has
        # one path to the GPU, so there is nothing to choose (CLAUDE.md §5)
        self.vsa_backend = vsa_backend
        self._net = _resonator.ResonatorNetwork(
            {k: grilly.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
             for k, v in codebooks.items()},
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            algebra=_bipolar,
        )

    def factorize(self, composite, init_estimates=None):
        """``(estimates, indices, iterations)`` — v1's order."""
        c = grilly.from_numpy(np.ascontiguousarray(composite, dtype=np.float32))
        init = None if init_estimates is None else {
            k: grilly.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            for k, v in init_estimates.items()
        }
        indices, estimates, iterations = self._net.factorize(c, init)
        return {k: v.numpy() for k, v in estimates.items()}, indices, iterations

    def factorize_partial(self, composite, known_factors: dict):
        """The remaining factor when the others are known. No search."""
        c = grilly.from_numpy(np.ascontiguousarray(composite, dtype=np.float32))
        known = {k: grilly.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
                 for k, v in known_factors.items()}
        return self._net.factorize_known(c, known).numpy()
