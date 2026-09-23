"""Resonator networks: factorizing a bound hypervector.

Given ``composite = bind(a, bind(b, c))`` and a codebook of the possible
values of each factor, recover which ones went in. Unbinding alone cannot
do it — you would need to already know two factors to get the third — so
the network guesses all of them and then, for each in turn, unbinds the
*other* current guesses and snaps the result to the nearest codebook entry.
The guesses reinforce each other until they stop moving.

    from grilly.vsa import bipolar, resonator

    codebooks = {"shape": bipolar.random(1024, n=16),
                 "colour": bipolar.random(1024, n=16)}
    net = resonator.ResonatorNetwork(codebooks)
    composite = bipolar.bind(codebooks["shape"][3], codebooks["colour"][9])
    indices, _, iterations = net.factorize(composite)
    assert indices == {"shape": 3, "colour": 9}

The search is not convex — it has fixed points that are not the answer —
so ``factorize`` restarts from several initial guesses and keeps the one
that reconstructs the composite best.

**How well it works, measured.** This is v1's algorithm, and it is not
very good: accuracy falls off with the size of the codebooks, because a
restart essentially has to guess one factor right before the others can
snap into place. 20 trials a row, bipolar, run against v1's own
implementation on identical problems:

| dim | items per factor | factors | ``restarts=10`` | ``restarts=100`` | v1 |
|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 2 | 11/20 | 18/20 | 11/20 |
| 1024 | 16 | 2 | 4/20 | 15/20 | 5/20 |
| 1024 | 8 | 3 | 5/20 | 13/20 | 4/20 |
| 2048 | 32 | 2 | 2/20 | 8/20 | 2/20 |

The v1 column is there to show this is a faithful port and not a
regression; the differences are one trial, which is noise at 20.

v1 hard-coded ten restarts. Here it is a parameter, defaulting to ten so a
ported caller gets exactly what it had, and raising it is the one lever
that works — the cost is linear in it and it is still not enough at 32
items.

**If you need it to work at that size, this is the wrong algorithm.** The
published resonator dynamics (Frady and Kleyko) carry each estimate as a
*superposition* over its codebook instead of snapping to the nearest entry
every step, which is exactly what lets them converge here. v1 did not
implement that, and neither does this.

Bipolar by default, because its unbinding is exact. Pass ``algebra`` to use
another: anything with ``bind``, ``unbind``, ``similarity`` and
``similarity_batch`` works, so ``grilly.vsa.holographic`` does too, more
slowly and approximately.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor
from . import bipolar as _bipolar

__all__ = ["ResonatorNetwork"]


class ResonatorNetwork:
    """Iterative factorization against one codebook per factor.

    ``codebooks`` maps a factor's name to its ``(n, dim)`` candidates; every
    codebook must be the same width, since they all have to bind together.
    """

    def __init__(self, codebooks, max_iterations: _builtins.int = 20,
                 convergence_threshold: _builtins.float = 0.95, *, algebra=None,
                 restarts: _builtins.int = 10, generator=None) -> None:
        if not codebooks:
            raise ValueError("a resonator network needs at least one codebook")
        self.algebra = algebra if algebra is not None else _bipolar
        self.codebooks = {name: cb for name, cb in codebooks.items()}
        self.factor_names = list(self.codebooks)
        widths = {cb.shape[-1] for cb in self.codebooks.values()}
        if len(widths) != 1:
            raise ValueError(f"every codebook must be the same width, got {sorted(widths)}")
        self.dim = widths.pop()
        self.max_iterations = _builtins.int(max_iterations)
        self.convergence_threshold = _builtins.float(convergence_threshold)
        self.restarts = _builtins.max(1, _builtins.int(restarts))
        self.generator = generator

    def factorize(self, composite, init_estimates=None):
        """Recover one entry per factor.

        Returns ``(indices, estimates, iterations)``: the codebook index of
        each factor, the vectors themselves, and how many iterations the
        winning run took.
        """
        if init_estimates is not None:
            return self._run(composite, init_estimates)

        # Several restarts: the dynamics have fixed points that are not the
        # answer, and the first guess decides which one is reached. With one
        # factor there is nothing to guess, so one pass settles it.
        restarts = self.restarts if len(self.factor_names) > 1 else 1
        best = None
        for attempt in range(restarts):
            if attempt == 0:
                init = {n: self._nearest(n, composite)[1] for n in self.factor_names}
            else:
                init = {n: self._random_entry(n) for n in self.factor_names}
            indices, estimates, iterations = self._run(composite, init)

            rebuilt = None
            for name in self.factor_names:
                rebuilt = estimates[name] if rebuilt is None else self.algebra.bind(rebuilt, estimates[name])
            score = self.algebra.similarity(rebuilt, composite)
            if best is None or score > best[0]:
                best = (score, indices, estimates, iterations)
            if score > 0.99:                      # an exact hit; nothing to improve on
                break
        return best[1], best[2], best[3]

    def factorize_known(self, composite, known):
        """The remaining factor, when all the others are known.

        No search: unbind each known factor and what is left is the answer.
        Exact for an algebra whose unbinding is (bipolar, block codes).
        """
        out = composite
        for factor in known.values():
            out = self.algebra.unbind(out, factor)
        return out

    # ── the dynamics ─────────────────────────────────────────────────────

    def _run(self, composite, init_estimates):
        estimates = dict(init_estimates)
        iterations = 0
        indices = {}
        for iterations in range(1, self.max_iterations + 1):
            settled = True
            fresh = {}
            for name in self.factor_names:
                unbound = composite
                for other, estimate in estimates.items():
                    if other != name:
                        unbound = self.algebra.unbind(unbound, estimate)
                index, match = self._nearest(name, unbound)
                if self.algebra.similarity(estimates[name], match) < self.convergence_threshold:
                    settled = False
                fresh[name], indices[name] = match, index
            estimates = fresh
            if settled:
                break
        return indices, estimates, iterations

    def _nearest(self, name: str, vector):
        """The codebook entry most like ``vector``, and its index."""
        codebook = self.codebooks[name]
        index = _builtins.int(self.algebra.similarity_batch(vector, codebook).argmax().item())
        return index, codebook[index]

    def _random_entry(self, name: str) -> Tensor:
        codebook = self.codebooks[name]
        index = _builtins.int(
            _g.randint(0, codebook.shape[0], (1,), generator=self.generator).item()
        )
        return codebook[index]
