"""Seeded random numbers, like ``torch.random``.

The same seed gives the same numbers on any machine, but not the numbers
torch gives for that seed. The algorithm (Philox4x32-10 with Box-Muller) is
specified in grilly-core's ``rng`` module so any stream can be reproduced.
"""

from __future__ import annotations

import os

from . import _C

__all__ = ["Generator", "default_generator", "initial_seed", "manual_seed", "seed"]


def _seed_u64(seed) -> int:
    """torch accepts [-2**63, 2**64); negatives wrap to u64."""
    seed = int(seed)
    if -(1 << 63) <= seed < 0:
        return seed + (1 << 64)
    if 0 <= seed < (1 << 64):
        return seed
    raise ValueError("Overflow when unpacking long long")  # torch's type and message


class Generator:
    """A random stream, like ``torch.Generator``. Generators live on the host,
    as torch's CPU generators do; ``device`` may be None or "cpu"."""

    def __init__(self, device=None) -> None:
        if device not in (None, "cpu"):
            raise NotImplementedError(
                f"grilly: Generator(device={device!r}) is not implemented yet "
                "(generators are host-side, like torch's CPU generator)"
            )
        self._g = _C.Generator(_C.DEFAULT_SEED)

    def manual_seed(self, seed) -> "Generator":
        self._g.manual_seed(_seed_u64(seed))
        return self

    def initial_seed(self) -> int:
        return self._g.initial_seed()

    def seed(self) -> int:
        """Seed from the OS's entropy source and return the seed."""
        s = int.from_bytes(os.urandom(8), "little")
        self._g.manual_seed(s)
        return s

    def get_state(self):
        raise NotImplementedError("grilly: Generator.get_state is not implemented yet")

    def set_state(self, state):
        raise NotImplementedError("grilly: Generator.set_state is not implemented yet")


default_generator = Generator()


def manual_seed(seed) -> Generator:
    return default_generator.manual_seed(seed)


def initial_seed() -> int:
    return default_generator.initial_seed()


def seed() -> int:
    return default_generator.seed()
