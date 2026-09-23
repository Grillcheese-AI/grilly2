"""Deprecated: v1's ``grilly.experimental.vsa``. Use ``grilly.vsa``.

v1 kept four algebras here. They are all in ``grilly.vsa`` now, each in its
own namespace, and they are still four different algebras rather than four
spellings of one (CLAUDE.md §8.3):

| v1 | grilly2 |
|---|---|
| ``BlockCodeOps`` | ``grilly.vsa`` — sparse block codes, ``(..., k, l)`` |
| ``BinaryOps`` | ``grilly.vsa.bipolar`` — dense ±1 |
| ``HolographicOps`` | ``grilly.vsa.holographic`` — dense real, circular convolution |
| ``ResonatorNetwork`` | ``grilly.vsa.resonator`` |

(``grilly.vsa.packed`` is the fifth, and v1 had no class for it: it lived
in the bridge as ``hdc_*_packed``.)
"""

from __future__ import annotations

from . import block_ops, ops, resonator  # noqa: F401
from .block_ops import BlockCodeOps  # noqa: F401
from .ops import BinaryOps, HolographicOps  # noqa: F401
from .resonator import ResonatorNetwork  # noqa: F401

__all__ = [
    "BinaryOps", "BlockCodeOps", "HolographicOps", "ResonatorNetwork",
    "block_ops", "ops", "resonator",
]
