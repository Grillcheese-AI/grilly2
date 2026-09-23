"""Spiking and neuromorphic pieces.

What is here so far is the part of v1's SNN work that is self-contained
and needed no rewrite:

- :mod:`grilly.snn.cells` — place cells, time cells, theta-gamma coupling;
- :mod:`grilly.snn.encoding` — continuous values to spike trains and back;
- :mod:`grilly.snn.hippocampus` — ``DentateGyrus``, sparse expansion with
  k-winners-take-all.

The neuron models themselves (LIF, IF, parametric LIF, the surrogate
gradients, the synapse and container types) are **not** here yet. v1 built
them on its own autograd ``Variable``, so bringing them over is a rewrite
onto ``grilly.nn.Module`` with real gradients through surrogate functions,
not a translation — see ``docs/v1-migration.md``. Until then, the v1 names
raise and say so rather than half-working.
"""

from __future__ import annotations

from . import cells, encoding, hippocampus  # noqa: F401
from .cells import place_cell, theta_gamma_encoding, time_cell  # noqa: F401
from .encoding import continuous_to_spikes, spikes_to_continuous  # noqa: F401
from .hippocampus import DentateGyrus  # noqa: F401

__all__ = [
    "DentateGyrus", "cells", "continuous_to_spikes", "encoding", "hippocampus",
    "place_cell", "spikes_to_continuous", "theta_gamma_encoding", "time_cell",
]
