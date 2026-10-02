"""Spiking neural networks.

    import grilly.nn as nn
    from grilly.snn import LIFNode, functional as SF

    net = nn.Sequential(nn.Linear(784, 100), LIFNode(), nn.Linear(100, 10), LIFNode())
    for x_t in stimulus:
        out = net(x_t)
    SF.reset_net(net)

- :mod:`grilly.snn.neuron` — ``IFNode``, ``LIFNode``, ``ParametricLIFNode``;
- :mod:`grilly.snn.gated` -- ``GatedLIFCell``, minGRU as a spiking neuron
  (the leak gated by the input);
- :mod:`grilly.snn.surrogate` — the gradients that make a spike
  differentiable: ``ATan``, ``Sigmoid``, ``FastSigmoid``;
- :mod:`grilly.snn.functional` — ``reset_net``, ``set_step_mode``,
  ``multi_step_forward``;
- :mod:`grilly.snn.plasticity` — learning without a loss: Hebbian, Oja and
  spike-timing-dependent;
- :mod:`grilly.snn.encoding` — continuous values to spike trains and back;
- :mod:`grilly.snn.cells` — place cells, time cells, theta-gamma coupling;
- :mod:`grilly.snn.hippocampus` — ``DentateGyrus``, sparse expansion with
  k-winners-take-all.

spikingjelly's API, because it is the one people write against: the class
names, the argument names, ``step_mode``, and time-first ``(T, N, ...)``
sequences.

v1 had all of this built on its own autograd, with hand-written
``backward`` methods on each neuron; this is a rewrite onto
``grilly.nn.Module``, so a gradient goes all the way back through time.
Not ported yet: the synapse models, the ANN-to-SNN converter, the spiking
attention variants and the monitors — see ``docs/v1-migration.md``.
"""

from __future__ import annotations

from . import cells, encoding, functional, gated, hippocampus, neuron, plasticity, surrogate  # noqa: F401
from .cells import place_cell, theta_gamma_encoding, time_cell  # noqa: F401
from .encoding import continuous_to_spikes, spikes_to_continuous  # noqa: F401
from .gated import GatedLIFCell  # noqa: F401
from .functional import multi_step_forward, reset_net, set_step_mode  # noqa: F401
from .hippocampus import DentateGyrus  # noqa: F401
from .neuron import BaseNode, IFNode, LIFNode, ParametricLIFNode  # noqa: F401
from .plasticity import STDPLearner, hebbian_update, oja_update  # noqa: F401
from .surrogate import ATan, FastSigmoid, Sigmoid  # noqa: F401

__all__ = [
    "ATan", "BaseNode", "DentateGyrus", "FastSigmoid", "GatedLIFCell", "IFNode", "LIFNode",
    "ParametricLIFNode", "STDPLearner", "Sigmoid", "cells", "continuous_to_spikes",
    "encoding", "functional", "gated", "hebbian_update", "hippocampus", "multi_step_forward",
    "neuron", "oja_update", "place_cell", "plasticity", "reset_net",
    "set_step_mode", "spikes_to_continuous", "surrogate", "theta_gamma_encoding",
    "time_cell",
]
