"""Deprecated: v1's ``grilly.nn.snn``.

A grab-bag of ~14 classes. What grilly2 kept from it is in ``grilly.snn``,
and most of the rest was a spiking spelling of an ordinary layer.

``HebbianLayer`` and ``STDPLayer`` are the two worth naming: they were a
``Linear`` plus a local update rule, and the rule is the part worth having.
``grilly.snn.plasticity`` has them as functions over a weight you own,
which composes with anything — an ``nn.Linear``, a convolution, a weight
you keep yourself — rather than only with the layer v1 built in.
"""

from __future__ import annotations

from .._v1 import v1_module_getattr

__all__: list[str] = []

_REPLACED = {
    "HebbianLayer": (
        "grilly.nn.Linear with grilly.snn.plasticity.hebbian_update on its weight — "
        "the rule applies to any weight, not only to the layer v1 wrapped it in"
    ),
    "STDPLayer": (
        "grilly.nn.Linear with grilly.snn.plasticity.STDPLearner on its weight. Note "
        "the rule changed: v1's STDP had no timing asymmetry (see _bridge.stdp_learning)"
    ),
    "LIFNeuron": "grilly.snn.LIFNode",
    "GIFNeuron": "not ported — grilly.snn has IFNode, LIFNode and ParametricLIFNode",
    "SNNLayer": "grilly.nn.Linear followed by a grilly.snn neuron",
    "SNNMatMul": "grilly.nn.Linear — a spiking input is still a tensor",
    "SNNRMSNorm": "grilly.nn.RMSNorm",
    "SNNSoftmax": "grilly.nn.functional.softmax",
    "SNNReadout": "grilly.nn.Linear over the spike counts",
    "SynapseFilter": "not ported yet (grilly.snn)",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
