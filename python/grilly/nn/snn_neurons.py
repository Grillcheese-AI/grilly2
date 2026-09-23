"""Deprecated: v1's ``grilly.nn.snn_neurons``. Use ``grilly.snn``.

Same spikingjelly names and arguments, rewritten as ``grilly.nn.Module``s
with real autograd: v1's neurons carried hand-written ``backward`` methods
over its own ``Variable``, so a gradient reached one layer. These take and
return ``grilly.Tensor``.
"""

from __future__ import annotations

from .._v1 import moved
from ..snn import neuron as _neuron

__all__ = ["BaseNode", "IFNode", "LIFNode", "ParametricLIFNode"]


class IFNode(_neuron.IFNode):
    def __init__(self, *args, **kwargs) -> None:
        moved("grilly.nn.snn_neurons.IFNode", "grilly.snn.IFNode")
        super().__init__(*args, **kwargs)


class LIFNode(_neuron.LIFNode):
    def __init__(self, *args, **kwargs) -> None:
        moved("grilly.nn.snn_neurons.LIFNode", "grilly.snn.LIFNode")
        super().__init__(*args, **kwargs)


class ParametricLIFNode(_neuron.ParametricLIFNode):
    def __init__(self, *args, **kwargs) -> None:
        moved("grilly.nn.snn_neurons.ParametricLIFNode", "grilly.snn.ParametricLIFNode")
        super().__init__(*args, **kwargs)


BaseNode = _neuron.BaseNode
