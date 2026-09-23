"""Deprecated: v1's ``grilly.functional.snn``. Use ``grilly.snn.functional``.

``reset_net``, ``multi_step_forward``, ``seq_to_ann_forward`` and
``set_step_mode`` moved as they were. ``if_step`` and ``lif_step`` were
v1's element-wise neuron steps, which are ``grilly.snn.IFNode`` and
``grilly.snn.LIFNode`` — a module with its own membrane rather than a
function you hand the state to.
"""

from __future__ import annotations

from .._v1 import moved, v1_module_getattr
from ..snn import functional as _functional

__all__ = ["multi_step_forward", "reset_net", "seq_to_ann_forward", "set_step_mode"]


def reset_net(net) -> None:
    """Clear every spiking neuron's membrane in ``net``."""
    moved("grilly.functional.snn.reset_net", "grilly.snn.functional.reset_net")
    return _functional.reset_net(net)


def set_step_mode(net, step_mode: str) -> None:
    """Put every spiking neuron in ``net`` into 's' or 'm'."""
    moved("grilly.functional.snn.set_step_mode", "grilly.snn.functional.set_step_mode")
    return _functional.set_step_mode(net, step_mode)


def multi_step_forward(x_seq, module):
    """Run a single-step module over a (T, N, ...) sequence."""
    moved("grilly.functional.snn.multi_step_forward",
          "grilly.snn.functional.multi_step_forward")
    return _functional.multi_step_forward(x_seq, module)


def seq_to_ann_forward(x_seq, module):
    """Fold time into the batch for a stateless module."""
    moved("grilly.functional.snn.seq_to_ann_forward",
          "grilly.snn.functional.seq_to_ann_forward")
    return _functional.seq_to_ann_forward(x_seq, module)


_REPLACED = {
    "if_step": "grilly.snn.IFNode — a module that holds its own membrane",
    "lif_step": "grilly.snn.LIFNode — a module that holds its own membrane",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
