"""Driving a spiking network: resetting it, and running it over time.

    from grilly.snn import functional as SF

    SF.set_step_mode(net, "m")         # every neuron takes (T, N, ...)
    spikes = net(stimulus)
    SF.reset_net(net)                  # before the next sequence

spikingjelly's names. The three that matter:

- :func:`reset_net` clears every membrane in a network. A spiking neuron
  keeps its state between calls, so without this the next sequence starts
  wherever the last one ended — which is the single most common way a
  spiking model produces results nobody can reproduce;
- :func:`set_step_mode` switches a whole network between one timestep a
  call and a whole sequence a call;
- :func:`multi_step_forward` runs a module that only knows single steps
  over a ``(T, N, ...)`` sequence, which is what wraps an ordinary
  ``nn.Linear`` for use inside one.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _require_tensor
from ..nn import Module
from .neuron import BaseNode

__all__ = ["multi_step_forward", "reset_net", "seq_to_ann_forward", "set_step_mode"]


def reset_net(net: Module) -> None:
    """Clear the membrane of every spiking neuron in ``net``, recursively."""
    for module in net.modules():
        reset = getattr(module, "reset", None)
        if module is not net and callable(reset) and isinstance(module, BaseNode):
            reset()
    if isinstance(net, BaseNode):
        net.reset()


def set_step_mode(net: Module, step_mode: str) -> None:
    """Put every spiking neuron in ``net`` into ``'s'`` or ``'m'``."""
    if step_mode not in ("s", "m"):
        raise ValueError(f"step_mode must be 's' or 'm', got {step_mode!r}")
    for module in net.modules():
        if isinstance(module, BaseNode):
            module.step_mode = step_mode


def multi_step_forward(x_seq, module: Module) -> Tensor:
    """Apply a single-step ``module`` at each step of a ``(T, N, ...)``
    sequence, stacking the results back along time."""
    x_seq = _require_tensor("x_seq", x_seq)
    if x_seq.dim() < 2:
        raise ValueError(f"a multi-step input is (T, N, ...), got {tuple(x_seq.shape)}")
    return _g.stack([module(x_seq[t]) for t in range(x_seq.shape[0])])


def seq_to_ann_forward(x_seq, module: Module) -> Tensor:
    """The same, but folding time into the batch for one call.

    Only valid for a module that treats its rows independently — a
    ``Linear``, a ``Conv2d``, an activation. It is the faster route because
    it is one dispatch instead of ``T``, and it is wrong for anything with
    state, which is why the spiking neurons do not use it.
    """
    x_seq = _require_tensor("x_seq", x_seq)
    if x_seq.dim() < 2:
        raise ValueError(f"a multi-step input is (T, N, ...), got {tuple(x_seq.shape)}")
    if isinstance(module, BaseNode):
        raise ValueError(
            "seq_to_ann_forward folds time into the batch, which loses the order the "
            "membrane depends on. Use multi_step_forward, or the neuron's own step_mode='m'."
        )
    shape = tuple(x_seq.shape)
    flat = module(x_seq.reshape(shape[0] * shape[1], *shape[2:]))
    return flat.reshape(shape[0], shape[1], *tuple(flat.shape)[1:])
