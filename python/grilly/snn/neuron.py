"""Spiking neurons: integrate-and-fire, with and without a leak.

    import grilly, grilly.nn as nn
    from grilly.snn import LIFNode, functional as SF

    net = nn.Sequential(nn.Linear(28 * 28, 100), LIFNode(), nn.Linear(100, 10), LIFNode())
    for x_t in stimulus:          # one timestep at a time
        out = net(x_t)
    SF.reset_net(net)             # membranes carry over until you say otherwise

Every neuron here is the same three steps, and the only difference between
them is the first:

1. **charge** — fold this step's input into the membrane, giving ``h``;
2. **fire** — ``h`` over the threshold is a spike, through a surrogate
   gradient (:mod:`grilly.snn.surrogate`);
3. **reset** — a neuron that fired goes back to ``v_reset`` (a *hard*
   reset), or has the threshold subtracted (a **soft** reset, which is
   ``v_reset=None``) — soft keeps the overshoot, so a strongly driven
   neuron fires again sooner.

spikingjelly's API: the class names, the argument names, ``step_mode`` of
``'s'`` or ``'m'``, and a multi-step input shaped ``(T, N, ...)`` with time
first.

**The membrane is state, not a buffer.** It persists across calls, which is
the point of a spiking network and the thing that catches people: call
``reset_net`` between sequences or the next one starts charged. It is also
why these are not pure functions and do not belong inside a captured graph
without thought (docs/capture.md).

**One dispatch a step.** Charge, fire and reset are elementwise over the
same tensor, so composing them from ordinary ops means twelve passes over
it. ``grilly-snn`` fuses them into one kernel, and the backward into one
more: measured on an RX 6750 XT, the step alone goes from 12 dispatches
and 0.841 ms to 1 and 0.068 ms at (1024, 4096), and 15x at (64, 256) where
the launches rather than the memory were the cost. A 32-step training loop
with its accumulation and backward goes from 33 dispatches a step to 6.

The neurons call the kernel whenever their constants are constants, and
fall back to the composed expression when ``tau`` is a learned parameter,
which the kernel takes as a push constant and so cannot differentiate. The
two paths are tested **bit-exact** against each other: same arithmetic,
same order, nothing for a tolerance to absorb.

v1's ``grilly.nn.snn_neurons``, rewritten rather than translated: its
neurons carried hand-written ``backward`` methods over its own autograd, so
gradients reached one layer. These are ordinary ``grilly.nn.Module``s and
the gradient goes all the way back through time.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _require_tensor
from .. import _C
from ..nn import Module, Parameter
from .surrogate import ATan, FastSigmoid, Sigmoid, SurrogateFunction

__all__ = ["BaseNode", "IFNode", "LIFNode", "ParametricLIFNode"]

#: Which surrogates the fused kernel knows. One it does not is not a
#: failure: the neuron composes the step instead and the answer is the same.
_FUSED_SURROGATES = {ATan: "atan", Sigmoid: "sigmoid", FastSigmoid: "fast_sigmoid"}


class BaseNode(Module):
    """Charge, fire, reset — with ``neuronal_charge`` left to subclasses.

    ``v_threshold`` is where it fires. ``v_reset`` is where it lands after
    firing, or ``None`` for a soft reset. ``detach_reset`` stops the
    gradient flowing through the reset itself, which shortens the path
    through time and is often what makes a deep spiking net trainable.
    """

    def __init__(self, v_threshold: _builtins.float = 1.0, v_reset=0.0,
                 surrogate_function: SurrogateFunction | None = None,
                 detach_reset: _builtins.bool = False, step_mode: str = "s") -> None:
        super().__init__()
        if step_mode not in ("s", "m"):
            raise ValueError(f"step_mode must be 's' (single) or 'm' (multi), got {step_mode!r}")
        self.v_threshold = _builtins.float(v_threshold)
        self.v_reset = None if v_reset is None else _builtins.float(v_reset)
        self.surrogate_function = surrogate_function if surrogate_function is not None else ATan()
        self.detach_reset = _builtins.bool(detach_reset)
        self.step_mode = step_mode
        self.v: Tensor | None = None

    # ── the three steps ──────────────────────────────────────────────────

    def neuronal_charge(self, x: Tensor) -> Tensor:
        """This step's membrane, before firing. Subclasses define it."""
        raise NotImplementedError(f"{type(self).__name__} must define neuronal_charge")

    def neuronal_fire(self, h: Tensor) -> Tensor:
        """A spike where the membrane passed the threshold."""
        return self.surrogate_function(h - self.v_threshold)

    def neuronal_reset(self, h: Tensor, spike: Tensor) -> Tensor:
        """The membrane after firing: hard to ``v_reset``, or soft by
        subtracting the threshold."""
        s = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            return h - self.v_threshold * s
        return h * (1.0 - s) + self.v_reset * s

    # ── stepping ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget the membrane. Call between sequences."""
        self.v = None

    def _start(self, x: Tensor) -> Tensor:
        if self.v is not None:
            return self.v
        start = 0.0 if self.v_reset is None else self.v_reset
        return _g.full(tuple(x.shape), start)

    def _fused(self, x: Tensor):
        """``(decay, in_scale)`` if this step can go through the kernel.

        ``None`` when something about it cannot: a learned time constant,
        which the kernel takes as a push constant and so cannot
        differentiate, or a surrogate it does not implement.
        """
        return None

    def single_step_forward(self, x) -> Tensor:
        x = _require_tensor("x", x)
        self.v = self._start(x)
        fused = self._fused(x)
        if fused is not None:
            decay, in_scale = fused
            spike, self.v = _C.snn_lif_step(
                self.v._t, x._t, decay, in_scale,
                0.0 if self.v_reset is None else self.v_reset, self.v_threshold,
                self.v_reset is None, self.detach_reset,
                _FUSED_SURROGATES[type(self.surrogate_function)], self.surrogate_function.alpha,
            )
            spike, self.v = Tensor._wrap(spike), Tensor._wrap(self.v)
            return spike
        h = self.neuronal_charge(x)
        spike = self.neuronal_fire(h)
        self.v = self.neuronal_reset(h, spike)
        return spike

    def multi_step_forward(self, x_seq) -> Tensor:
        """``(T, N, ...)`` in, ``(T, N, ...)`` spikes out, time first."""
        x_seq = _require_tensor("x_seq", x_seq)
        if x_seq.dim() < 2:
            raise ValueError(f"a multi-step input is (T, N, ...), got {tuple(x_seq.shape)}")
        return _g.stack([self.single_step_forward(x_seq[t]) for t in range(x_seq.shape[0])])

    def forward(self, x) -> Tensor:
        return self.single_step_forward(x) if self.step_mode == "s" else self.multi_step_forward(x)

    def extra_repr(self) -> str:
        return (f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, "
                f"detach_reset={self.detach_reset}, step_mode='{self.step_mode}', "
                f"surrogate_function={self.surrogate_function}")


class IFNode(BaseNode):
    """Integrate-and-fire: ``h = v + x``.

    No leak at all, so the membrane holds whatever it has until something
    pushes it over. A perfect integrator of its input.
    """

    def neuronal_charge(self, x: Tensor) -> Tensor:
        return self.v + x

    def _fused(self, x: Tensor):
        if type(self.surrogate_function) in _FUSED_SURROGATES:
            return 1.0, 1.0
        return None


class LIFNode(BaseNode):
    """Leaky integrate-and-fire, with ``tau`` the membrane time constant.

    ``decay_input=False`` (the default, and v1's):

        h = (1 - 1/tau) * (v - v_reset) + v_reset + x

    the membrane leaks towards its resting value and the input arrives at
    full strength. ``decay_input=True`` is the textbook form, which divides
    the input by ``tau`` as well:

        h = v + (x - (v - v_reset)) / tau

    That is the physically-derived one, and it makes neurons fire much less
    for the same input, which is why the practical default is the other.
    ``tau = 1`` is an IF neuron with no memory at all; below 1 the membrane
    would grow, so it is refused.
    """

    def __init__(self, tau: _builtins.float = 2.0, decay_input: _builtins.bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        if tau < 1.0:
            raise ValueError(f"tau must be at least 1.0, got {tau}")
        self.tau = _builtins.float(tau)
        self.decay_input = _builtins.bool(decay_input)

    def _charge(self, x: Tensor, tau) -> Tensor:
        """``decay * (v - rest) + rest + in_scale * x`` — the one form.

        v1 wrote ``decay_input=True`` as ``v + (x - (v - rest)) / tau``,
        which is the same thing algebraically and a different order of
        float32 operations. Written this way the composed path and the
        fused kernel are bit-exact against each other, which is worth more
        than matching v1's rounding on a path v1 never ran on a GPU.
        """
        rest = 0.0 if self.v_reset is None else self.v_reset
        in_scale = (1.0 / tau) if self.decay_input else 1.0
        return (1.0 - 1.0 / tau) * (self.v - rest) + rest + in_scale * x

    def neuronal_charge(self, x: Tensor) -> Tensor:
        return self._charge(x, self.tau)

    def _fused(self, x: Tensor):
        if type(self.surrogate_function) in _FUSED_SURROGATES:
            return 1.0 - 1.0 / self.tau, (1.0 / self.tau) if self.decay_input else 1.0
        return None

    def extra_repr(self) -> str:
        return f"tau={self.tau}, decay_input={self.decay_input}, " + super().extra_repr()


class ParametricLIFNode(LIFNode):
    """A :class:`LIFNode` whose ``tau`` is learned.

    The time constant becomes a parameter, so the network chooses how long
    each layer remembers rather than being told. It is clamped to at least
    1.0 on every use, because below that the membrane grows instead of
    leaking — the clamp is in the graph, so the gradient stops there rather
    than pushing it further out of range.
    """

    def __init__(self, init_tau: _builtins.float = 2.0, decay_input: _builtins.bool = False, **kwargs) -> None:
        super().__init__(tau=_builtins.max(1.0, _builtins.float(init_tau)),
                         decay_input=decay_input, **kwargs)
        self.tau = Parameter(_g.full((1,), _builtins.float(init_tau)))

    def neuronal_charge(self, x: Tensor) -> Tensor:
        return self._charge(x, self.tau.clamp(min=1.0))

    def _fused(self, x: Tensor):
        # tau is a push constant in the kernel, so it would get no gradient.
        return None

    def extra_repr(self) -> str:
        tau = _builtins.float(self.tau.detach().reshape(1)[0].item())
        return (f"tau={tau:.3f} (learned), decay_input={self.decay_input}, "
                + BaseNode.extra_repr(self))
