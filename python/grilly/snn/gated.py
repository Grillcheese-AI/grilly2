"""Gated leaky integrate-and-fire: minGRU's recurrence as a spiking neuron.

    from grilly.snn import GatedLIFCell

    cell = GatedLIFCell(64, 128)            # spikes out, soft reset
    for x_t in events:                      # one call per event, not per clock tick
        s = cell(x_t)
    cell.reset()

minGRU (Feng et al., "Were RNNs All We Needed?") is

    z_t = sigmoid(W_z x_t),   h~_t = W_h x_t,   h_t = (1 - z_t) h_{t-1} + z_t h~_t

and a leaky integrate-and-fire membrane is ``v_t = beta v_{t-1} + I_t``. They
are the same equation once the leak depends on the input: ``beta_t = 1 - z_t``
(an activity-dependent conductance -- shunting inhibition), and the input is
scaled by the same gate, ``z_t I_t``, so the membrane is always a convex mix of
what it held and what just arrived. :class:`LIFNode` has one fixed leak per
layer; this cell has one per neuron per input.

Then fire and reset exactly as :class:`~grilly.snn.neuron.BaseNode` does --
a spike where ``v`` passes ``v_threshold`` (through a surrogate gradient), and
a soft reset (subtract the threshold, ``v_reset=None``, the default here) or a
hard one (back to ``v_reset``).

``v_threshold=None`` turns firing off: the cell is then minGRU itself and
returns the membrane. That mode is also the only one with a parallel form --
the reset makes each step depend on whether the last one fired, so a spiking
sequence is stepped; a silent one goes through the same log-domain scan as
``grilly.infer.models.cubby.MinGRU`` and is tested against the stepped path.

Unlike the neurons in :mod:`grilly.snn.neuron`, this one owns its weights:
the gate is computed from the input, so the cell has to see the input before
the leak exists. It is a layer, not an elementwise node.

Built for CubbyLLM's external layers -- state that lives outside the language
model and changes only when something happens (a VM step returns a value),
which is what a spiking membrane models and a per-token recurrence does not.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _require_tensor
from ..nn import Linear, Module
from .surrogate import ATan, SurrogateFunction

__all__ = ["GatedLIFCell"]

#: log(1 - z) is floored here; z = 1 is "forget everything", which the scan
#: cannot take a log of. exp(-30) is far below float32's step at 1.
_LOG_FLOOR = 1e-13
#: added inside log(x+) / log(x-) so a zero input part has a log at all.
_SCAN_EPS = 1e-30


class GatedLIFCell(Module):
    """``v = (1 - z) v + z I`` with ``z = sigmoid(W_z x)``, ``I = W_h x``; then fire and reset.

    ``v_threshold`` is where it fires, or ``None`` for no firing (minGRU, the
    membrane is the output). ``v_reset`` is where a neuron that fired lands, or
    ``None`` (the default) for a soft reset that keeps the overshoot.
    ``detach_reset`` stops the gradient through the reset. ``step_mode`` is
    ``'s'`` (one ``(N, in)`` step) or ``'m'`` (a ``(T, N, in)`` sequence, time
    first), as for every neuron in :mod:`grilly.snn`.
    """

    def __init__(self, in_features: _builtins.int, hidden_features: _builtins.int,
                 v_threshold: _builtins.float | None = 1.0, v_reset: _builtins.float | None = None,
                 surrogate_function: SurrogateFunction | None = None,
                 detach_reset: _builtins.bool = False, step_mode: str = "s",
                 bias: _builtins.bool = True) -> None:
        super().__init__()
        if step_mode not in ("s", "m"):
            raise ValueError(f"step_mode must be 's' (single) or 'm' (multi), got {step_mode!r}")
        self.in_features, self.hidden_features = _builtins.int(in_features), _builtins.int(hidden_features)
        self.gate = Linear(in_features, hidden_features, bias=bias)
        self.current = Linear(in_features, hidden_features, bias=bias)
        self.v_threshold = None if v_threshold is None else _builtins.float(v_threshold)
        self.v_reset = None if v_reset is None else _builtins.float(v_reset)
        self.surrogate_function = surrogate_function if surrogate_function is not None else ATan()
        self.detach_reset = _builtins.bool(detach_reset)
        self.step_mode = step_mode
        self.v: Tensor | None = None

    # ── state ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Forget the membrane. Call between sequences."""
        self.v = None

    def _start(self, like: Tensor) -> Tensor:
        if self.v is not None:
            return self.v
        rest = 0.0 if self.v_reset is None else self.v_reset
        return _g.full(tuple(like.shape), rest)

    def _terms(self, x: Tensor):
        """The gate and the current, from the input alone -- no ``v`` in either."""
        return self.gate(x).sigmoid(), self.current(x)

    # ── fire and reset ───────────────────────────────────────────────────

    def _fire_reset(self, h: Tensor) -> Tensor:
        spike = self.surrogate_function(h - self.v_threshold)
        s = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = h - self.v_threshold * s
        else:
            self.v = h * (1.0 - s) + self.v_reset * s
        return spike

    # ── stepping ─────────────────────────────────────────────────────────

    def single_step_forward(self, x) -> Tensor:
        x = _require_tensor("x", x)
        z, current = self._terms(x)
        v = self._start(z)
        h = (1.0 - z) * v + z * current
        if self.v_threshold is None:
            self.v = h
            return h
        return self._fire_reset(h)

    def multi_step_forward(self, x_seq) -> Tensor:
        """``(T, N, in)`` in, ``(T, N, hidden)`` out: spikes, or the membrane when
        ``v_threshold is None`` -- that one by the parallel scan."""
        x_seq = _require_tensor("x_seq", x_seq)
        if x_seq.dim() < 2:
            raise ValueError(f"a multi-step input is (T, N, ...), got {tuple(x_seq.shape)}")
        if self.v_threshold is None:
            return self._scan(x_seq)
        return _g.stack([self.single_step_forward(x_seq[t]) for t in range(x_seq.shape[0])])

    def _scan(self, x_seq: Tensor) -> Tensor:
        """All T membranes at once: with ``a_t = 1 - z_t`` and ``a*_t = sum log a``,
        ``v_t = exp(a*_t) v_0 + exp(a*_t) sum_{s<=t} exp(-a*_s) z_s I_s``, the sum
        as ``logcumsumexp`` over the positive and negative parts apart (the log
        of a signed sum does not exist)."""
        z, current = self._terms(x_seq)
        t = x_seq.shape[0]
        v0 = self._start(z[0])
        a_star = (1.0 - z).clamp(min=_LOG_FLOOR).log().cumsum(0)
        drive = z * current
        pos = (drive.clamp(min=0.0) + _SCAN_EPS).log() - a_star
        neg = ((-drive).clamp(min=0.0) + _SCAN_EPS).log() - a_star
        h = (a_star + pos.logcumsumexp(0)).exp() - (a_star + neg.logcumsumexp(0)).exp() + a_star.exp() * v0
        self.v = h.narrow(0, t - 1, 1).reshape(*tuple(h.shape[1:]))
        return h

    def forward(self, x) -> Tensor:
        return self.single_step_forward(x) if self.step_mode == "s" else self.multi_step_forward(x)

    def extra_repr(self) -> str:
        return (f"{self.in_features}, {self.hidden_features}, v_threshold={self.v_threshold}, "
                f"v_reset={self.v_reset}, detach_reset={self.detach_reset}, "
                f"step_mode='{self.step_mode}', surrogate_function={self.surrogate_function}")
