"""Surrogate gradients: a spike forward, something differentiable backward.

A neuron fires when its membrane crosses a threshold, which is a Heaviside
step. Its derivative is zero everywhere and infinite at one point, so a
network of them has no gradient at all. The standard fix is to keep the
step in the forward pass — the spikes have to be binary or nothing
downstream is a spiking network — and substitute a smooth bump for its
derivative on the way back.

    from grilly.snn import surrogate

    spike = surrogate.ATan()(v - threshold)     # 0.0 or 1.0, and differentiable

spikingjelly's names and defaults, so code written against it reads the
same here: ``ATan(alpha=2.0)``, ``Sigmoid(alpha=4.0)``,
``FastSigmoid(alpha=2.0)``. Larger ``alpha`` is a narrower, taller bump —
closer to the true derivative, and worse to train through, because a
neuron more than about ``1/alpha`` from its threshold gets no gradient.

**How it is done here.** Not with a custom autograd node:

    g = self.gradient(x).detach()       # the bump, held as a constant
    return heaviside(x) + (g * x - (g * x).detach())

The added term is exactly zero in the forward pass and has derivative
``g`` in the backward one, so the value is the true step and the gradient
is the surrogate. It is the straight-through estimator written out, it
composes with grilly's autograd like any other expression, and it works
for **any** ``gradient``, including one that has no antiderivative in the
ops available.

v1 wrote these with hand-rolled ``backward`` methods on the neurons, which
is why its gradients only reached one layer.
"""

from __future__ import annotations

import builtins as _builtins
import math as _math

import grilly as _g
from .. import Tensor, _require_tensor

__all__ = ["ATan", "FastSigmoid", "Sigmoid", "SurrogateFunction", "heaviside"]


def heaviside(x) -> Tensor:
    """``1.0`` where ``x >= 0``, else ``0.0``. No gradient path — that is
    what the surrogate is for."""
    return (_require_tensor("x", x) >= 0.0).to(_g.float32)


class SurrogateFunction:
    """Base class: subclasses give ``gradient``, the bump to use backward.

    ``alpha`` sets its width. Calling the object applies the step forward
    and that bump backward.
    """

    def __init__(self, alpha: _builtins.float = 2.0) -> None:
        self.alpha = _builtins.float(alpha)

    def gradient(self, x: Tensor) -> Tensor:
        """The surrogate derivative at ``x``, as a tensor of the same shape."""
        raise NotImplementedError(f"{type(self).__name__} must define gradient()")

    def __call__(self, x) -> Tensor:
        x = _require_tensor("x", x)
        spikes = heaviside(x)
        if not x.requires_grad or not _g.is_grad_enabled():
            return spikes
        straight = self.gradient(x).detach() * x
        return spikes + (straight - straight.detach())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self.alpha})"


class ATan(SurrogateFunction):
    """``alpha / (2 * (1 + (pi * alpha * x / 2) ** 2))`` — the derivative of
    a scaled arctangent, and spikingjelly's default.

    The heaviest tails of the three, so a neuron far from threshold still
    gets a little gradient. That is usually what you want early in
    training.
    """

    def __init__(self, alpha: _builtins.float = 2.0) -> None:
        super().__init__(alpha)

    def gradient(self, x: Tensor) -> Tensor:
        scaled = (_math.pi * self.alpha / 2.0) * x
        return self.alpha / (2.0 * (1.0 + scaled * scaled))


class Sigmoid(SurrogateFunction):
    """``alpha * s * (1 - s)`` with ``s = sigmoid(alpha * x)`` — the
    derivative of a logistic, which is the smoothed step itself."""

    def __init__(self, alpha: _builtins.float = 4.0) -> None:
        super().__init__(alpha)

    def gradient(self, x: Tensor) -> Tensor:
        s = (self.alpha * x).sigmoid()
        return self.alpha * s * (1.0 - s)


class FastSigmoid(SurrogateFunction):
    """``alpha / (2 * (1 + alpha * |x|) ** 2)`` — no exponential, so it is
    the cheap one, and its tails fall off fastest."""

    def __init__(self, alpha: _builtins.float = 2.0) -> None:
        super().__init__(alpha)

    def gradient(self, x: Tensor) -> Tensor:
        denom = 1.0 + self.alpha * x.abs()
        return self.alpha / (2.0 * denom * denom)
