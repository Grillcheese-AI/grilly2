"""Deprecated: v1's ``grilly.nn.snn_surrogate``. Use ``grilly.snn.surrogate``.

Same names, same ``alpha`` defaults, same formulas. The difference is that
these apply the gradient through grilly's autograd rather than being asked
for it by a hand-written backward.
"""

from __future__ import annotations

from .._v1 import moved
from ..snn import surrogate as _surrogate

__all__ = ["ATan", "FastSigmoid", "Sigmoid", "SurrogateFunction"]

SurrogateFunction = _surrogate.SurrogateFunction


class ATan(_surrogate.ATan):
    def __init__(self, *args, **kwargs) -> None:
        moved("grilly.nn.snn_surrogate.ATan", "grilly.snn.surrogate.ATan")
        super().__init__(*args, **kwargs)


class Sigmoid(_surrogate.Sigmoid):
    def __init__(self, *args, **kwargs) -> None:
        moved("grilly.nn.snn_surrogate.Sigmoid", "grilly.snn.surrogate.Sigmoid")
        super().__init__(*args, **kwargs)


class FastSigmoid(_surrogate.FastSigmoid):
    def __init__(self, *args, **kwargs) -> None:
        moved("grilly.nn.snn_surrogate.FastSigmoid", "grilly.snn.surrogate.FastSigmoid")
        super().__init__(*args, **kwargs)
