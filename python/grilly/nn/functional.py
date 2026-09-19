"""grilly.nn.functional, like torch.nn.functional."""

from __future__ import annotations

from .. import Tensor, _C, _require_tensor
from .. import relu as _relu

__all__ = ["linear", "relu"]


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``input @ weight.T + bias`` over the last dimension of ``input``."""
    x = _require_tensor("input", input)
    w = _require_tensor("weight", weight)
    b = None if bias is None else _require_tensor("bias", bias)._t
    return Tensor._wrap(_C.linear(x._t, w._t, b))


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        raise NotImplementedError("grilly: in-place relu is not implemented yet")
    return _relu(input)
