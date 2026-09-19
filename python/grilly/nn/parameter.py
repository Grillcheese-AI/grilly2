"""``torch.nn.Parameter``."""

from __future__ import annotations

from .. import Tensor, _require_tensor


class Parameter(Tensor):
    """A tensor that is a module parameter: a leaf that requires grad by
    default and that ``Module`` registers when assigned as an attribute.
    Shares data with the tensor it wraps, as torch's does."""

    __slots__ = ()

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            from .. import empty
            data = empty(0)
        t = Tensor._wrap.__func__(cls, _require_tensor("data", data)._t.detach())
        t.requires_grad_(requires_grad)
        return t

    def __init__(self, data=None, requires_grad=True) -> None:
        pass  # everything happens in __new__

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()
