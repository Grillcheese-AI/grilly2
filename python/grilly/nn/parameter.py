"""``torch.nn.Parameter``."""

from __future__ import annotations

from .. import MetaTensor, Tensor, _require_tensor


class Parameter(Tensor):
    """A tensor that is a module parameter: a leaf that requires grad by
    default and that ``Module`` registers when assigned as an attribute.
    Shares data with the tensor it wraps, as torch's does."""

    __slots__ = ()

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            from .. import empty
            data = empty(0)
        if isinstance(data, MetaTensor):  # a parameter of a model built on the meta device
            return MetaParameter._new(data.shape, data.dtype, requires_grad)
        t = Tensor._wrap.__func__(cls, _require_tensor("data", data)._t.detach())
        t.requires_grad_(requires_grad)
        return t

    def __init__(self, data=None, requires_grad=True) -> None:
        pass  # everything happens in __new__

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super().__repr__()


class MetaParameter(Parameter, MetaTensor):
    """A parameter with no data: what ``Parameter(...)`` gives while a
    ``with device("meta")`` is open. It is a ``Parameter``, so a module
    registers it and ``state_dict`` reports its name and shape; it is a
    ``MetaTensor``, so touching its data raises."""

    __slots__ = ()

    def __new__(cls, data=None, requires_grad=True):
        return MetaTensor._new.__func__(cls, _require_tensor("data", data).shape,
                                        _require_tensor("data", data).dtype, requires_grad)

    def __repr__(self) -> str:
        return "Parameter containing:\n" + MetaTensor.__repr__(self)
