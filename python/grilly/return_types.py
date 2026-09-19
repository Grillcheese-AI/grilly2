"""torch.return_types: the named (values, indices) pairs that ``max(dim)``,
``min(dim)``, ``sort`` and ``topk`` return. They unpack and index like
tuples, as torch's do."""

from __future__ import annotations

from collections import namedtuple

__all__ = ["max", "min", "sort", "topk"]


def _pair(name: str):
    cls = namedtuple(name, ["values", "indices"])
    cls.__module__ = __name__
    cls.__repr__ = lambda self: f"grilly.return_types.{name}(\nvalues={self.values!r},\nindices={self.indices!r})"
    return cls


max = _pair("max")  # noqa: A001
min = _pair("min")  # noqa: A001
sort = _pair("sort")
topk = _pair("topk")
