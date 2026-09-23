"""Deprecated: v1's ``grilly.nn.capsule``.

Reading this group closely is what decided how it was migrated, and the
answer was not the one expected: **two of its three layers are
``nn.Linear``**.

``CapsuleProject`` and ``SemanticEncoder`` have byte-identical ``__init__``
bodies once their default dimensions and shader names are normalised, and
both return exactly ``x @ self.weight.T + self.bias``. Their own source
says so — "Use linear backend for now (capsule-project is similar to
linear)". They are a Linear with a Xavier-uniform weight and a zero bias,
under two names, neither of which is about capsules.

So they raise, with the replacement spelled out. The parameters keep the
same two names and shapes, so a checkpoint written under either loads into
an ``nn.Linear`` unchanged.

``DentateGyrus`` was the real one — a sparse projection followed by
k-winners-take-all — and it is ported, as ``grilly.snn.DentateGyrus``.
"""

from __future__ import annotations

from .._v1 import v1_module_getattr

__all__: list[str] = []

_LINEAR = (
    "grilly.nn.Linear(in_dim, out_dim) — this was a Linear under another name: "
    "x @ weight.T + bias, with the same two parameters, so a checkpoint loads "
    "straight into one. For v1's initialization, "
    "grilly.nn.init.xavier_uniform_(layer.weight) and zero the bias; grilly.nn.Linear "
    "defaults to torch's kaiming_uniform instead"
)

_REPLACED = {
    "CapsuleProject": _LINEAR,
    "SemanticEncoder": _LINEAR,
    "DentateGyrus": "grilly.snn.DentateGyrus — the one layer here that was not a Linear",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
