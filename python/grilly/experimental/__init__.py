"""Deprecated: grilly v1's ``experimental`` namespace.

What grilly2 kept from it lives under its own name now — the VSA algebras
are ``grilly.vsa``. The aliases here keep v1's import paths working; see
docs/v1-migration.md for the whole table.
"""

from __future__ import annotations

from .._v1 import v1_module_getattr

from . import cognitive  # noqa: F401  -- present so its names can raise by name

#: v1 names this package held that grilly2 does not reimplement.
_REPLACED = {
    "language": "the torch API — build these as grilly.nn modules",
    "moe": "the torch API — build these as grilly.nn modules",
    "temporal": "the torch API — build these as grilly.nn modules",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
