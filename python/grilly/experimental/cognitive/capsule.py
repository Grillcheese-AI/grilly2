"""Deprecated: v1's ``grilly.experimental.cognitive.capsule``.

``cosine_similarity`` and ``batch_cosine_similarity`` are
``grilly.nn.functional.cosine_similarity``; ``CapsuleEncoder`` is a fixed
seeded projection. See the package docstring for why each raises rather
than being reimplemented.
"""

from __future__ import annotations

from . import _REPLACED
from ..._v1 import v1_module_getattr

__all__: list[str] = []

__getattr__ = v1_module_getattr(__name__, _REPLACED)
