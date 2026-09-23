"""Deprecated: v1's ``grilly.experimental.cognitive``.

A working memory, a world model, a simulator, an "understander" — model
code, and grilly2 does not carry model code (CLAUDE.md §1). Build them as
``grilly.nn`` modules in the repo that uses them.

The two helpers here that other repos actually import are
``cosine_similarity`` and ``batch_cosine_similarity``, and those are
``grilly.nn.functional.cosine_similarity``: torch has the name, so the
torch name is what grilly2 provides. Both raise and say so, including the
exact call, because a capability check must not be able to mistake this for
a missing GPU.
"""

from __future__ import annotations

from ..._v1 import v1_module_getattr

__all__: list[str] = []

_COSINE = (
    "grilly.nn.functional.cosine_similarity — torch has this name. For one pair: "
    "F.cosine_similarity(a, b, dim=0). For a query against every row of an (n, dim) "
    "matrix: F.cosine_similarity(matrix, query.unsqueeze(0), dim=1)"
)

_REPLACED = {
    "cosine_similarity": _COSINE,
    "batch_cosine_similarity": _COSINE,
    "CapsuleEncoder": (
        "a fixed seeded projection you build yourself: draw a (capsule_dim, input_dim) "
        "weight with a Generator, normalize its rows, and apply it. It is four lines, "
        "and the three that mattered were specific to one caller's feature layout"
    ),
    "CognitiveController": "the torch API — this is model code, not framework",
    "WorkingMemory": "the torch API — this is model code, not framework",
    "WorkingMemorySlot": "the torch API — this is model code, not framework",
    "WorldModel": "the torch API — this is model code, not framework",
    "InternalSimulator": "the torch API — this is model code, not framework",
    "Understander": "the torch API — this is model code, not framework",
    "UnderstandingResult": "the torch API — this is model code, not framework",
    "SimulationResult": "the torch API — this is model code, not framework",
    "Fact": "the torch API — this is model code, not framework",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
