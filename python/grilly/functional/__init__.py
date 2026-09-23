"""Deprecated: grilly v1's top-level ``functional`` package.

Not ``grilly.nn.functional``, which is torch's ``F``. This one was v1's own
flat namespace of ~124 GPU helpers. What grilly2 kept moved to where it
belongs; the rest is torch's job:

| v1 | grilly2 |
|---|---|
| ``functional.cells`` | ``grilly.snn.cells`` |
| ``functional.bridge`` | ``grilly.snn.encoding`` |
| ``functional.activations``, ``.linear``, ``.loss``, ``.dropout``, ``.normalization``, ``.attention``, ``.embedding`` | ``grilly.nn.functional`` — the torch names |
| ``functional.snn`` | not ported yet (``grilly.snn``) |
| ``functional.learning``, ``.faiss``, ``.memory``, ``.fft``, ``.mf_activations`` | not ported yet |

See ``docs/v1-migration.md``.
"""

from __future__ import annotations

from . import bridge, cells  # noqa: F401
from .._v1 import v1_module_getattr
from .bridge import continuous_to_spikes, spikes_to_continuous  # noqa: F401
from .cells import place_cell, theta_gamma_encoding, time_cell  # noqa: F401

__all__ = [
    "bridge", "cells", "continuous_to_spikes", "place_cell",
    "spikes_to_continuous", "theta_gamma_encoding", "time_cell",
]

_REPLACED = {
    "gelu": "grilly.nn.functional.gelu",
    "relu": "grilly.nn.functional.relu",
    "silu": "grilly.nn.functional.silu",
    "softmax": "grilly.nn.functional.softmax",
    "softplus": "grilly.nn.functional.softplus",
    "linear": "grilly.nn.functional.linear",
    "dropout": "grilly.nn.functional.dropout",
    "layer_norm": "grilly.nn.functional.layer_norm",
    "cross_entropy": "grilly.nn.functional.cross_entropy",
    "binary_cross_entropy": "grilly.nn.functional.binary_cross_entropy",
    "attention": "grilly.nn.functional.scaled_dot_product_attention",
    "flash_attention2": "grilly.nn.functional.scaled_dot_product_attention",
    "embedding_lookup": "grilly.nn.Embedding",
    "activations": "grilly.nn.functional",
    "loss": "grilly.nn.functional",
    "normalization": "grilly.nn.functional",
    "attention_": "grilly.nn.functional.scaled_dot_product_attention",
}

__getattr__ = v1_module_getattr(__name__, _REPLACED)
