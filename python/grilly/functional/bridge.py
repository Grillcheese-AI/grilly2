"""Deprecated: v1's ``grilly.functional.bridge``. Use ``grilly.snn.encoding``.

Same signatures, numpy in and numpy out. One difference, loud on purpose:
v1 took an ``encoding_type`` of 0 (rate), 1 (temporal) or 2 (phase) and
**implemented only rate**, so asking for temporal or phase quietly gave
rate coding back. Here anything but rate raises.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly._v1 import moved
from grilly.snn import encoding as _encoding

__all__ = ["bridge_temporal_weights", "continuous_to_spikes", "spikes_to_continuous"]


def _in(a):
    return None if a is None else grilly.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def _rate_only(encoding_type: int, where: str) -> None:
    if int(encoding_type) != 0:
        raise NotImplementedError(
            f"{where}: encoding_type {encoding_type} (temporal or phase coding) was never "
            f"implemented. v1 accepted the argument and rate-coded anyway; asking for another "
            f"now says so rather than returning the wrong code."
        )


def continuous_to_spikes(continuous, num_timesteps: int = 10, encoding_type: int = 0,
                         projection_weights=None, projection_bias=None) -> np.ndarray:
    """``(batch, features)`` to ``(batch, num_timesteps, spike_dim)`` spikes."""
    moved("grilly.functional.bridge.continuous_to_spikes",
          "grilly.snn.encoding.continuous_to_spikes")
    _rate_only(encoding_type, "continuous_to_spikes")
    return _encoding.continuous_to_spikes(
        _in(continuous), num_timesteps,
        projection_weights=_in(projection_weights), projection_bias=_in(projection_bias),
    ).numpy()


def spikes_to_continuous(spikes, encoding_type: int = 0, time_window: int = 5,
                         temporal_weights=None, projection_weights=None,
                         projection_bias=None) -> np.ndarray:
    """``(batch, time, features)`` spikes to ``(batch, output_dim)`` values."""
    moved("grilly.functional.bridge.spikes_to_continuous",
          "grilly.snn.encoding.spikes_to_continuous")
    _rate_only(encoding_type, "spikes_to_continuous")
    if temporal_weights is not None:
        raise NotImplementedError(
            "spikes_to_continuous: temporal_weights belongs to the temporal code, which was "
            "never implemented. v1 ignored this argument."
        )
    return _encoding.spikes_to_continuous(
        _in(spikes), time_window,
        projection_weights=_in(projection_weights), projection_bias=_in(projection_bias),
    ).numpy()


def bridge_temporal_weights(weights, temporal_window: int = 10) -> np.ndarray:
    """Deprecated and **not** ported.

    v1's implementation multiplied the whole weight matrix by one scalar,
    ``exp(-(window - 1) / window)``, and its own comment called it
    "Simplified". There is nothing here worth preserving, so it says that
    instead of doing it.
    """
    raise NotImplementedError(
        "grilly.functional.bridge.bridge_temporal_weights is not ported: v1's implementation "
        "scaled every weight by a single constant and was marked Simplified in its own source. "
        "Scale the weights yourself if that is what you wanted."
    )
