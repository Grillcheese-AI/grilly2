"""Deprecated: v1's ``grilly.functional.cells``. Use ``grilly.snn.cells``.

Same signatures, numpy in and numpy out, and now on the GPU: v1's versions
tried a bridge call its own extension did not implement and fell through to
numpy every time.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly._v1 import moved
from grilly.snn import cells as _cells

__all__ = ["place_cell", "theta_gamma_encoding", "time_cell"]


def _in(a):
    return grilly.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def place_cell(agent_position, field_centers, field_width: float = 1.0,
               max_rate: float = 20.0, baseline_rate: float = 0.1) -> np.ndarray:
    """Place-cell firing rates: ``(n_neurons,)`` for one position,
    ``(batch, n_neurons)`` for several."""
    moved("grilly.functional.cells.place_cell", "grilly.snn.cells.place_cell")
    return _cells.place_cell(_in(agent_position), _in(field_centers),
                             field_width, max_rate, baseline_rate).numpy()


def time_cell(current_time: float, preferred_times, temporal_width: float = 1.0,
              max_rate: float = 15.0, baseline_rate: float = 0.1,
              membrane_state=None):
    """``(firing_rates, updated_membrane_state)``."""
    moved("grilly.functional.cells.time_cell", "grilly.snn.cells.time_cell")
    rates, state = _cells.time_cell(
        current_time, _in(preferred_times), temporal_width, max_rate, baseline_rate,
        None if membrane_state is None else _in(membrane_state))
    return rates.numpy(), state.numpy()


def theta_gamma_encoding(time: float, n_theta: int = 8, n_gamma: int = 40,
                         theta_freq: float = 8.0, gamma_freq: float = 40.0,
                         coupling_strength: float = 0.5) -> np.ndarray:
    """``(n_theta * n_gamma,)`` phase-amplitude coupling."""
    moved("grilly.functional.cells.theta_gamma_encoding",
          "grilly.snn.cells.theta_gamma_encoding")
    return _cells.theta_gamma_encoding(time, n_theta, n_gamma, theta_freq,
                                       gamma_freq, coupling_strength).numpy()
