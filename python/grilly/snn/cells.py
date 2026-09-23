"""Place cells, time cells, and theta-gamma coupling.

Hippocampal tuning curves: a population where each unit fires most when the
animal is at *its* place, or at *its* moment. Both are Gaussians over a
preferred value, which is what makes a population code out of them — the
pattern of activity names a position without any one unit doing so.

    from grilly.snn import cells

    centres = grilly.rand(64, 2) * 10          # 64 fields over a 10x10 arena
    rates = cells.place_cell(grilly.tensor([3.0, 4.0]), centres, field_width=0.8)

Rate coding, not spikes: these give firing *rates*, which
``grilly.snn.encoding.continuous_to_spikes`` turns into spike trains if that
is what you need.

v1's ``grilly.functional.cells``, on tensors instead of numpy arrays. No
kernels — a squared distance, an exponential and an outer product are ops
the framework already has, and the GPU is worth having here only because
the population is large.
"""

from __future__ import annotations

import builtins as _builtins
import math as _math

import grilly as _g
from .. import Tensor, _require_tensor

__all__ = ["place_cell", "theta_gamma_encoding", "time_cell"]

#: Keeps a zero width from dividing by zero. v1's constant, kept so the
#: rates match to the last digit at small widths.
_EPS = 1e-8


def _phases(n: _builtins.int) -> Tensor:
    """``n`` phases evenly spaced over a full cycle, endpoints included.

    ``torch.linspace`` by hand: grilly does not have it yet, and one caller
    is not a reason to grow the core surface.
    """
    if n < 1:
        raise ValueError(f"a cycle needs at least one phase, got {n}")
    if n == 1:
        return _g.zeros(1)
    return _g.arange(n, dtype=_g.float32) * (2 * _math.pi / (n - 1))


def place_cell(position, field_centers, field_width: _builtins.float = 1.0,
               max_rate: _builtins.float = 20.0, baseline_rate: _builtins.float = 0.1) -> Tensor:
    """Firing rates of a place-cell population.

    ``position`` is ``(dims,)`` or ``(batch, dims)``; ``field_centers`` is
    ``(cells, dims)``. Returns ``(cells,)`` or ``(batch, cells)``: a
    Gaussian of the distance from each field's centre, running from
    ``baseline_rate`` far away to ``max_rate`` at the centre.
    """
    position = _require_tensor("position", position)
    centers = _require_tensor("field_centers", field_centers)
    if centers.dim() != 2:
        raise ValueError(f"place_cell: field_centers must be (cells, dims), got {tuple(centers.shape)}")
    flat = position.dim() == 1
    if flat:
        position = position.reshape(1, position.shape[0])
    if position.dim() != 2 or position.shape[1] != centers.shape[1]:
        raise ValueError(
            f"place_cell: position must be (dims,) or (batch, dims) with dims = {centers.shape[1]}, "
            f"got {tuple(position.shape)}"
        )
    diff = position.unsqueeze(1) - centers.unsqueeze(0)          # (batch, cells, dims)
    d2 = diff.pow(2).sum(-1)
    rates = baseline_rate + (max_rate - baseline_rate) * (-d2 / (2.0 * field_width * field_width + _EPS)).exp()
    return rates.reshape(rates.shape[1]) if flat else rates


def time_cell(current_time: _builtins.float, preferred_times, temporal_width: _builtins.float = 1.0,
              max_rate: _builtins.float = 15.0, baseline_rate: _builtins.float = 0.1,
              membrane_state=None, leak: _builtins.float = 0.9):
    """Firing rates of a time-cell population, and its membrane state.

    The same Gaussian as ``place_cell`` but over time, plus a leaky
    membrane that carries a little of the last step: ``leak * state +
    (1 - leak) * rates``. Returns ``(rates, state)``; pass the state back in
    on the next step, or leave it ``None`` to start, where the state is the
    rates themselves.
    """
    preferred = _require_tensor("preferred_times", preferred_times)
    offset = preferred - _builtins.float(current_time)
    rates = baseline_rate + (max_rate - baseline_rate) * (
        -offset.pow(2) / (2.0 * temporal_width * temporal_width + _EPS)
    ).exp()
    if membrane_state is None:
        return rates, rates
    state = _require_tensor("membrane_state", membrane_state)
    return rates, leak * state + (1.0 - leak) * rates


def theta_gamma_encoding(time: _builtins.float, n_theta: _builtins.int = 8, n_gamma: _builtins.int = 40,
                         theta_freq: _builtins.float = 8.0, gamma_freq: _builtins.float = 40.0,
                         coupling_strength: _builtins.float = 0.5) -> Tensor:
    """Phase-amplitude coupling of a theta and a gamma rhythm, flattened.

    The outer product of two sinusoids sampled at ``n_theta`` and
    ``n_gamma`` phases: a ``(n_theta * n_gamma,)`` code whose value depends
    on where in the slow cycle the fast one sits. Nesting gamma cycles
    inside a theta cycle is the standard account of how a sequence of items
    fits in one hippocampal cycle.
    """
    theta = (2 * _math.pi * theta_freq * time + _phases(n_theta)).sin()
    gamma = (2 * _math.pi * gamma_freq * time + _phases(n_gamma)).sin()
    return (coupling_strength * theta.unsqueeze(1) * gamma.unsqueeze(0)).reshape(n_theta * n_gamma)
