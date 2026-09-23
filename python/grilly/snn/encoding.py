"""Between continuous values and spike trains.

The seam every spiking network needs at both ends: real numbers in, spikes
through the middle, real numbers out.

    from grilly.snn import encoding

    spikes = encoding.continuous_to_spikes(x, num_timesteps=20)   # (b, t, f)
    back = encoding.spikes_to_continuous(spikes, time_window=20)  # (b, f)

**Rate coding.** A value in [0, 1] is the probability that the unit fires
on any given step, so over ``num_timesteps`` the mean spike count
approaches the value — with sampling noise of about
``sqrt(x(1-x)/t)``, which is why 10 steps is coarse and 100 is not.
Decoding is the mean over a window, the exact inverse in expectation.

Values outside [0, 1] are **clamped**, not scaled: a rate cannot exceed one
spike a step, and silently rescaling would make the encoding depend on the
batch. Normalize before you get here if that is not what you want.

v1's ``grilly.functional.bridge``, on tensors. Its ``encoding_type``
argument offered rate, temporal and phase codes but only ever implemented
rate; the others are not carried over as dead parameters — see
``docs/v1-migration.md``.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from .. import Tensor, _require_tensor

__all__ = ["continuous_to_spikes", "spikes_to_continuous"]


def continuous_to_spikes(x, num_timesteps: _builtins.int = 10, *, generator=None,
                         projection_weights=None, projection_bias=None) -> Tensor:
    """Rate-code ``(batch, features)`` into ``(batch, time, features)`` spikes.

    Each step is an independent coin flip weighted by the (clamped) value,
    so the spike train is a Bernoulli sample and not a deterministic
    pattern. Pass a ``generator`` to make it reproducible.

    ``projection_weights`` ``(spike_dim, features)`` applies a linear map
    first, which is how a network of one width drives a spiking layer of
    another without a separate call.
    """
    x = _require_tensor("x", x)
    flat = x.dim() == 1
    if flat:
        x = x.reshape(1, x.shape[0])
    if x.dim() != 2:
        raise ValueError(f"continuous_to_spikes: expected (batch, features), got {tuple(x.shape)}")
    if projection_weights is not None:
        x = _g.nn.functional.linear(x, _require_tensor("projection_weights", projection_weights),
                                    None if projection_bias is None
                                    else _require_tensor("projection_bias", projection_bias))
    rate = x.clamp(0.0, 1.0)
    steps = _builtins.int(num_timesteps)
    if steps < 1:
        raise ValueError(f"continuous_to_spikes: num_timesteps must be at least 1, got {steps}")
    draws = _g.rand(x.shape[0], steps, x.shape[1], generator=generator)
    spikes = (draws < rate.unsqueeze(1)).to(_g.float32)
    return spikes.reshape(steps, x.shape[1]) if flat else spikes


def spikes_to_continuous(spikes, time_window: _builtins.int = 5, *,
                         projection_weights=None, projection_bias=None) -> Tensor:
    """Decode ``(batch, time, features)`` spikes to ``(batch, features)``.

    The mean over the **last** ``time_window`` steps, which is the inverse
    of the rate code in expectation. Taking the last steps rather than all
    of them is what lets a value change mid-train and still be read.
    """
    s = _require_tensor("spikes", spikes)
    if s.dim() != 3:
        raise ValueError(f"spikes_to_continuous: expected (batch, time, features), got {tuple(s.shape)}")
    window = _builtins.max(1, _builtins.min(_builtins.int(time_window), s.shape[1]))
    out = s[:, s.shape[1] - window:, :].mean(1)
    if projection_weights is not None:
        out = _g.nn.functional.linear(out, _require_tensor("projection_weights", projection_weights),
                                      None if projection_bias is None
                                      else _require_tensor("projection_bias", projection_bias))
    return out
