"""Learning rules that are not gradient descent.

Three unrelated things that v1 kept in one file, kept together here because
that is where a v1 user will look for them:

- **Elastic weight consolidation** — `fisher_info`, `ewc_penalty`,
  `natural_gradient`, `fisher_normalize`. Training a model on a second task
  overwrites the first unless something holds the parameters that mattered
  in place. EWC (Kirkpatrick et al., 2017) measures "mattered" as the
  diagonal Fisher information — the mean squared gradient — and adds a
  quadratic penalty around the old values, weighted by it.
- **Normalized least mean squares** — `nlms_predict`, `nlms_update`, and
  friends. A one-layer online regressor whose step is scaled by the input's
  own energy, so it converges at the same rate whatever scale the features
  arrive at. Useful as a cheap always-learning readout beside a frozen
  model.
- **Whitening** — `whitening_transform` and friends. Per-feature mean and
  standard deviation, and the transform that applies them.

All of it is element-wise arithmetic and a dot product or two, so none of
it needs a kernel; what it needs is to run on the same tensors as
everything else, which is what v1's numpy version could not do.

v1's ``grilly.functional.learning``. The epsilons are v1's, because the
numbers have to match for a checkpoint written under one to behave the same
under the other.
"""

from __future__ import annotations

import builtins as _builtins

import grilly as _g
from . import Tensor, _require_tensor

__all__ = [
    "ewc_penalty", "fisher_info", "fisher_normalize", "natural_gradient",
    "nlms_ensemble", "nlms_metrics", "nlms_predict", "nlms_update",
    "whitening_apply", "whitening_batch_stats", "whitening_transform",
]

#: v1's constants, kept so the arithmetic matches.
_WHITEN_EPS = 1e-8


# ── elastic weight consolidation ─────────────────────────────────────────


def fisher_info(gradients, fisher=None, momentum: _builtins.float = 0.9,
                use_ema: _builtins.bool = True, reset: _builtins.bool = False) -> Tensor:
    """Fold a step's gradients into a running diagonal Fisher estimate.

    The Fisher's diagonal is the expected squared gradient, so this is a
    moving average of ``gradients ** 2`` — an EMA with ``momentum`` by
    default, or a plain sum with ``use_ema=False``. ``reset`` starts over,
    and ``fisher=None`` means the same thing.
    """
    gradients = _require_tensor("gradients", gradients)
    squared = gradients.pow(2)
    if reset or fisher is None:
        return (1.0 - momentum) * squared if use_ema else squared
    fisher = _require_tensor("fisher", fisher)
    if use_ema:
        return momentum * fisher + (1.0 - momentum) * squared
    return fisher + squared


def ewc_penalty(current_params, important_params, fisher,
                lambda_ewc: _builtins.float = 1.0) -> _builtins.float:
    """``0.5 * lambda * sum(F * (theta - theta_old) ** 2)``.

    The quadratic wall around the old solution. Add it to the loss; a
    parameter with a large Fisher entry is one the old task depended on,
    and this makes moving it expensive.
    """
    current = _require_tensor("current_params", current_params)
    old = _require_tensor("important_params", important_params)
    f = _require_tensor("fisher", fisher)
    diff = current - old
    return 0.5 * lambda_ewc * _builtins.float((f * diff * diff).sum().item())


def natural_gradient(gradients, fisher, eps: _builtins.float = 1e-8) -> Tensor:
    """``gradients / (fisher + eps)`` — the diagonal natural gradient.

    Dividing by the Fisher rescales each direction by how much the model's
    output actually moves along it, so a step means the same amount of
    change everywhere. The diagonal is the cheap approximation; the full
    Fisher is a matrix nobody stores.
    """
    gradients = _require_tensor("gradients", gradients)
    return gradients / (_require_tensor("fisher", fisher) + eps)


def fisher_normalize(fisher) -> Tensor:
    """Scale a Fisher estimate to sum to 1, leaving an all-zero one alone.

    Makes ``lambda_ewc`` mean the same thing from task to task, where the
    raw scale depends on the loss and the batch size.
    """
    f = _require_tensor("fisher", fisher)
    total = _builtins.float(f.sum().item())
    return f / total if total > 0.0 else f


# ── normalized least mean squares ────────────────────────────────────────


def nlms_predict(x, w, bias: _builtins.float = 0.0) -> _builtins.float:
    """``x . w + bias``."""
    x, w = _require_tensor("x", x), _require_tensor("w", w)
    return _builtins.float((x * w).sum().item()) + _builtins.float(bias)


def nlms_update(x, y_true: _builtins.float, w, bias: _builtins.float,
                mu: _builtins.float = 0.5, eps: _builtins.float = 1e-6):
    """One NLMS step. Returns ``(weights, bias)``.

    The step is ``mu * error / (x . x + eps)``, which is what the
    "normalized" means: dividing by the input's energy makes the correction
    the same size whatever scale the features come in at, so one learning
    rate works across them.
    """
    x, w = _require_tensor("x", x), _require_tensor("w", w)
    error = _builtins.float(y_true) - nlms_predict(x, w, bias)
    energy = _builtins.float((x * x).sum().item())
    step = _builtins.float(mu) * error / (energy + eps)
    return w + step * x, _builtins.float(bias) + step


def nlms_ensemble(x, weights_list, biases_list) -> Tensor:
    """Every member's prediction for one input, as ``(members,)``."""
    if len(weights_list) != len(biases_list):
        raise ValueError(
            f"nlms_ensemble: {len(weights_list)} weight vectors and {len(biases_list)} biases"
        )
    return _g.tensor([nlms_predict(x, w, b) for w, b in zip(weights_list, biases_list)])


def nlms_metrics(errors, update_count: _builtins.int) -> dict:
    """``{"rmse", "mae", "update_count"}`` over a run of errors."""
    errors = _require_tensor("errors", errors)
    return {
        "rmse": _builtins.float(errors.pow(2).mean().item()) ** 0.5,
        "mae": _builtins.float(errors.abs().mean().item()),
        "update_count": _builtins.int(update_count),
    }


# ── whitening ────────────────────────────────────────────────────────────


def whitening_batch_stats(data):
    """Per-feature ``(mean, std)`` over the batch dimension."""
    data = _require_tensor("data", data)
    mean = data.mean(0)
    return mean, (data - mean).pow(2).mean(0).sqrt()


def whitening_apply(data, mean, std) -> Tensor:
    """``(data - mean) / (std + 1e-8)`` with precomputed statistics."""
    data = _require_tensor("data", data)
    return (data - _require_tensor("mean", mean)) / (_require_tensor("std", std) + _WHITEN_EPS)


def whitening_transform(data, mean=None, std=None):
    """Whiten ``(batch, features)``, returning ``(whitened, mean, std)``.

    With ``mean`` and ``std`` given, applies them; otherwise measures them
    from this batch. Keep the returned pair and pass it back for anything
    that has to be whitened the same way — measuring again on a different
    batch gives a different transform.
    """
    data = _require_tensor("data", data)
    if mean is None or std is None:
        batch_mean, batch_std = whitening_batch_stats(data)
        mean = batch_mean if mean is None else _require_tensor("mean", mean)
        std = batch_std if std is None else _require_tensor("std", std)
    return whitening_apply(data, mean, std), mean, std
