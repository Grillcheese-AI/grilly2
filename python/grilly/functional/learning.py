"""Deprecated: v1's ``grilly.functional.learning``. Use ``grilly.learning``.

Same signatures, numpy in and numpy out, same epsilons — the numbers have
to match, because a Fisher estimate or an NLMS weight vector saved under v1
has to keep behaving the same way here.
"""

from __future__ import annotations

import numpy as np

import grilly
from grilly import learning as _learning
from grilly._v1 import moved

__all__ = [
    "ewc_penalty", "fisher_info", "fisher_normalize", "natural_gradient",
    "nlms_ensemble", "nlms_metrics", "nlms_predict", "nlms_update",
    "whitening_apply", "whitening_batch_stats", "whitening_transform",
]


def _in(a):
    return None if a is None else grilly.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def fisher_info(gradients, fisher, momentum: float = 0.9, use_ema: bool = True,
                reset: bool = False) -> np.ndarray:
    """Update a running diagonal Fisher estimate from a step's gradients."""
    moved("grilly.functional.learning.fisher_info", "grilly.learning.fisher_info")
    return _learning.fisher_info(_in(gradients), _in(fisher), momentum, use_ema, reset).numpy()


def ewc_penalty(current_params, important_params, fisher, lambda_ewc: float = 1.0) -> float:
    """``0.5 * lambda * sum(F * (theta - theta_old) ** 2)``."""
    moved("grilly.functional.learning.ewc_penalty", "grilly.learning.ewc_penalty")
    return _learning.ewc_penalty(_in(current_params), _in(important_params), _in(fisher), lambda_ewc)


def natural_gradient(gradients, fisher, eps: float = 1e-8) -> np.ndarray:
    """``gradients / (fisher + eps)``."""
    moved("grilly.functional.learning.natural_gradient", "grilly.learning.natural_gradient")
    return _learning.natural_gradient(_in(gradients), _in(fisher), eps).numpy()


def fisher_normalize(fisher) -> np.ndarray:
    """Scale a Fisher estimate to sum to 1."""
    moved("grilly.functional.learning.fisher_normalize", "grilly.learning.fisher_normalize")
    return _learning.fisher_normalize(_in(fisher)).numpy()


def nlms_predict(x, w, bias: float = 0.0) -> float:
    """``x . w + bias``."""
    moved("grilly.functional.learning.nlms_predict", "grilly.learning.nlms_predict")
    return _learning.nlms_predict(_in(x), _in(w), bias)


def nlms_update(x, y_true: float, w, bias: float, mu: float = 0.5, eps: float = 1e-6):
    """One NLMS step: ``(updated_weights, updated_bias)``."""
    moved("grilly.functional.learning.nlms_update", "grilly.learning.nlms_update")
    weights, new_bias = _learning.nlms_update(_in(x), y_true, _in(w), bias, mu, eps)
    return weights.numpy(), new_bias


def nlms_ensemble(x, weights_list, biases_list) -> np.ndarray:
    """Every member's prediction for one input."""
    moved("grilly.functional.learning.nlms_ensemble", "grilly.learning.nlms_ensemble")
    return _learning.nlms_ensemble(_in(x), [_in(w) for w in weights_list], list(biases_list)).numpy()


def nlms_metrics(errors, update_count: int) -> dict:
    """``{"rmse", "mae", "update_count"}``."""
    moved("grilly.functional.learning.nlms_metrics", "grilly.learning.nlms_metrics")
    return _learning.nlms_metrics(_in(errors), update_count)


def whitening_batch_stats(data):
    """Per-feature ``(mean, std)`` over the batch dimension."""
    moved("grilly.functional.learning.whitening_batch_stats",
          "grilly.learning.whitening_batch_stats")
    mean, std = _learning.whitening_batch_stats(_in(data))
    return mean.numpy(), std.numpy()


def whitening_apply(data, mean, std) -> np.ndarray:
    """``(data - mean) / (std + 1e-8)``."""
    moved("grilly.functional.learning.whitening_apply", "grilly.learning.whitening_apply")
    return _learning.whitening_apply(_in(data), _in(mean), _in(std)).numpy()


def whitening_transform(data, mean=None, std=None):
    """``(whitened, mean, std)``; measures the statistics if not given."""
    moved("grilly.functional.learning.whitening_transform", "grilly.learning.whitening_transform")
    out, mean_t, std_t = _learning.whitening_transform(_in(data), _in(mean), _in(std))
    return out.numpy(), mean_t.numpy(), std_t.numpy()
