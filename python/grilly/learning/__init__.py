"""Learning rules that are not gradient descent.

Two groups, kept in one package because that is where a caller looks:

- the classic ones v1 carried — elastic weight consolidation, NLMS,
  whitening — in :mod:`grilly.learning._classic`, re-exported here;
- :mod:`grilly.learning.es`, evolution strategies: a forward-only
  optimizer for the case where there is no gradient to take, because the
  objective runs outside the model (a verifier, a simulator, a human).
"""

from ._classic import (  # noqa: F401
    ewc_penalty,
    fisher_info,
    fisher_normalize,
    natural_gradient,
    nlms_ensemble,
    nlms_metrics,
    nlms_predict,
    nlms_update,
    whitening_apply,
    whitening_batch_stats,
    whitening_transform,
)
from .es import EvolutionStrategy, population_linear, population_lora  # noqa: F401

__all__ = [
    "EvolutionStrategy",
    "ewc_penalty",
    "fisher_info",
    "fisher_normalize",
    "natural_gradient",
    "nlms_ensemble",
    "nlms_metrics",
    "nlms_predict",
    "nlms_update",
    "population_linear",
    "population_lora",
    "whitening_apply",
    "whitening_batch_stats",
    "whitening_transform",
]
