"""``torch.utils.checkpoint``: trade compute for memory in backward.

Only the non-reentrant variant (``use_reentrant=False``) exists, and it
follows torch's algorithm: the forward pass records the graph as usual, so
parameters used inside the function (not only its arguments) get gradients.
But every tensor an op saves for backward is replaced by an index. The
first time backward needs one, the function is rerun, with the RNG state
of the original call, to regenerate them.
"""

from __future__ import annotations

from ..autograd import enable_grad, is_grad_enabled
from ..autograd.graph import saved_tensors_hooks
from ..random import default_generator

__all__ = ["CheckpointError", "checkpoint"]


class CheckpointError(RuntimeError):
    """The recomputation did not reproduce the original forward pass."""


class _Frame:
    """One checkpointed call: how to recompute it, and what it recomputed."""

    def __init__(self, function, args, kwargs, rng_state) -> None:
        self.function, self.args, self.kwargs = function, args, kwargs
        self.rng_state = rng_state
        self.num_saved = 0
        self.recomputed = None

    def pack(self, _tensor) -> int:
        index = self.num_saved
        self.num_saved += 1
        return index  # the activation itself is not kept

    def unpack(self, index: int):
        if self.recomputed is None:
            self._recompute()
        return self.recomputed[index]

    def _recompute(self) -> None:
        saved = []

        def keep(t):
            saved.append(t.detach())

        def refuse(_):
            raise CheckpointError("a saved tensor was unpacked during checkpoint recomputation")

        outer = default_generator._g.get_state_pair()
        if self.rng_state is not None:
            default_generator._g.set_state_pair(self.rng_state)
        try:
            with enable_grad(), saved_tensors_hooks(keep, refuse):
                self.function(*self.args, **self.kwargs)
        finally:
            default_generator._g.set_state_pair(outer)
        if len(saved) != self.num_saved:
            raise CheckpointError(
                "torch.utils.checkpoint: A different number of tensors was saved during the "
                f"original forward and recomputation. Number of tensors saved during forward: "
                f"{self.num_saved}. Number of tensors saved during recomputation: {len(saved)}."
            )
        self.recomputed = saved


def checkpoint(function, *args, use_reentrant=None, context_fn=None,
               determinism_check="default", debug=False, preserve_rng_state=True, **kwargs):
    """Run ``function(*args, **kwargs)`` without keeping its activations for
    backward; they are recomputed when backward needs them."""
    if use_reentrant is None or use_reentrant:
        raise NotImplementedError(
            "grilly: checkpoint(use_reentrant=True) is not implemented yet; pass "
            "use_reentrant=False (torch 2.10 treats an omitted use_reentrant as True)"
        )
    if context_fn is not None:
        raise NotImplementedError("grilly: checkpoint(context_fn=...) is not implemented yet")
    if determinism_check != "default" or debug:
        raise NotImplementedError("grilly: checkpoint determinism_check/debug are not implemented yet")
    if not is_grad_enabled():
        return function(*args, **kwargs)  # nothing is saved, so nothing to recompute

    rng_state = default_generator._g.get_state_pair() if preserve_rng_state else None
    frame = _Frame(function, args, kwargs, rng_state)
    with saved_tensors_hooks(frame.pack, frame.unpack):
        return function(*args, **kwargs)
