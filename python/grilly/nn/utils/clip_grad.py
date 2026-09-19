"""``torch.nn.utils.clip_grad_norm_`` and ``clip_grad_value_``."""

from __future__ import annotations

import math

from ... import Tensor, _not_yet, no_grad, stack, tensor


def _grads(parameters):
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    return [p.grad for p in parameters if p.grad is not None]


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0,
                    error_if_nonfinite: bool = False, foreach=None) -> Tensor:
    """Scale gradients in place so their total norm is at most ``max_norm``;
    returns the total norm before clipping, as torch does."""
    if float(norm_type) != 2.0:
        raise _not_yet(f"clip_grad_norm_(norm_type={norm_type})")
    grads = _grads(parameters)
    if not grads:
        return tensor(0.0)
    with no_grad():
        total = stack([(g * g).sum() for g in grads]).sum().sqrt()
        norm = total.item()
        if error_if_nonfinite and not math.isfinite(norm):
            raise RuntimeError(f"The total norm of order {float(norm_type)} for gradients from "
                               "`parameters` is non-finite, so it cannot be clipped.")
        # torch always multiplies by the clamped coefficient
        coef = min(float(max_norm) / (norm + 1e-6), 1.0)
        for g in grads:
            g.mul_(coef)
    return total


def clip_grad_value_(parameters, clip_value: float, foreach=None) -> None:
    with no_grad():
        for g in _grads(parameters):
            g.copy_(g.clamp(-clip_value, clip_value).contiguous())
