"""grilly.optim, like torch.optim: ``SGD``, ``Adam``, ``AdamW``, with torch's
update rules (the single-tensor, non-fused variants)."""

from __future__ import annotations

import math

from .. import Tensor, _not_yet, no_grad, zeros_like

__all__ = ["Adam", "AdamW", "Optimizer", "SGD"]


class Optimizer:
    """Base class: parameter groups, per-parameter state and ``zero_grad``."""

    def __init__(self, params, defaults: dict) -> None:
        self.defaults = defaults
        self.state: dict = {}
        self.param_groups: list = []
        if isinstance(params, Tensor):
            raise TypeError("params argument given to the optimizer should be an iterable of Tensors or dicts, "
                            f"but got {type(params).__name__}")
        groups = list(params)
        if not groups:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        for g in groups:
            self.add_param_group(g)

    def add_param_group(self, param_group: dict) -> None:
        group = dict(param_group)
        params = group["params"]
        group["params"] = [params] if isinstance(params, Tensor) else list(params)
        for key, value in self.defaults.items():
            group.setdefault(key, value)
        for p in group["params"]:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, but one of the params is {type(p).__name__}")
            if not p.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")
        self.param_groups.append(group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self, closure=None):
        raise NotImplementedError

    def _closure(self, closure):
        if closure is None:
            return None
        from .. import enable_grad
        with enable_grad():
            return closure()

    def state_dict(self):
        raise _not_yet("Optimizer.state_dict")

    def load_state_dict(self, state_dict):
        raise _not_yet("Optimizer.load_state_dict")

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__} ("]
        for i, group in enumerate(self.param_groups):
            lines.append(f"Parameter Group {i}")
            lines += [f"    {k}: {v}" for k, v in sorted(group.items()) if k != "params"]
        return "\n".join(lines) + "\n)"


class SGD(Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0, dampening: float = 0,
                 weight_decay: float = 0, nesterov: bool = False, *, maximize: bool = False,
                 foreach=None, differentiable: bool = False, fused=None) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening,
                                      weight_decay=weight_decay, nesterov=nesterov, maximize=maximize))

    def step(self, closure=None):
        loss = self._closure(closure)
        with no_grad():
            for group in self.param_groups:
                lr, mom, damp, wd = group["lr"], group["momentum"], group["dampening"], group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = -p.grad if group["maximize"] else p.grad
                    if wd != 0:
                        g = g + wd * p
                    if mom != 0:
                        state = self.state.setdefault(p, {})
                        buf = state.get("momentum_buffer")
                        buf = g.clone() if buf is None else buf * mom + (1 - damp) * g
                        state["momentum_buffer"] = buf
                        g = g + mom * buf if group["nesterov"] else buf
                    p.copy_(p - lr * g)
        return loss


class Adam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0, amsgrad: bool = False, *, foreach=None, maximize: bool = False,
                 capturable: bool = False, differentiable: bool = False, fused=None,
                 decoupled_weight_decay: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                                      amsgrad=amsgrad, maximize=maximize,
                                      decoupled_weight_decay=decoupled_weight_decay))

    def step(self, closure=None):
        loss = self._closure(closure)
        with no_grad():
            for group in self.param_groups:
                lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = -p.grad if group["maximize"] else p.grad
                    state = self.state.setdefault(p, {})
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = zeros_like(p)
                        state["exp_avg_sq"] = zeros_like(p)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    value = p
                    if wd != 0:
                        if group["decoupled_weight_decay"]:
                            value = p * (1 - lr * wd)
                        else:
                            g = g + wd * p
                    m = state["exp_avg"] * b1 + g * (1 - b1)
                    v = state["exp_avg_sq"] * b2 + (g * g) * (1 - b2)
                    state["exp_avg"], state["exp_avg_sq"] = m, v
                    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                    if group["amsgrad"]:
                        from .. import maximum
                        state["max_exp_avg_sq"] = maximum(state["max_exp_avg_sq"], v)
                        v = state["max_exp_avg_sq"]
                    denom = v.sqrt() / math.sqrt(bc2) + eps
                    p.copy_(value - (lr / bc1) * (m / denom))
        return loss


class AdamW(Adam):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, amsgrad: bool = False, *, maximize: bool = False,
                 foreach=None, capturable: bool = False, differentiable: bool = False, fused=None) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, maximize=maximize,
                         decoupled_weight_decay=True)
