"""``torch.nn.init``: in-place initializers, run under no_grad. Same
distributions as torch, drawn from grilly's RNG (so not torch's values)."""

from __future__ import annotations

import math

from .. import no_grad

__all__ = [
    "calculate_gain", "constant_", "kaiming_normal_", "kaiming_uniform_", "normal_",
    "ones_", "uniform_", "xavier_normal_", "xavier_uniform_", "zeros_",
]


def calculate_gain(nonlinearity: str, param=None) -> float:
    linear = {"linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d",
              "conv_transpose2d", "conv_transpose3d", "sigmoid"}
    if nonlinearity in linear:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + slope ** 2))
    if nonlinearity == "selu":
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _fans(tensor):
    if tensor.dim() < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive = math.prod(tensor.shape[2:]) if tensor.dim() > 2 else 1
    return tensor.shape[1] * receptive, tensor.shape[0] * receptive


def uniform_(tensor, a: float = 0.0, b: float = 1.0, generator=None):
    with no_grad():
        return tensor.uniform_(a, b, generator=generator)


def normal_(tensor, mean: float = 0.0, std: float = 1.0, generator=None):
    with no_grad():
        return tensor.normal_(mean, std, generator=generator)


def constant_(tensor, val: float):
    with no_grad():
        return tensor.fill_(val)


def ones_(tensor):
    return constant_(tensor, 1.0)


def zeros_(tensor):
    return constant_(tensor, 0.0)


def xavier_uniform_(tensor, gain: float = 1.0, generator=None):
    fan_in, fan_out = _fans(tensor)
    a = gain * math.sqrt(6.0 / float(fan_in + fan_out))
    return uniform_(tensor, -a, a, generator)


def xavier_normal_(tensor, gain: float = 1.0, generator=None):
    fan_in, fan_out = _fans(tensor)
    return normal_(tensor, 0.0, gain * math.sqrt(2.0 / float(fan_in + fan_out)), generator)


def _kaiming_std(tensor, a, mode, nonlinearity):
    fan_in, fan_out = _fans(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"Mode {mode} not supported, please use one of fan_in, fan_out")
    return calculate_gain(nonlinearity, a) / math.sqrt(fan)


def kaiming_uniform_(tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu", generator=None):
    if 0 in tensor.shape:
        return tensor
    bound = math.sqrt(3.0) * _kaiming_std(tensor, a, mode, nonlinearity)
    return uniform_(tensor, -bound, bound, generator)


def kaiming_normal_(tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu", generator=None):
    if 0 in tensor.shape:
        return tensor
    return normal_(tensor, 0.0, _kaiming_std(tensor, a, mode, nonlinearity), generator)
