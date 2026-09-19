"""int8 weight-only quantization, with torchao's API. To port code, change
the import::

    from torchao.quantization import quantize_, Int8WeightOnlyConfig      # torch
    from grilly.infer.quantization import quantize_, Int8WeightOnlyConfig  # grilly

    quantize_(model, Int8WeightOnlyConfig())

``quantize_`` replaces the weight of every ``nn.Linear`` (or of the modules
``filter_fn`` picks) with an ``Int8Tensor``: int8 values with one float32
scale per output row, in persistent device memory. ``F.linear`` then runs it
with the weight dequantized inside the matmul kernel, as torchao's
weight-only path computes it: ``(x @ q.T) * scale + bias``.

The quantized values are exactly torchao's for a float32 model:
- ``scale = max|w| / 127.5`` per row, at least float32 eps;
- ``q = clamp(round(w * (1 / scale)), -128, 127)``, rounding half to even.

The weight is frozen. Gradients still reach the layer's input and bias, so a
quantized model runs with or without ``no_grad``. Any use of the weight
other than ``F.linear`` (printing it, indexing it, a ``state_dict``) sees
the dequantized float32 weight, computed on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import Size, Tensor, _C, _DEVICE, _not_yet, _require_tensor, _resolve_device, float32
from ..nn.modules.layers import Linear
from ..nn.parameter import Parameter

__all__ = ["Int8Tensor", "Int8WeightOnlyConfig", "PerRow", "load_quantized_", "quantize_"]


@dataclass(frozen=True)
class PerRow:
    """One scale per row of the weight, i.e. per output feature
    (torchao.quantization.PerRow)."""

    dim: int = -1


@dataclass
class Int8WeightOnlyConfig:
    """torchao's ``Int8WeightOnlyConfig`` (version 2). Per-row granularity is
    the only one grilly has."""

    group_size: int | None = None
    granularity: object = field(default_factory=PerRow)
    set_inductor_config: bool = True
    version: int = 2

    def __post_init__(self) -> None:
        if self.version == 1:
            raise ValueError("version 1 of Int8WeightOnlyConfig has been removed, please use version 2")
        assert self.group_size is None, (
            f"Only support version 2 with group_size=None, got {self.group_size}. "
            f"Use granularity=PerGroup({self.group_size}) instead."
        )
        if self.granularity != PerRow():
            raise _not_yet(f"Int8WeightOnlyConfig(granularity={self.granularity!r}) (grilly has PerRow())")


class Int8Tensor(Parameter):
    """A linear weight quantized by ``Int8WeightOnlyConfig`` (torchao's
    ``Int8Tensor``). Its shape and dtype are the original weight's, and it
    never requires grad. ``F.linear`` uses the int8 values directly; anything
    else sees ``dequantize()``."""

    __slots__ = ("_q",)

    def __new__(cls, *args, **kwargs):
        raise TypeError("create an Int8Tensor with Int8Tensor.from_hp(weight, PerRow())")

    @classmethod
    def from_hp(cls, hp_tensor, granularity=None) -> "Int8Tensor":
        """Quantize a float32 ``(out_features, in_features)`` weight."""
        if granularity is not None and granularity != PerRow():
            raise _not_yet(f"Int8Tensor.from_hp(granularity={granularity!r}) (grilly has PerRow())")
        t = object.__new__(cls)
        t._q = _C.Int8Weight.quantize(_require_tensor("hp_tensor", hp_tensor)._t)
        t._grad = None
        return t

    @classmethod
    def _cat(cls, parts) -> "Int8Tensor":
        """One weight holding the parts' rows: a fused q/k/v or gate/up
        projection. Per-row scales make it exact."""
        t = object.__new__(cls)
        t._q = _C.Int8Weight.cat([p._q for p in parts])
        t._grad = None
        return t

    def __init__(self, *args, **kwargs) -> None:
        pass

    # every op but F.linear sees the dequantized weight
    @property
    def _t(self):
        return self._q.dequantize()

    def dequantize(self, output_dtype=None) -> Tensor:
        if output_dtype not in (None, float32):
            raise _not_yet(f"dequantize(output_dtype={output_dtype!r})")
        return Tensor._wrap(self._q.dequantize())

    @property
    def scale(self) -> Tensor:
        """The per-row scales, ``(out_features,)``."""
        return Tensor._wrap(self._q.scale())

    @property
    def nbytes(self) -> int:
        """Device bytes held by the int8 values and scales."""
        return self._q.nbytes

    def _grilly_linear(self, input, bias):
        x = _require_tensor("input", input)
        b = None if bias is None else _require_tensor("bias", bias)._t
        return Tensor._wrap(self._q.linear(x._t, b))

    # metadata, without dequantizing
    @property
    def shape(self) -> Size:
        return Size(self._q.shape)

    def dim(self) -> int:
        return 2

    def numel(self) -> int:
        out_features, in_features = self._q.shape
        return out_features * in_features

    @property
    def dtype(self):
        return float32

    @property
    def device(self):
        return _DEVICE

    @property
    def requires_grad(self) -> bool:
        return False

    @requires_grad.setter
    def requires_grad(self, value) -> None:
        self.requires_grad_(value)

    def requires_grad_(self, requires_grad: bool = True) -> "Int8Tensor":
        if requires_grad:
            raise RuntimeError("an Int8Tensor weight is frozen: gradients reach the layer's input and bias only")
        return self

    @property
    def is_leaf(self) -> bool:
        return True

    @property
    def grad_fn(self):
        return None

    @property
    def grad(self):
        return None

    @grad.setter
    def grad(self, value) -> None:
        if value is not None:
            raise RuntimeError("an Int8Tensor weight has no gradient")

    def __repr__(self) -> str:
        out_features, in_features = self._q.shape
        return f"Int8Tensor(shape=({out_features}, {in_features}), granularity=PerRow(), nbytes={self.nbytes})"


def _is_linear(mod, *args) -> bool:
    """torchao's default filter: an ``nn.Linear`` whose weight is not
    quantized yet."""
    return isinstance(mod, Linear) and hasattr(mod, "weight") and not isinstance(mod.weight, Int8Tensor)


def load_quantized_(model, filename, config, filter_fn=_is_linear, strict: bool = True) -> None:
    """Load a safetensors checkpoint into ``model``, quantizing each weight as
    it arrives.

        model.load_state_dict(load_file(path), assign=True)   # float32, then
        quantize_(model, Int8WeightOnlyConfig())              # int8: both at once

    holds every float32 weight *and* its int8 copy at the peak. This holds one
    float32 tensor at a time instead (measured on Qwen2.5-0.5B: 2.98 GiB
    against 1.12). The result is the same tensors, bit for bit.

    ``strict`` reports missing and unexpected names as ``load_state_dict``
    does. A weight the checkpoint does not carry is left as it is, so a tied
    output projection is restored and quantized afterwards, exactly as after
    ``load_state_dict``::

        load_quantized_(model, path, Int8WeightOnlyConfig(), strict=False)
        model.tie_weights()
        quantize_(model, Int8WeightOnlyConfig())  # only what is still float32

    Quantizing it matters: left float32, a 152K-row output projection costs
    more per token than everything the streaming saved (measured 110.7
    tokens/s against 163.4).
    """
    if not isinstance(config, Int8WeightOnlyConfig):
        raise _not_yet(f"load_quantized_ with {type(config).__name__} (grilly has Int8WeightOnlyConfig)")
    from .safetensors import safe_open

    targets = {f"{name}.weight" if name else "weight": module
               for name, module in model.named_modules() if filter_fn(module, name)}
    with safe_open(filename, framework="pt") as f:
        names = f.offset_keys()
        unexpected = [k for k in names if k not in model.state_dict()]
        for key in names:
            value = f.get_tensor(key)
            module = targets.get(key)
            if module is None:
                model.load_state_dict({key: value}, assign=True, strict=False)
            elif tuple(value.shape) != tuple(module.weight.shape):
                raise RuntimeError(f"size mismatch for {key}: copying a param with shape {value.shape} from "
                                   f"checkpoint, the shape in current model is {module.weight.shape}.")
            else:
                module.weight = Int8Tensor.from_hp(value, config.granularity)
            del value  # the float32 tensor's device memory goes back now
    if strict:
        missing = [k for k in model.state_dict() if k not in set(names)]
        errors = []
        if missing:
            errors.append("Missing key(s) in state_dict: " + ", ".join(f'"{k}"' for k in missing))
        if unexpected:
            errors.append("Unexpected key(s) in state_dict: " + ", ".join(f'"{k}"' for k in unexpected))
        if errors:
            joined = "\n\t".join(errors)
            raise RuntimeError(f"Error(s) in loading state_dict for {type(model).__name__}:\n\t{joined}")


def quantize_(model, config, filter_fn=_is_linear, device=None) -> None:
    """Quantize, in place, the weight of every module of ``model`` (itself
    included) for which ``filter_fn(module, fully_qualified_name)`` is true.
    ``device`` accepts grilly's device (or its ``"cpu"`` alias); there is
    nowhere else to move to."""
    if not isinstance(config, Int8WeightOnlyConfig):
        raise _not_yet(f"quantize_ with {type(config).__name__} (grilly has Int8WeightOnlyConfig)")
    if device is not None:
        _resolve_device(device)
    for name, module in list(model.named_modules()):
        if filter_fn(module, name):
            module.weight = Int8Tensor.from_hp(module.weight, config.granularity)
