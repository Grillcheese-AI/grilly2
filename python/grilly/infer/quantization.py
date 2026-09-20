"""int8 and int4 weight-only quantization, with torchao's API. To port code,
change the import::

    from torchao.quantization import quantize_, Int8WeightOnlyConfig      # torch
    from grilly.infer.quantization import quantize_, Int8WeightOnlyConfig  # grilly

    quantize_(model, Int8WeightOnlyConfig())
    quantize_(model, Int4WeightOnlyConfig())      # group_size 128, half the bytes

``quantize_`` replaces the weight of every ``nn.Linear`` (or of the modules
``filter_fn`` picks) with an ``Int8Tensor``: int8 values with one float32
scale per output row, in persistent device memory. ``F.linear`` then runs it
with the weight dequantized inside the matmul kernel, as torchao's
weight-only path computes it: ``(x @ q.T) * scale + bias``.

The int8 values are exactly torchao's for a float32 model:
- ``scale = max|w| / 127.5`` per row, at least float32 eps;
- ``q = clamp(round(w * (1 / scale)), -128, 127)``, rounding half to even.

``Int4WeightOnlyConfig`` is torchao's group-wise asymmetric scheme, which is
what makes four bits usable — one scale for a whole row would spend them all
on its largest value:
- ``scale = (max - min) / 15`` over each group of ``group_size`` (128);
- ``zero = min + 8 * scale`` (torchao's float zero-point domain);
- ``q = clamp(round((w - min) / scale), 0, 15)``, rounding half to even;
- ``dequantize(q) = (q - 8) * scale + zero``.

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

__all__ = ["Int4Tensor", "Int4WeightOnlyConfig", "Int8Tensor", "Int8WeightOnlyConfig", "PerRow",
           "QuantizedTensor", "load_quantized_", "quantize_"]


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


@dataclass
class Int4WeightOnlyConfig:
    """torchao's ``Int4WeightOnlyConfig``: group-wise, asymmetric, with
    ``group_size`` values sharing a scale and a zero point."""

    group_size: int = 128
    version: int = 2

    def __post_init__(self) -> None:
        if self.version == 1:
            raise ValueError("version 1 of Int4WeightOnlyConfig has been removed, please use version 2")
        # 32 values share a 16-byte vector in the kernel, and a vector must
        # lie in one group
        if not isinstance(self.group_size, int) or self.group_size <= 0 or self.group_size % 32:
            raise ValueError(f"group_size must be a positive multiple of 32, got {self.group_size}")


class QuantizedTensor(Parameter):
    """What the quantized weights share: the original weight's shape and
    dtype, no gradient, and a ``_q`` holding the device data. ``F.linear``
    uses that data directly; every other use sees ``dequantize()``."""

    __slots__ = ("_q",)

    def __new__(cls, *args, **kwargs):
        raise TypeError(f"create a {cls.__name__} with {cls.__name__}.from_hp(weight)")

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
    def nbytes(self) -> int:
        """Device bytes held by the quantized values and their parameters."""
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

    def requires_grad_(self, requires_grad: bool = True) -> "QuantizedTensor":
        if requires_grad:
            raise RuntimeError(f"an {type(self).__name__} weight is frozen: gradients reach the layer's "
                               "input and bias only")
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
            raise RuntimeError(f"an {type(self).__name__} weight has no gradient")

class Int8Tensor(QuantizedTensor):
    """A linear weight quantized by ``Int8WeightOnlyConfig`` (torchao's
    ``Int8Tensor``): int8 values with one float32 scale per output row."""

    __slots__ = ()

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

    @property
    def scale(self) -> Tensor:
        """The per-row scales, ``(out_features,)``."""
        return Tensor._wrap(self._q.scale())

    def __repr__(self) -> str:
        out_features, in_features = self._q.shape
        return f"Int8Tensor(shape=({out_features}, {in_features}), granularity=PerRow(), nbytes={self.nbytes})"


class Int4Tensor(QuantizedTensor):
    """A linear weight quantized by ``Int4WeightOnlyConfig`` (torchao's
    ``Int4Tensor``): 4-bit levels with a float32 scale and zero point per
    group of ``group_size``."""

    __slots__ = ()

    @classmethod
    def from_hp(cls, hp_tensor, group_size: int = 128) -> "Int4Tensor":
        """Quantize a float32 ``(out_features, in_features)`` weight."""
        t = object.__new__(cls)
        t._q = _C.Int4Weight.quantize(_require_tensor("hp_tensor", hp_tensor)._t, int(group_size))
        t._grad = None
        return t

    @classmethod
    def from_parts(cls, levels, scales, zeros, group_size: int = 128) -> "Int4Tensor":
        """An int4 weight from levels and parameters computed elsewhere:
        ``levels`` is ``(out_features, in_features)`` of 0..15, ``scales``
        and ``zeros`` are ``(out_features, groups)``. This is how a
        quantizer that chooses its own levels — GPTQ — installs them."""
        import numpy as np

        from .. import from_numpy

        levels = np.ascontiguousarray(levels, dtype=np.uint8)
        out_features, in_features = levels.shape
        words = -(-in_features // 32) * 4
        padded = np.zeros((out_features, words * 8), dtype=np.uint32)
        padded[:, :in_features] = levels
        nibbles = padded.reshape(out_features, words, 8)
        packed = np.zeros((out_features, words), dtype=np.uint32)
        for j in range(8):
            packed |= nibbles[:, :, j] << (4 * j)
        t = object.__new__(cls)
        t._q = _C.Int4Weight.from_parts(
            from_numpy(packed.view(np.int32))._t,
            from_numpy(np.ascontiguousarray(scales, dtype=np.float32))._t,
            from_numpy(np.ascontiguousarray(zeros, dtype=np.float32))._t,
            in_features, int(group_size))
        t._grad = None
        return t

    @classmethod
    def _cat(cls, parts) -> "Int4Tensor":
        """One weight holding the parts' rows: a fused q/k/v or gate/up
        projection. Each row keeps its groups' scales, so it is exact."""
        t = object.__new__(cls)
        t._q = _C.Int4Weight.cat([p._q for p in parts])
        t._grad = None
        return t

    @property
    def group_size(self) -> int:
        return self._q.group_size

    @property
    def scale(self) -> Tensor:
        """The per-group scales, ``(out_features, groups)``."""
        return Tensor._wrap(self._q.scale())

    @property
    def zero(self) -> Tensor:
        """The per-group zero points, ``(out_features, groups)``."""
        return Tensor._wrap(self._q.zero())

    def __repr__(self) -> str:
        out_features, in_features = self._q.shape
        return (f"Int4Tensor(shape=({out_features}, {in_features}), group_size={self.group_size}, "
                f"nbytes={self.nbytes})")


def _is_linear(mod, *args) -> bool:
    """torchao's default filter: an ``nn.Linear`` whose weight is not
    quantized yet."""
    return isinstance(mod, Linear) and hasattr(mod, "weight") and not isinstance(mod.weight, QuantizedTensor)


def _quantized(weight, config):
    """``weight`` under ``config``: the quantized tensor a filter picked."""
    if isinstance(config, Int8WeightOnlyConfig):
        return Int8Tensor.from_hp(weight, config.granularity)
    if isinstance(config, Int4WeightOnlyConfig):
        return Int4Tensor.from_hp(weight, config.group_size)
    raise _not_yet(f"quantizing with {type(config).__name__} "
                   "(grilly has Int8WeightOnlyConfig and Int4WeightOnlyConfig)")


def load_quantized_(model, filename, config, filter_fn=_is_linear, strict: bool = True) -> set[str]:
    """Load a safetensors checkpoint into ``model``, quantizing each weight as
    it arrives.

        model.load_state_dict(load_file(path), assign=True)   # float32, then
        quantize_(model, Int8WeightOnlyConfig())              # int8: both at once

    holds every float32 weight *and* its int8 copy at the peak. This holds one
    float32 tensor at a time instead. The result is the same tensors, bit for
    bit: the same kernel on the same values.

    ``strict`` reports missing and unexpected names as ``load_state_dict``
    does. A weight the checkpoint does not carry is left as it is, so a tied
    output projection is restored and quantized afterwards, exactly as after
    ``load_state_dict``::

        load_quantized_(model, path, Int8WeightOnlyConfig(), strict=False)
        model.tie_weights()
        quantize_(model, Int8WeightOnlyConfig())  # only what is still float32

    Quantizing it matters: left float32, a 152K-row output projection costs
    more per token than everything the streaming saved (measured 110.7
    tokens/s against 168.4).

    Returns the names the file held.
    """
    from .safetensors import stream_into

    targets = {f"{name}.weight" if name else "weight": module
               for name, module in model.named_modules() if filter_fn(module, name)}

    def quantize(key, value):
        module = targets.get(key)
        return None if module is None else _quantized(value, config)

    return stream_into(model, filename, strict=strict, convert=quantize)


def quantize_(model, config, filter_fn=_is_linear, device=None) -> None:
    """Quantize, in place, the weight of every module of ``model`` (itself
    included) for which ``filter_fn(module, fully_qualified_name)`` is true.
    ``device`` accepts grilly's device (or its ``"cpu"`` alias); there is
    nowhere else to move to."""
    if not isinstance(config, (Int8WeightOnlyConfig, Int4WeightOnlyConfig)):
        raise _not_yet(f"quantize_ with {type(config).__name__} "
                       "(grilly has Int8WeightOnlyConfig and Int4WeightOnlyConfig)")
    if device is not None:
        _resolve_device(device)
    for name, module in list(model.named_modules()):
        if filter_fn(module, name):
            module.weight = _quantized(module.weight, config)
