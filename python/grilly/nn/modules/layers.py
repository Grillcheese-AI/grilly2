"""Layers: ``Linear``, ``Identity``, ``LayerNorm``, activations, ``Embedding``."""

from __future__ import annotations

import math

from ... import _parameter_storage
from .. import functional as F
from .. import init
from ..parameter import Parameter
from .module import Module


def _no_device(device, dtype):
    """grilly has one real device; ``device="meta"`` (or a ``with
    device("meta")``) builds parameters without storage."""
    from ... import _check_factory_args, _code, _is_meta
    if _is_meta(device):
        if dtype is not None:
            _code(dtype)
        return
    _check_factory_args(dtype, device)


class Identity(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input):
        return input


class Linear(Module):
    """``y = x W^T + b``; weight (out_features, in_features). Initialized like
    torch (kaiming-uniform, bound 1/sqrt(in_features)) from grilly's RNG, so
    the values are grilly's, not torch's, for a given seed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        _no_device(device, dtype)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(_parameter_storage(out_features, in_features, device=device))
        if bias:
            self.bias = Parameter(_parameter_storage(out_features, device=device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class RMSNorm(Module):
    """``torch.nn.RMSNorm``: ``x * rsqrt(mean(x^2) + eps) * weight`` over the
    last dimensions."""

    def __init__(self, normalized_shape, eps: float | None = None, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        _no_device(device, dtype)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(_parameter_storage(*self.normalized_shape, fill=1.0, device=device))
        else:
            self.register_parameter("weight", None)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        return F.rms_norm(input, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        _no_device(device, dtype)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(_parameter_storage(*self.normalized_shape, fill=1.0, device=device))
            self.bias = Parameter(_parameter_storage(*self.normalized_shape, fill=0.0, device=device)) if bias else None
            if not bias:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class _Activation(Module):
    _fn = None

    def forward(self, input):
        return type(self)._fn(input)


class ReLU(_Activation):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        if inplace:
            from ... import _not_yet
            raise _not_yet("ReLU(inplace=True)")

    _fn = staticmethod(F.relu)


class Tanh(_Activation):
    _fn = staticmethod(F.tanh)


class Sigmoid(_Activation):
    _fn = staticmethod(F.sigmoid)


class SiLU(_Activation):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    _fn = staticmethod(F.silu)


class GELU(Module):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class Softmax(Module):
    def __init__(self, dim=None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmax(input, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Embedding(Module):
    """A lookup table of ``num_embeddings`` rows, initialized N(0, 1) as in
    torch (from grilly's RNG); the padding row starts at zero and gets no
    gradient."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, max_norm=None,
                 norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False,
                 _weight=None, _freeze: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        _no_device(device, dtype)
        if max_norm is not None or scale_grad_by_freq or sparse:
            from ... import _not_yet
            raise _not_yet("Embedding(max_norm / scale_grad_by_freq / sparse)")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        if _weight is None:
            self.weight = Parameter(_parameter_storage(num_embeddings, embedding_dim, device=device), requires_grad=not _freeze)
            init.normal_(self.weight)
            if padding_idx is not None:
                from ... import no_grad
                with no_grad():
                    self.weight[padding_idx] = 0.0
        else:
            self.weight = Parameter(_weight, requires_grad=not _freeze)

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"
