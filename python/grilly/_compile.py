"""``torch.compile`` (docs/fusion.md, "the capture rules"): wraps a function
or an ``nn.Module`` so that, for the duration of a call, float32 elementwise
ops build a lazy fusion graph instead of running one kernel each (see
``grilly._lazy``). Whatever the callable returns is walked and
materialized, so callers only ever see real tensors — there is no graph
tracer or graph-break machinery here, only the elementwise capture.
"""

from __future__ import annotations

import functools

from . import Tensor, _lazy
from .nn import Module

__all__ = ["OptimizedModule", "compile"]

_BACKENDS = ("inductor", "eager", "aot_eager")


def _materialize_tree(value):
    """Real tensors throughout, recursively: a compiled callable's return
    value never exposes a LazyTensor to its caller."""
    if isinstance(value, _lazy.LazyTensor):
        return Tensor._wrap(_lazy.materialize(value))
    if isinstance(value, dict):
        return {k: _materialize_tree(v) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):  # namedtuple
        return type(value)(*(_materialize_tree(v) for v in value))
    if isinstance(value, tuple):
        return tuple(_materialize_tree(v) for v in value)
    if isinstance(value, list):
        return [_materialize_tree(v) for v in value]
    return value


def _wrap(fn, capture: bool):
    @functools.wraps(fn)
    def compiled(*args, **kwargs):
        _lazy.push_compiling()
        if capture:
            _lazy.push_capture()
        try:
            return _materialize_tree(fn(*args, **kwargs))
        finally:
            if capture:
                _lazy.pop_capture()
            _lazy.pop_compiling()

    compiled._grilly_compiled = True
    return compiled


def compile(model=None, *, fullgraph=False, dynamic=None, backend="inductor",  # noqa: A001 - torch's name
            mode=None, options=None, disable=False):
    """Like ``torch.compile``: usable bare (``grilly.compile(fn)``), as a
    decorator with or without arguments, and on a function or an
    ``nn.Module``.

    Only ``backend`` changes behaviour: the default and any name other than
    ``"eager"``/``"aot_eager"`` fuses float32 elementwise chains inside the
    callable into one kernel each (docs/fusion.md); ``"eager"`` and
    ``"aot_eager"`` run the callable unchanged. An unrecognised backend
    raises, as torch's does. ``fullgraph``, ``dynamic``, ``mode`` and
    ``options`` are accepted for torch's calling convention and have no
    effect here.
    """
    if disable:
        return model if model is not None else (lambda fn: fn)
    if not isinstance(backend, str):
        raise NotImplementedError("grilly.compile(backend=<callable>) is not implemented yet; pass a backend name")
    if backend not in _BACKENDS:
        raise RuntimeError(f"Unknown backend: {backend!r}; grilly knows {list(_BACKENDS)}")
    capture = backend not in ("eager", "aot_eager")

    def apply(target):
        if isinstance(target, Module):
            return OptimizedModule(target, _wrap(lambda *a, **kw: target(*a, **kw), capture))
        return _wrap(target, capture)

    return apply if model is None else apply(model)


class OptimizedModule(Module):
    """Wraps a compiled ``nn.Module``: forwards attribute access to it and
    shares its parameters and buffers, like torch's ``OptimizedModule``.

    Unlike torch's, ``state_dict()`` keys are the wrapped module's own, with
    no ``"_orig_mod."`` prefix: that prefix is a well-known drop-in gotcha in
    torch itself (a checkpoint saved from a compiled model won't load into
    an uncompiled one, or vice versa, without renaming keys), and usability
    is the acceptance criterion here (CLAUDE.md §10).
    """

    def __init__(self, mod: Module, compiled_forward) -> None:
        super().__init__()
        object.__setattr__(self, "_orig_mod", mod)
        object.__setattr__(self, "_compiled_forward", compiled_forward)

    def forward(self, *args, **kwargs):
        return self._compiled_forward(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_mod, name)

    def __setattr__(self, name, value) -> None:
        if name in ("_orig_mod", "_compiled_forward"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._orig_mod, name, value)

    def __delattr__(self, name) -> None:
        if name in self.__dict__:
            object.__delattr__(self, name)
        else:
            delattr(self._orig_mod, name)

    # traversal, state and mode delegate to the wrapped module, so
    # parameters are shared and state_dict keys are unchanged.
    def parameters(self, recurse: bool = True):
        return self._orig_mod.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self._orig_mod.named_parameters(prefix, recurse, remove_duplicate)

    def buffers(self, recurse: bool = True):
        return self._orig_mod.buffers(recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self._orig_mod.named_buffers(prefix, recurse, remove_duplicate)

    def children(self):
        return self._orig_mod.children()

    def named_children(self):
        return self._orig_mod.named_children()

    def modules(self):
        return self._orig_mod.modules()

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):
        return self._orig_mod.named_modules(memo, prefix, remove_duplicate)

    def state_dict(self, *args, **kwargs):
        return self._orig_mod.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self._orig_mod.load_state_dict(state_dict, strict, assign)

    def train(self, mode: bool = True) -> "OptimizedModule":
        self._orig_mod.train(mode)
        object.__setattr__(self, "training", mode)
        return self

    def eval(self) -> "OptimizedModule":
        return self.train(False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._orig_mod.zero_grad(set_to_none)

    def __repr__(self) -> str:
        return "OptimizedModule(\n  (_orig_mod): " + repr(self._orig_mod).replace("\n", "\n  ") + "\n)"
