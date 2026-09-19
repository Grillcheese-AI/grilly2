"""Grad mode, like ``torch.autograd.grad_mode``: ``no_grad``, ``enable_grad``
and ``set_grad_enabled`` work as context managers and as decorators. Grad
mode is per thread, as in torch."""

from __future__ import annotations

import functools

from .. import _C

__all__ = ["enable_grad", "is_grad_enabled", "no_grad", "set_grad_enabled"]


def is_grad_enabled() -> bool:
    return _C.is_grad_enabled()


class _GradMode:
    """Sets grad mode for a ``with`` block or a decorated function."""

    _mode: bool

    def __enter__(self):
        self._prev = _C.swap_grad_mode(self._mode)
        return None

    def __exit__(self, *exc) -> None:
        _C.swap_grad_mode(self._prev)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.__class__():
                return func(*args, **kwargs)
        return wrapper

    def clone(self):
        return self.__class__()


class no_grad(_GradMode):
    """Ops inside record no gradients (``torch.no_grad``)."""
    _mode = False


class enable_grad(_GradMode):
    """Ops inside record gradients, even under no_grad (``torch.enable_grad``)."""
    _mode = True


class set_grad_enabled(_GradMode):
    """Sets grad mode immediately, and restores it on leaving a ``with``
    block, like ``torch.set_grad_enabled(mode)``."""

    def __init__(self, mode: bool) -> None:
        self._mode = bool(mode)
        self._prev = _C.swap_grad_mode(self._mode)

    def __enter__(self):
        return None  # already applied in __init__, as in torch

    def __call__(self, func):
        _C.swap_grad_mode(self._prev)  # used as a decorator: undo __init__
        mode = self._mode

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with set_grad_enabled(mode):
                return func(*args, **kwargs)
        return wrapper

    def clone(self):
        return set_grad_enabled(self._mode)


from . import graph  # noqa: E402,F401
