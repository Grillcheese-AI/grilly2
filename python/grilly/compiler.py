"""``torch.compiler``: introspection and controls for grilly's compiled
callables (docs/fusion.md). Hugging Face and other libraries probe
``is_compiling()`` and use ``disable()`` to opt code out of capture."""

from __future__ import annotations

import functools

from . import _lazy

__all__ = ["disable", "is_compiling", "reset"]


def is_compiling() -> bool:
    """True only while inside a callable made by ``grilly.compile``
    (``torch.compiler.is_compiling``)."""
    return _lazy.is_compiling()


def disable(fn=None, recursive: bool = True):
    """Run ``fn`` (or the decorated function) without lazy capture, even
    inside a compiled callable (``torch.compiler.disable``).

    ``recursive`` is accepted for torch's signature. grilly has no
    bytecode-level tracer to scope capture to a single frame, so suspending
    capture here already covers every call made while ``fn`` runs,
    including nested ones — torch's ``recursive=True`` behaviour."""

    def apply(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            depth = _lazy.capture_depth()
            for _ in range(depth):
                _lazy.pop_capture()
            try:
                return f(*args, **kwargs)
            finally:
                for _ in range(depth):
                    _lazy.push_capture()
        return wrapper

    return apply if fn is None else apply(fn)


def reset() -> None:
    """Clear grilly's Python-side compile caches (``torch.compiler.reset``).
    The generated-kernel cache is per Vulkan device and lives for the
    process (docs/fusion.md: "Kernels are cached per device by their
    generated source"), so there is nothing there to drop; this resets the
    stats hook used by tests."""
    _lazy.reset_stats()


def _stats() -> dict:
    """For tests: ``{"materializations": int, "node_counts": [int, ...]}``
    since the last :func:`reset`."""
    return _lazy.stats()
