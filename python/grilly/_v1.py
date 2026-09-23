"""Helpers for the v1 compatibility shims.

grilly2 replaces grilly v1 in place: the same distribution name, the same
import name. Code written against v1 keeps working through aliases at v1's
old paths, and every one of them says where the thing lives now.

There are two kinds of v1 name, and they are treated differently on
purpose:

- **ported** — a domain library that grilly2 kept and rewrote on its own
  kernels: the VSA algebras, and the rest of §4's list as each lands. The
  alias keeps v1's signature exactly, numpy in and numpy out where v1 took
  numpy, and warns once;
- **replaced** — v1's parallel framework: its own autograd, its own nn
  layers and optimizers, `VulkanTensor`, `torch_api`, and the raw bridge
  ops underneath them. grilly2 does not reimplement these, because the
  torch API is the replacement and that is the entire point of the rewrite
  (CLAUDE.md §1). These **raise**, naming what to use instead.

A replaced name raises `NotImplementedError`, not `AttributeError`,
deliberately: `AttributeError` is what `hasattr` swallows, and a capability
check that quietly answers "no" is how a caller ends up on a slow path
without knowing. v1's own `_detect_backend` did exactly that. Failing loud
is the whole difference between a migration and a silent regression.
"""

from __future__ import annotations

import warnings

__all__ = ["moved", "replaced", "v1_module_getattr"]


def moved(old: str, new: str) -> None:
    """Warn that `old` is now `new`. Called by an alias, once per call
    site, so a big codebase gets one line per place that needs editing."""
    warnings.warn(
        f"{old} is grilly v1's name for {new}. The alias keeps v1's signature and will be "
        f"removed; move to {new}.",
        DeprecationWarning,
        stacklevel=3,
    )


def replaced(old: str, instead: str):
    """Raise for a v1 name grilly2 deliberately does not provide."""
    raise NotImplementedError(
        f"{old} is grilly v1's parallel framework, which grilly2 replaced with the torch API "
        f"rather than reimplementing (CLAUDE.md §1). Use {instead}. This raises rather than "
        f"returning None or missing quietly, so a capability check cannot mistake it for absence."
    )


def v1_module_getattr(module: str, replacements: dict):
    """A module-level ``__getattr__`` that raises for known v1 names.

    `replacements` maps a v1 attribute to what to use instead. Anything else
    gets an ordinary `AttributeError`, and dunder lookups are left alone so
    the import machinery and `inspect` still work.
    """

    def __getattr__(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        instead = replacements.get(name)
        if instead is not None:
            replaced(f"{module}.{name}", instead)
        raise AttributeError(
            f"module {module!r} has no attribute {name!r}. It is grilly v1's module, kept for "
            f"imports that have not moved yet; only the parts grilly2 ported are here "
            f"(see docs/v1-migration.md)."
        )

    return __getattr__
