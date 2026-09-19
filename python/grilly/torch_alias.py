"""Make ``import torch`` resolve to grilly.

As a pytest plugin, this runs an existing torch test suite on grilly
without editing it::

    python -m pytest -p grilly.torch_alias

or call :func:`install` before anything imports torch. Every grilly module
``grilly.X`` becomes importable as ``torch.X``; a torch module grilly does not
have fails to import, and a missing attribute raises AttributeError, so gaps
show up as test failures rather than silently using real torch.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


def install() -> None:
    if "torch" in sys.modules and sys.modules["torch"].__name__ != "grilly":
        raise RuntimeError("torch was imported before grilly.torch_alias.install()")
    import grilly

    modules = {"grilly": grilly}
    for info in pkgutil.walk_packages(grilly.__path__, prefix="grilly."):
        if info.name.startswith("grilly._") or info.name == __name__:
            continue
        modules[info.name] = importlib.import_module(info.name)
    for name, module in modules.items():
        sys.modules["torch" + name[len("grilly"):]] = module


install()
