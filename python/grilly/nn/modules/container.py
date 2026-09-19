"""``torch.nn.Sequential``, ``ModuleList`` and ``ModuleDict``."""

from __future__ import annotations

import operator
from collections import OrderedDict

from .module import Module


class Sequential(Module):
    """Modules run in order, each on the previous one's output."""

    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self.add_module(name, module)
        else:
            for i, module in enumerate(args):
                self.add_module(str(i), module)

    def _index(self, idx: int) -> int:
        n = len(self)
        idx = operator.index(idx)
        if not -n <= idx < n:
            raise IndexError(f"index {idx} is out of range")
        return idx % n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        return list(self._modules.values())[self._index(idx)]

    def __setitem__(self, idx, module) -> None:
        key = list(self._modules.keys())[self._index(idx)]
        self.add_module(key, module)

    def __delitem__(self, idx) -> None:
        keys = list(self._modules.keys())
        for key in (keys[idx] if isinstance(idx, slice) else [keys[self._index(idx)]]):
            del self._modules[key]
        # renumber, as torch does
        items = list(self._modules.values())
        self._modules.clear()
        for i, m in enumerate(items):
            self._modules[str(i)] = m

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module) -> "Sequential":
        self.add_module(str(len(self)), module)
        return self

    def forward(self, input):
        for module in self:
            input = module(input)
        return input


class ModuleList(Module):
    """A list of modules, registered as submodules."""

    def __init__(self, modules=None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def _index(self, idx: int) -> int:
        n = len(self)
        idx = operator.index(idx)
        if not -n <= idx < n:
            raise IndexError(f"index {idx} is out of range")
        return idx % n

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        return self._modules[str(self._index(idx))]

    def __setitem__(self, idx, module) -> None:
        self.add_module(str(self._index(idx)), module)

    def __delitem__(self, idx) -> None:
        items = list(self._modules.values())
        del items[idx if isinstance(idx, slice) else self._index(idx)]
        self._modules.clear()
        for i, m in enumerate(items):
            self._modules[str(i)] = m

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules) -> "ModuleList":
        return self.extend(modules)

    def append(self, module) -> "ModuleList":
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules) -> "ModuleList":
        for m in modules:
            self.append(m)
        return self

    def insert(self, index: int, module) -> None:
        items = list(self._modules.values())
        items.insert(index, module)
        self._modules.clear()
        for i, m in enumerate(items):
            self.add_module(str(i), m)


class ModuleDict(Module):
    """A dict of modules, registered as submodules."""

    def __init__(self, modules=None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str):
        return self._modules[key]

    def __setitem__(self, key: str, module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules) -> None:
        items = modules.items() if hasattr(modules, "items") else modules
        for key, module in items:
            self[key] = module
