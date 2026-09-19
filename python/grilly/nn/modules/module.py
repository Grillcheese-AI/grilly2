"""``torch.nn.Module``: attribute-based registration of parameters, buffers
and submodules, and the traversal, state and mode methods built on it."""

from __future__ import annotations

from collections import OrderedDict, namedtuple

from ... import Tensor, _not_yet, _resolve_device, device, dtype, float32, no_grad
from ..parameter import Parameter

_IncompatibleKeys = namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class Module:
    """Base class for all neural network modules, like ``torch.nn.Module``."""

    training: bool

    def __init__(self, *args, **kwargs) -> None:
        if args or kwargs:
            raise TypeError(f"{type(self).__name__}.__init__() takes no arguments")
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_non_persistent_buffers_set", set())
        object.__setattr__(self, "_modules", OrderedDict())

    # ── registration ─────────────────────────────────────────────────────

    def _check_init(self):
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign before Module.__init__() call")

    def register_parameter(self, name: str, param) -> None:
        self._check_init()
        if "." in name or name == "":
            raise KeyError(f'parameter name can\'t contain "." or be empty, got {name!r}')
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param).__name__}' object to parameter '{name}' "
                            "(torch.nn.Parameter or None required)")
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor, persistent: bool = True) -> None:
        self._check_init()
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(f"cannot assign '{type(tensor).__name__}' object to buffer '{name}' "
                            "(torch Tensor or None required)")
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def add_module(self, name: str, module) -> None:
        self._check_init()
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{type(module).__name__} is not a Module subclass")
        self._modules[name] = module

    register_module = add_module

    def __setattr__(self, name: str, value) -> None:
        d = self.__dict__
        params, buffers, modules = d.get("_parameters"), d.get("_buffers"), d.get("_modules")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            d.pop(name, None)
            buffers.pop(name, None)
            modules.pop(name, None)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value).__name__}' as parameter '{name}' "
                                "(torch.nn.Parameter or None expected)")
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError("cannot assign module before Module.__init__() call")
            d.pop(name, None)
            params.pop(name, None)
            buffers.pop(name, None)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value).__name__}' as child module '{name}' "
                                "(torch.nn.Module or None expected)")
            modules[name] = value
        elif buffers is not None and name in buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"cannot assign '{type(value).__name__}' as buffer '{name}' "
                                "(torch.Tensor or None expected)")
            buffers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        d = self.__dict__
        for store in ("_parameters", "_buffers", "_modules"):
            if store in d and name in d[store]:
                return d[store][name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        for store in (self._parameters, self._buffers, self._modules):
            if name in store:
                del store[name]
                self._non_persistent_buffers_set.discard(name)
                return
        object.__delattr__(self, name)

    # ── traversal ────────────────────────────────────────────────────────

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):
        if memo is None:
            memo = set()
        if remove_duplicate and id(self) in memo:
            return
        if remove_duplicate:
            memo.add(id(self))
        yield prefix, self
        for name, module in self._modules.items():
            if module is None:
                continue
            yield from module.named_modules(memo, prefix + ("." if prefix else "") + name, remove_duplicate)

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def named_children(self):
        seen = set()
        for name, module in self._modules.items():
            if module is not None and id(module) not in seen:
                seen.add(id(module))
                yield name, module

    def children(self):
        for _, m in self.named_children():
            yield m

    def _named_members(self, get_members, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            for name, value in get_members(module):
                if value is None or (remove_duplicate and id(value) in seen):
                    continue
                if remove_duplicate:
                    seen.add(id(value))
                yield module_prefix + ("." if module_prefix else "") + name, value

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield from self._named_members(lambda m: m._parameters.items(), prefix, recurse, remove_duplicate)

    def parameters(self, recurse: bool = True):
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield from self._named_members(lambda m: m._buffers.items(), prefix, recurse, remove_duplicate)

    def buffers(self, recurse: bool = True):
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def get_submodule(self, target: str) -> "Module":
        mod = self
        for part in target.split(".") if target else []:
            if part not in mod._modules:
                raise AttributeError(f"{type(mod).__name__} has no attribute `{part}`")
            mod = mod._modules[part]
        return mod

    def apply(self, fn) -> "Module":
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    # ── modes and gradients ──────────────────────────────────────────────

    def train(self, mode: bool = True) -> "Module":
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> "Module":
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters():
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def to(self, *args, **kwargs) -> "Module":
        """Device moves are no-ops (grilly has one device; "cpu" is an alias
        for it); dtype conversion supports float32 only."""
        for t in list(args) + list(kwargs.values()):
            if isinstance(t, Tensor):
                t = t.dtype
            if isinstance(t, dtype):
                if t is not float32:
                    raise _not_yet(f"Module.to({t!r}) (only float32 parameters so far)")
            elif isinstance(t, (str, device, int)) and not isinstance(t, bool):
                _resolve_device(t)
            elif t is not None and not isinstance(t, bool):
                raise TypeError(f"to() received an invalid argument {t!r}")
        return self

    def cpu(self) -> "Module":
        return self

    def cuda(self, *args, **kwargs) -> "Module":
        raise _not_yet("Module.cuda() (grilly runs on Vulkan)")

    def float(self) -> "Module":
        return self

    def compile(self, *args, **kwargs) -> None:
        """Compiles ``self.forward`` in place (``torch.nn.Module.compile``,
        torch >= 2.2): afterwards every ``self(...)`` call runs through
        ``grilly.compile`` (docs/fusion.md). Unlike ``grilly.compile(self)``,
        ``self`` keeps its own type — there is no wrapper object."""
        from ... import compile as _compile
        self._compiled_call = _compile(lambda *a, **kw: self.forward(*a, **kw), *args, **kwargs)

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    def __call__(self, *args, **kwargs):
        hook = self.__dict__.get("_compiled_call")
        return hook(*args, **kwargs) if hook is not None else self.forward(*args, **kwargs)

    # ── state ────────────────────────────────────────────────────────────

    def state_dict(self, *args, destination=None, prefix: str = "", keep_vars: bool = False):
        if destination is None:
            destination = OrderedDict()
        for name, p in self._parameters.items():
            if p is not None:
                destination[prefix + name] = p if keep_vars else p.detach()
        for name, b in self._buffers.items():
            if b is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = b if keep_vars else b.detach()
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Copy ``state_dict`` into this module's parameters and buffers, as
        torch does. With ``assign=True`` the state dict's tensors become the
        parameters and buffers, with no copy; each parameter keeps its
        ``requires_grad``. That is how a checkpoint from
        ``grilly.infer.safetensors.load_file`` is held with no second copy."""
        own = self.state_dict(keep_vars=True)
        missing = [k for k in own if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own]
        errors = []
        if strict and (missing or unexpected):
            if unexpected:
                errors.append("Unexpected key(s) in state_dict: " + ", ".join(f'"{k}"' for k in unexpected) + ". ")
            if missing:
                errors.append("Missing key(s) in state_dict: " + ", ".join(f'"{k}"' for k in missing) + ". ")
        with no_grad():
            for key, target in own.items():
                if key not in state_dict:
                    continue
                value = state_dict[key]
                if tuple(value.shape) != tuple(target.shape):
                    errors.append(f"size mismatch for {key}: copying a param with shape {value.shape} from "
                                  f"checkpoint, the shape in current model is {target.shape}.")
                    continue
                if not assign:
                    target.copy_(value)
                    continue
                owner, _, name = key.rpartition(".")
                module = self.get_submodule(owner) if owner else self
                if isinstance(target, Parameter):
                    if not isinstance(value, Parameter):
                        value = Parameter(value, requires_grad=target.requires_grad)
                    value.requires_grad_(target.requires_grad)
                setattr(module, name, value)
        if errors:
            raise RuntimeError(f"Error(s) in loading state_dict for {type(self).__name__}:\n\t" + "\n\t".join(errors))
        return _IncompatibleKeys(missing, unexpected)

    # ── repr ─────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = []
        extra = self.extra_repr()
        if extra:
            lines = extra.split("\n")
        children = []
        for name, module in self._modules.items():
            mod = repr(module).replace("\n", "\n  ")
            children.append(f"({name}): {mod}")
        lines += children
        main = type(self).__name__ + "("
        if lines:
            if len(lines) == 1 and not children:
                main += lines[0]
            else:
                main += "\n  " + "\n  ".join(lines) + "\n"
        return main + ")"

    def __dir__(self):
        return sorted(set(object.__dir__(self)) | set(self._parameters) | set(self._buffers) | set(self._modules))
