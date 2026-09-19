"""Load safetensors checkpoints onto the GPU.

This module has the API of ``safetensors.torch``: ``load_file``, ``load``,
``load_model``, ``safe_open`` and ``SafetensorError``. To port code, change the import::

    from safetensors.torch import load_file          # torch
    from grilly.infer.safetensors import load_file   # grilly

``safetensors.torch`` itself cannot run on grilly: importing it needs torch's
``bfloat16``, ``uint8`` and other dtypes grilly does not have.

The file is memory-mapped, and each tensor is uploaded to grilly's device
when it is read. F32, I64, I32 and BOOL tensors load unchanged. BF16 and F16
tensors load as float32: every value is exact, but ``dtype`` reports
``float32``, because grilly has no half-precision dtypes yet. Any other
dtype raises NotImplementedError, naming the tensor.
"""

from __future__ import annotations

import os

from .. import Tensor, _C, _not_yet, _resolve_device

__all__ = ["SafetensorError", "load", "load_file", "load_model", "safe_open", "stream_into"]

SafetensorError = _C.SafetensorError

# the framework names safetensors accepts for torch
_TORCH = ("pt", "torch", "pytorch")


class safe_open:
    """``safe_open(filename, framework, device="cpu")``, as in safetensors.

    Opening the file reads its header only. ``get_tensor(name)`` uploads one
    tensor. Use it as a context manager; the file is released on exit.
    ``framework`` must name torch (``"pt"``), since grilly returns grilly
    tensors. ``device`` accepts what grilly's factories accept: ``"cpu"`` is
    an alias for grilly's one device.
    """

    def __init__(self, filename, framework, device="cpu") -> None:
        if framework not in _TORCH:
            raise _not_yet(f"safe_open with framework={framework!r} (grilly returns grilly tensors; use 'pt')")
        _resolve_device(device)
        self._f = _C.SafeFile.open(os.fspath(filename))

    def __enter__(self) -> "safe_open":
        return self

    def __exit__(self, *exc) -> None:
        self._f = None

    def _file(self):
        if self._f is None:
            raise SafetensorError("File is closed")
        return self._f

    def keys(self) -> list[str]:
        """Tensor names, sorted."""
        return self._file().keys()

    def offset_keys(self) -> list[str]:
        """Tensor names in the order their data appears in the file."""
        return self._file().offset_keys()

    def metadata(self) -> dict[str, str] | None:
        """The header's ``__metadata__`` strings, or None."""
        return self._file().metadata()

    def get_tensor(self, name: str) -> Tensor:
        return Tensor._wrap(self._file().get(name))

    def get_slice(self, name: str):
        raise _not_yet("safe_open.get_slice (load the tensor with get_tensor and index it)")


def load_file(filename, device="cpu") -> dict[str, Tensor]:
    """Every tensor in the file, in file order, uploaded to the device."""
    with safe_open(filename, framework="pt", device=device) as f:
        return {k: f.get_tensor(k) for k in f.offset_keys()}


def stream_into(model, filename, strict: bool = True, convert=None) -> set[str]:
    """Load ``filename`` into ``model`` one tensor at a time, and return the
    names it held.

    ``convert(key, tensor)`` may return what to set in the tensor's place —
    a quantized weight, say — or None to assign the tensor as it is. Nothing
    keeps a reference to a tensor after its key is done, so the file's
    tensors are never all alive at once, unlike
    ``load_state_dict(load_file(path), assign=True)``.
    """
    # shapes only: holding the state dict itself would keep every weight the
    # loop is replacing alive until it finished, which is the whole point
    own = {k: tuple(v.shape) for k, v in model.state_dict(keep_vars=True).items()}
    unexpected, errors = [], []
    with safe_open(filename, framework="pt") as f:
        names = f.offset_keys()
        for key in names:
            shape = own.get(key)
            if shape is None:
                unexpected.append(key)
                continue
            value = f.get_tensor(key)
            if tuple(value.shape) != shape:
                errors.append(f"size mismatch for {key}: copying a param with shape {list(value.shape)} from "
                              f"checkpoint, the shape in current model is {list(shape)}.")
                continue
            replacement = None if convert is None else convert(key, value)
            if replacement is None:
                model.load_state_dict({key: value}, assign=True, strict=False)
            else:
                owner, _, attr = key.rpartition(".")
                setattr(model.get_submodule(owner) if owner else model, attr, replacement)
            del value  # its device memory goes back before the next tensor is read
    if strict:
        missing = [k for k in own if k not in set(names)]
        if missing:
            errors.insert(0, "Missing key(s) in state_dict: " + ", ".join(f'"{k}"' for k in missing))
        if unexpected:
            errors.insert(0, "Unexpected key(s) in state_dict: " + ", ".join(f'"{k}"' for k in unexpected))
    if errors:
        joined = "\n\t".join(errors)
        raise RuntimeError(f"Error(s) in loading state_dict for {type(model).__name__}:\n\t{joined}")
    return set(names)


def load_model(model, filename, strict: bool = True, device="cpu") -> tuple[list[str], list[str]]:
    """``safetensors.torch.load_model``: load a checkpoint into a model.

    One tensor is on the device at a time, so this needs about the model's
    own size rather than twice it. Returns ``(missing, unexpected)`` as
    safetensors does when ``strict`` is False.
    """
    _resolve_device(device)
    loaded = stream_into(model, filename, strict=strict)
    return [k for k in model.state_dict() if k not in loaded], []


def load(data: bytes) -> dict[str, Tensor]:
    """Every tensor in a safetensors file held in ``data``, in file order."""
    f = _C.SafeFile.from_bytes(data)
    return {k: Tensor._wrap(f.get(k)) for k in f.offset_keys()}
