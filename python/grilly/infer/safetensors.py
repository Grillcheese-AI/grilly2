"""Load safetensors checkpoints onto the GPU.

This module has the API of ``safetensors.torch``: ``load_file``, ``load``,
``safe_open`` and ``SafetensorError``. To port code, change the import::

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

__all__ = ["SafetensorError", "load", "load_file", "safe_open"]

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


def load(data: bytes) -> dict[str, Tensor]:
    """Every tensor in a safetensors file held in ``data``, in file order."""
    f = _C.SafeFile.from_bytes(data)
    return {k: Tensor._wrap(f.get(k)) for k in f.offset_keys()}
