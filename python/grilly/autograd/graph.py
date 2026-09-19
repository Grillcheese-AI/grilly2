"""``torch.autograd.graph``: hooks on the tensors autograd saves."""

from __future__ import annotations

from .. import Tensor, _C, _require_tensor

__all__ = ["saved_tensors_hooks"]


class saved_tensors_hooks:
    """While active, every tensor an op saves for backward is passed to
    ``pack_hook``, and backward calls ``unpack_hook`` on whatever it returned
    to get the tensor back. Like torch's, hooks nest and the innermost apply."""

    def __init__(self, pack_hook, unpack_hook) -> None:
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        pack, unpack = self.pack_hook, self.unpack_hook
        _C.push_saved_tensor_hooks(
            lambda handle: pack(Tensor._wrap(handle)),
            lambda packed: _require_tensor("unpack_hook result", unpack(packed))._t,
        )

    def __exit__(self, *exc) -> None:
        _C.pop_saved_tensor_hooks()
