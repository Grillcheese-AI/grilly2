"""A key/value cache that stays on the GPU, for decoding.

    cache = KVCache(num_layers, batch_size, num_kv_heads, head_dim, max_length)
    cache.append(layer, keys, values)   # (batch, kv_heads, T, head_dim), in place
    out = cache.attend(layer, query)    # query (batch, heads, 1, head_dim)

Keys and values are exact float32, preallocated per layer in persistent
device memory. ``append`` writes the new positions in place. ``attend`` runs
one query token against the filled positions. ``heads`` may be any multiple
of ``kv_heads`` (grouped-query attention), handled by index, so the cache
is never copied or read back to the host. A prompt's own attention (prefill)
is ordinary causal ``scaled_dot_product_attention`` on its keys and values;
append them after it.

Not v1's cache: that one moves keys and values through host arrays on every
call and stores them lossily (docs/inference.md).
"""

from __future__ import annotations

from .. import Tensor, _C, _require_tensor

__all__ = ["KVCache"]


class KVCache:
    def __init__(self, num_layers: int, batch_size: int, num_kv_heads: int, head_dim: int, max_length: int) -> None:
        self._c = _C.KVCache(int(num_layers), int(batch_size), int(num_kv_heads), int(head_dim), int(max_length))

    def append(self, layer: int, keys, values) -> None:
        """Write ``(batch, kv_heads, T, head_dim)`` keys and values at the
        layer's next ``T`` positions."""
        self._c.append(int(layer), _require_tensor("keys", keys)._t, _require_tensor("values", values)._t)

    def attend(self, layer: int, query, scale: float | None = None) -> Tensor:
        """``softmax(query @ K.T * scale) @ V`` over the layer's filled
        positions; ``query`` is ``(batch, heads, 1, head_dim)``. ``scale``
        defaults to ``1 / sqrt(head_dim)``."""
        return Tensor._wrap(self._c.attend(int(layer), _require_tensor("query", query)._t, scale))

    def get_seq_length(self, layer: int = 0) -> int:
        """Positions filled in ``layer`` (Hugging Face's name)."""
        return self._c.len(int(layer))

    def reset(self) -> None:
        """Forget every position; the memory stays allocated."""
        self._c.reset()

    @property
    def num_layers(self) -> int:
        return self._c.layers

    @property
    def max_length(self) -> int:
        return self._c.max_len

    @property
    def nbytes(self) -> int:
        """Device bytes held by the keys and values."""
        return self._c.nbytes
