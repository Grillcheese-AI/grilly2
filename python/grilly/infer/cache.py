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

The fill level is a **device** value — a cursor the append and decode
shaders read — so a decode step over the cache can be recorded once and
replayed (``grilly.graphed``, docs/capture.md). Were it pushed to the
shaders as a constant, a replay would write the same slot and attend over
the same positions, and since the cache's shapes never change with the
length, nothing downstream would notice.

Not v1's cache: that one moves keys and values through host arrays on every
call and stores them lossily (docs/inference.md).

:class:`PaddedKVCache` is the cache for a batch whose prompts differ in
length (left-padded) or whose finished rows should be dropped: plain
tensors, masked reads, and ``select_rows``.
"""

from __future__ import annotations

import numpy as np

from .. import Tensor, _C, _require_tensor, arange, tensor, zeros

__all__ = ["KVCache", "PaddedKVCache"]


class KVCache:
    def __init__(self, num_layers: int, batch_size: int, num_kv_heads: int, head_dim: int, max_length: int) -> None:
        self._c = _C.KVCache(int(num_layers), int(batch_size), int(num_kv_heads), int(head_dim), int(max_length))

    def append(self, layer: int, keys, values) -> None:
        """Write ``(batch, kv_heads, T, head_dim)`` keys and values at the
        layer's next ``T`` positions."""
        self._c.append(int(layer), _require_tensor("keys", keys)._t, _require_tensor("values", values)._t)

    def append_rotary(self, layer: int, keys, k_col: int, values, v_col: int, cos, sin, pos0: int) -> None:
        """Rotate a step's keys into the cache and copy its values, taking
        both from projections in their own ``(batch, T, row)`` layout: no
        transpose or slice in between. Two dispatches."""
        self._c.append_rotary(int(layer), _require_tensor("keys", keys)._t, int(k_col),
                              _require_tensor("values", values)._t, int(v_col),
                              _require_tensor("cos", cos)._t, _require_tensor("sin", sin)._t, int(pos0))

    def attend(self, layer: int, query, scale: float | None = None) -> Tensor:
        """``softmax(query @ K.T * scale) @ V`` over the layer's filled
        positions; ``query`` is ``(batch, heads, 1, head_dim)``. ``scale``
        defaults to ``1 / sqrt(head_dim)``."""
        return Tensor._wrap(self._c.attend(int(layer), _require_tensor("query", query)._t, scale))

    def get_seq_length(self, layer: int = 0) -> int:
        """Positions filled in ``layer`` (Hugging Face's name), as the host
        last saw it. Replaying a captured step advances the device cursor
        without running any host code, so after replays this lags; use
        ``device_len`` for the number the shaders are acting on."""
        return self._c.len(int(layer))

    def device_len(self, layer: int = 0) -> int:
        """Positions filled in ``layer``, read back from the device cursor.
        Costs a submit; ``get_seq_length`` is the cheap answer whenever the
        host is the one driving the cache."""
        return self._c.device_len(int(layer))

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


class PaddedKVCache:
    """A cache for a batch of **left-padded** prompts, which can also drop
    rows as their streams finish.

        cache = PaddedKVCache(num_layers, batch_size, num_kv_heads, head_dim, max_length, keep)

    ``keep`` is the prompt's ``attention_mask``, ``(batch, prompt_length)``:
    0 on each row's left padding, 1 on its tokens (transformers'
    convention). Every row is written at the same positions, so a row with
    ``p`` pads holds its tokens at ``p, p+1, ...`` rather than ``0, 1, ...``.
    That shift changes no attention score, because a rotary embedding's
    ``q . k`` depends only on the distance between the two positions; with
    the padding masked out of every key read, each row attends exactly as
    it would alone (up to rounding).

    What :class:`KVCache` cannot do and this can:

    - mask each row's padding (``prompt_mask``, ``decode_mask``,
      ``pad_columns``);
    - **drop rows** (``select_rows``), so a batch stops paying for streams
      that have finished. The C++ cache has one fixed batch.

    It is plain grilly tensors, written in place with ``index_copy_``, so a
    decode step costs a masked attention per layer where the C++ cache
    costs one dispatch, and it is not built for capture. ``generate`` uses
    it only for batches that need it.
    """

    #: what the attention layers check to take the masked path
    padded = True

    def __init__(self, num_layers: int, batch_size: int, num_kv_heads: int, head_dim: int, max_length: int,
                 keep) -> None:
        keep = np.asarray(keep)
        if keep.ndim != 2 or keep.shape[0] != batch_size or keep.shape[1] > max_length:
            raise ValueError(f"keep must be (batch_size={batch_size}, prompt_length <= {max_length}), "
                             f"got {keep.shape}")
        self._keep = np.ones((batch_size, max_length), dtype=bool)
        self._keep[:, :keep.shape[1]] = keep != 0
        shape = (batch_size, num_kv_heads, max_length, head_dim)
        self.keys = [zeros(shape) for _ in range(num_layers)]
        self.values = [zeros(shape) for _ in range(num_layers)]
        self._len = [0] * num_layers
        self._max_len = int(max_length)
        self._masks: dict = {}  # kind -> (argument, tensor): the layers of one step share one mask

    def append(self, layer: int, keys, values) -> None:
        """Write ``(batch, kv_heads, T, head_dim)`` keys and values at the
        layer's next ``T`` positions, in place."""
        keys, values = _require_tensor("keys", keys), _require_tensor("values", values)
        start, t = self._len[layer], keys.shape[2]
        if start + t > self._max_len:
            raise RuntimeError(f"the cache holds {self._max_len} positions; {start} are filled and {t} more came")
        where = arange(start, start + t)
        self.keys[layer].index_copy_(2, where, keys.contiguous())
        self.values[layer].index_copy_(2, where, values.contiguous())
        self._len[layer] = start + t

    def layer(self, layer: int) -> tuple:
        """The layer's whole ``(batch, kv_heads, max_length, head_dim)``
        key and value buffers, filled or not; ``decode_mask`` says which
        positions to read."""
        return self.keys[layer], self.values[layer]

    def _mask(self, kind: str, arg, build) -> Tensor:
        held = self._masks.get(kind)
        if held is None or held[0] != arg:
            held = self._masks[kind] = (arg, build())
        return held[1]

    def prompt_mask(self, length: int) -> Tensor:
        """``(batch, 1, 1, length)`` bool: the prompt keys each row may read,
        which is its tokens and not its padding. Attention adds the causal
        triangle (``is_causal=True``)."""
        return self._mask("prompt", length, lambda: tensor(np.ascontiguousarray(self._keep[:, None, None, :length])))

    def decode_mask(self, layer: int) -> Tensor:
        """``(batch, 1, 1, max_length)`` bool: the layer's filled positions,
        minus each row's padding. One mask serves every layer of a step."""
        n = self._len[layer]

        def build():
            m = self._keep.copy()
            m[:, n:] = False
            return tensor(np.ascontiguousarray(m[:, None, None, :]))

        return self._mask("decode", n, build)

    def pad_columns(self, length: int):
        """``(batch, 1, length)`` float32, 0 on padding and 1 elsewhere, for
        a layer that mixes positions without attention (LFM2's short
        convolution) to zero its padding with. None when none of the first
        ``length`` positions is padding."""
        part = self._keep[:, :length]
        if part.all():
            return None
        return self._mask("columns", length, lambda: tensor(np.ascontiguousarray(part[:, None, :], dtype=np.float32)))

    def select_rows(self, index) -> None:
        """Keep only batch rows ``index`` (an int64 tensor), in that order.
        The buffers are reallocated at the smaller batch, so the dropped
        rows' memory is freed."""
        index = _require_tensor("index", index)
        rows = index.tolist()
        self.keys = [k.index_select(0, index) for k in self.keys]
        self.values = [v.index_select(0, index) for v in self.values]
        self._keep = self._keep[rows]
        self._masks.clear()

    def get_seq_length(self, layer: int = 0) -> int:
        """Positions filled in ``layer``, padding included."""
        return self._len[layer]

    def reset(self) -> None:
        """Forget every position; the buffers stay allocated. The padding
        stays as given."""
        self._len = [0] * len(self._len)
        self._masks.clear()

    @property
    def batch_size(self) -> int:
        return self._keep.shape[0]

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def max_length(self) -> int:
        return self._max_len

    @property
    def nbytes(self) -> int:
        """Device bytes held by the keys and values."""
        return sum(4 * k.numel() for k in self.keys + self.values)
