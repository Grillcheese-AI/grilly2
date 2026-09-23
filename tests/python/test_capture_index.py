"""Index ops inside a captured graph, and their bounds checked on the device.

An index op used to ask the host whether its indices were in range. That
cost a readback, which ends the command batch — and it made the op
impossible to capture, because a replay re-issues commands without running
the host and anything the host decided cannot be reproduced. Every model
with an embedding was therefore outside ``grilly.graphed``.

The check now runs on the device and the fault is raised at the next host
read, which is what torch does on CUDA. Two things follow, and both are
tested here:

- **capture: bit-exact.** A replay issues the same commands over the same
  buffers, so the comparison against running the same op eagerly is ``==``.
- **the error still arrives**, with torch's wording, at the next read
  rather than at the call.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly
import grilly.nn.functional as F
from grilly import _C


def gen(seed=0):
    return grilly.Generator().manual_seed(seed)


def captured(fn):
    """Run ``fn`` once eagerly, then captured, then replayed. Returns the
    eager result and the replayed one."""
    ref = fn()
    ref_np = ref.numpy().copy()
    _C.begin_capture()
    try:
        out = fn()
    except BaseException:
        _C.end_capture()
        raise
    graph = _C.end_capture()
    graph.replay()
    return ref_np, out.numpy()


# ── reads ────────────────────────────────────────────────────────────────


def test_every_row_lookup_form_captures_and_replays_bit_exactly():
    """The five spellings a model reaches for. All of them were refused."""
    with grilly.no_grad():
        w = grilly.randn(1000, 64, generator=gen(1))
        emb = grilly.nn.Embedding(1000, 64)
        idx = grilly.tensor([7])
        wide = idx.reshape(1, 1).expand(1, 64)

        for name, fn in [
            ("nn.Embedding", lambda: emb(idx)),
            ("F.embedding", lambda: F.embedding(idx, w)),
            ("index_select", lambda: w.index_select(0, idx)),
            ("x[idx]", lambda: w[idx]),
            ("gather", lambda: grilly.gather(w, 0, wide)),
        ]:
            ref, got = captured(fn)
            assert (ref == got).all(), name


def test_a_replay_picks_up_the_index_written_since_the_capture():
    """The point of capturing a lookup: the token changes every step, and
    the replay has to read the new one rather than the captured one."""
    with grilly.no_grad():
        w = grilly.from_numpy(np.arange(40, dtype=np.float32).reshape(10, 4))
        idx = grilly.tensor([3])
        out = None

        def lookup():
            nonlocal out
            out = F.embedding(idx, w)
            return out

        lookup()
        _C.begin_capture()
        lookup()
        graph = _C.end_capture()
        assert (out.numpy() == [[12.0, 13.0, 14.0, 15.0]]).all()

        idx.copy_(grilly.tensor([7]))
        graph.replay()
        assert (out.numpy() == [[28.0, 29.0, 30.0, 31.0]]).all()


# ── writes: what a ring buffer needs ─────────────────────────────────────


def test_the_in_place_index_writes_capture_and_replay():
    with grilly.no_grad():
        dst = grilly.zeros(8, 4)
        src = grilly.ones(1, 4)
        row = grilly.tensor([3], dtype=grilly.int64)
        wide = row.reshape(1, 1).expand(1, 4)

        for name, fn in [
            ("scatter_", lambda: dst.scatter_(0, wide, src)),
            ("index_copy_", lambda: dst.index_copy_(0, row, src)),
            ("index_add_", lambda: dst.index_add_(0, row, src)),
            ("index_fill_", lambda: dst.index_fill_(0, row, 1.0)),
            ("dst[row] = src", lambda: dst.__setitem__(row, src)),
        ]:
            dst.copy_(grilly.zeros(8, 4))
            fn()
            ref = dst.numpy().copy()

            dst.copy_(grilly.zeros(8, 4))
            _C.begin_capture()
            fn()
            graph = _C.end_capture()
            dst.copy_(grilly.zeros(8, 4))
            graph.replay()
            assert (dst.numpy() == ref).all(), name


def test_a_ring_buffer_write_follows_a_device_cursor():
    """What the whole thing is for: a slot chosen on the device, written in
    place, replayed with the cursor advancing."""
    with grilly.no_grad():
        ring = grilly.zeros(4, 2)
        value = grilly.ones(1, 2)
        slot = grilly.zeros(1, dtype=grilly.int64)

        def write():
            ring.index_copy_(0, slot, value)

        write()
        _C.begin_capture()
        write()
        graph = _C.end_capture()

        for step in (1, 2, 3):
            slot.copy_(grilly.tensor([step], dtype=grilly.int64))
            value.copy_(grilly.full((1, 2), float(step + 1)))
            graph.replay()
        assert (ring.numpy()[:, 0] == [1.0, 2.0, 3.0, 4.0]).all()


def test_the_new_index_writes_match_torch():
    import torch

    rng = np.random.default_rng(0)
    for name in ("index_copy_", "index_add_", "index_fill_"):
        for dim in (0, 1):
            x = rng.standard_normal((5, 4)).astype(np.float32)
            idx = np.array([2, 0, 2] if name == "index_add_" else [2, 0, 3], dtype=np.int64)
            if dim == 1:
                idx = idx % 4
            shape = list(x.shape)
            shape[dim] = len(idx)
            src = rng.standard_normal(shape).astype(np.float32)
            g, t = grilly.from_numpy(x.copy()), torch.from_numpy(x.copy())
            if name == "index_fill_":
                getattr(g, name)(dim, grilly.from_numpy(idx), 3.5)
                getattr(t, name)(dim, torch.from_numpy(idx), 3.5)
            else:
                getattr(g, name)(dim, grilly.from_numpy(idx), grilly.from_numpy(src))
                getattr(t, name)(dim, torch.from_numpy(idx), torch.from_numpy(src))
            # index_add_ sums duplicates, so its order is the hardware's
            bound = 1e-6 if name == "index_add_" else 0.0
            assert np.abs(g.numpy() - t.numpy()).max() <= bound, f"{name} dim={dim}"


# ── the error still arrives ──────────────────────────────────────────────


def test_a_bad_index_raises_at_the_next_read_with_torchs_wording():
    w = grilly.randn(10, 4, generator=gen(2))
    with pytest.raises(IndexError, match="index out of range in self"):
        F.embedding(grilly.tensor([10]), w)
        grilly._C.synchronize()

    with pytest.raises(RuntimeError, match="out of bounds"):
        grilly.gather(w, 0, grilly.tensor([[99, 0, 0, 0]]))
        grilly._C.synchronize()


def test_the_message_says_the_report_is_not_the_indexing_line():
    """Because it is not, and a reader who does not know that will look in
    the wrong place."""
    w = grilly.randn(4, 2, generator=gen(3))
    with pytest.raises(RuntimeError) as caught:
        w.index_select(0, grilly.tensor([9]))
        grilly._C.synchronize()
    assert "on the device" in str(caught.value)
    assert "index_select" in str(caught.value)


def test_one_fault_is_reported_once_and_then_cleared():
    w = grilly.randn(4, 2, generator=gen(4))
    with pytest.raises(RuntimeError):
        w.index_select(0, grilly.tensor([9]))
        grilly._C.synchronize()
    # the record is clear again, so the next read is not poisoned
    grilly._C.synchronize()
    assert w.index_select(0, grilly.tensor([1])).numpy().shape == (1, 2)


def test_a_fault_inside_a_capture_surfaces_after_it_closes():
    """A capture cannot read, so the fault waits — rather than being lost."""
    w = grilly.randn(4, 2, generator=gen(5))
    bad = grilly.tensor([9])          # built outside: a capture cannot write
    with grilly.no_grad():
        _C.begin_capture()
        w.index_select(0, bad)
        _C.end_capture()
    with pytest.raises(RuntimeError, match="out of bounds|DATA bounds"):
        grilly._C.synchronize()
