"""``grilly.graphed`` (docs/capture.md): record a step's GPU commands once
and replay them instead of running the Python again.

Tolerance: **bit-exact**. This is case (a) of CLAUDE.md §9 at its strictest
— a replay issues the *same commands over the same buffers* as the capture,
so not even the order of a summation differs. Anything but `==` here is a
bug in the recording, not float32.
"""

from __future__ import annotations

import numpy as np
import pytest

import grilly


def rand(*shape, seed=0):
    return grilly.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def chain(x, w):
    return ((x @ w).relu() * 2.0 - 1.0).tanh()


def test_a_replay_recomputes_bit_exactly_from_the_new_inputs():
    w = rand(32, 32, seed=1)
    g = grilly.graphed(chain)
    with grilly.no_grad():
        for seed in range(6):
            x = rand(4, 32, seed=seed)
            got = g(x, w).numpy().copy()
            assert (got == chain(x, w).numpy()).all(), f"seed {seed}"
        assert g.replays >= 3, "nothing replayed, so the comparison proved nothing"


def test_the_first_call_runs_eagerly_and_the_second_records():
    """Recording on the second sighting is what makes a growing cache free:
    a shape seen once is never captured."""
    w = rand(16, 16, seed=2)
    g = grilly.graphed(chain)
    with grilly.no_grad():
        g(rand(2, 16), w)
        assert (g.captures, g.misses, g.replays) == (0, 1, 0)
        g(rand(2, 16, seed=3), w)
        assert (g.captures, g.misses, g.replays) == (1, 1, 0)
        g(rand(2, 16, seed=4), w)
        assert (g.captures, g.misses, g.replays) == (1, 1, 1)


def test_a_shape_that_never_repeats_is_never_recorded():
    """A KV cache growing by one token per step: every call is a new shape,
    so the wrapper must cost a dictionary lookup and nothing else."""
    w = rand(8, 8, seed=5)
    g = grilly.graphed(chain)
    with grilly.no_grad():
        for rows in range(1, 12):
            g(rand(rows, 8, seed=rows), w)
    assert (g.captures, g.replays) == (0, 0)
    assert g.misses == 11


def test_the_cache_is_bounded():
    """A graph retains every buffer its commands name, so an unbounded cache
    on a changing shape would hold the whole history."""
    w = rand(8, 8, seed=6)
    g = grilly.graphed(chain, max_graphs=2)
    with grilly.no_grad():
        for rows in (1, 2, 3, 1, 2, 3):      # each shape seen twice -> each records
            g(rand(rows, 8, seed=rows), w)
    assert g.captures == 3 and g.graphs == 2


def test_outputs_are_the_same_tensors_every_replay():
    w = rand(8, 8, seed=7)
    g = grilly.graphed(chain)
    with grilly.no_grad():
        g(rand(2, 8), w)
        first = g(rand(2, 8, seed=8), w)
        assert g(rand(2, 8, seed=9), w) is first


def test_a_host_read_inside_the_capture_is_refused():
    """A replay cannot re-run the host, so a step that decides something from
    a value must fail loudly rather than replay a stale one."""
    w = rand(8, 8, seed=10)

    def peeks(x, w):
        y = x @ w
        return y * float(y.sum().item())

    g = grilly.graphed(peeks)
    with grilly.no_grad():
        g(rand(2, 8), w)                                  # first sighting: eager, fine
        with pytest.raises(RuntimeError, match="read data back"):
            g(rand(2, 8, seed=11), w)
        # a refused capture must leave the device usable
        assert chain(rand(2, 8), w).numpy().shape == (2, 8)


def test_enabled_false_is_the_undecorated_callable():
    w = rand(8, 8, seed=12)
    g = grilly.graphed(chain, enabled=False)
    x = rand(2, 8)
    with grilly.no_grad():
        assert (g(x, w).numpy() == chain(x, w).numpy()).all()



def test_a_kv_cache_decode_step_replays_and_the_cache_advances():
    """The cache's fill level is a device cursor, so a recorded append writes
    the next slot and a recorded attend covers everything held. Bit-exact
    against the same steps run eagerly on a second cache."""
    from grilly.infer.cache import KVCache

    def token(seed):
        return rand(1, 2, 1, 8, seed=seed)

    def step(cache, k, v, q):
        cache.append(0, k, v)
        return cache.attend(0, q)

    with grilly.no_grad():
        eager, graphed_cache = KVCache(1, 1, 2, 8, 64), KVCache(1, 1, 2, 8, 64)
        for cache in (eager, graphed_cache):
            cache.append(0, token(100), token(200))

        want = [step(eager, token(t), token(50 + t), token(80 + t)).numpy().copy()
                for t in range(5)]

        g = grilly.graphed(step)
        got = [g(graphed_cache, token(t), token(50 + t), token(80 + t)).numpy().copy()
               for t in range(5)]

    assert g.replays >= 2, "nothing replayed, so the comparison proved nothing"
    for t, (a, b) in enumerate(zip(want, got)):
        assert (a == b).all(), f"step {t}"
    assert graphed_cache.device_len(0) == 6


def test_a_compiled_region_can_be_captured():
    """``graphed(compile(fn))`` — fusion removes launches, replay removes
    the Python, and they were mutually exclusive: a fused kernel uploads a
    metadata blob from the host and a capture refused the write.

    Those bytes are the shapes and buffer offsets a capture already fixes,
    plus the scalars written into the expression, so they are constant for
    the recording's life and the replay re-reads the same buffer
    (``Buffer::write_constant``). Bit-exact against the uncompiled path.
    """
    w = rand(32, 32, seed=30)
    b = rand(32, seed=31)

    def block(x):
        h = (x @ w + b).tanh()
        return (h * 0.5 + 0.25).relu()

    plain = grilly.graphed(block)
    fused = grilly.graphed(grilly.compile(block))
    with grilly.no_grad():
        for seed in range(5):
            x = rand(4, 32, seed=40 + seed)
            assert (fused(x).numpy() == plain(x).numpy()).all(), f"seed {seed}"
    assert fused.replays >= 2 and plain.replays >= 2, "neither actually replayed"


def test_a_captured_step_can_look_up_an_embedding():
    """The op that kept every language model out of a graph."""
    table = grilly.from_numpy(
        np.arange(200, dtype=np.float32).reshape(50, 4))
    token = grilly.tensor([3])

    def step(tok):
        return grilly.nn.functional.embedding(tok, table) * 2.0

    g = grilly.graphed(step)
    with grilly.no_grad():
        want = [(step(grilly.tensor([t])).numpy().copy()) for t in (3, 11, 47)]
        got = [g(grilly.tensor([t])).numpy().copy() for t in (3, 11, 47)]
    assert g.replays >= 1
    for a, b in zip(want, got):
        assert (a == b).all()


def test_a_captured_step_can_build_a_mask_with_arange():
    """The other op that kept a decode step out of a graph.

    ``arange`` used to build its values on the host and upload them, which
    is a host write — refused inside a capture. Almost every mask is
    written ``arange(n) < length``, so a ring buffer, a validity mask and a
    fixed-capacity store were all outside ``graphed`` for want of a
    counting sequence. It is a device op now."""
    ring = grilly.from_numpy(np.arange(32, dtype=np.float32).reshape(8, 4))
    pos = grilly.zeros(())

    def masked(pos):
        live = grilly.arange(8, dtype=grilly.float32) < grilly.clamp(pos + 1.0, max=8.0)
        return grilly.where(live.reshape(8, 1), ring, grilly.zeros((8, 4)))

    g = grilly.graphed(masked)
    with grilly.no_grad():
        want, got = [], []
        for p in (0.0, 1.0, 3.0, 7.0):
            pos.fill_(p)
            want.append(masked(pos).numpy().copy())
            got.append(g(pos).numpy().copy())
    assert g.replays >= 2, "nothing replayed, so the comparison proved nothing"
    for a, b in zip(want, got):
        assert (a == b).all()


def test_a_replayed_ring_slot_follows_the_position_counter():
    """``pos % window`` chosen on the device, written in place, replayed —
    the shape a windowed KV cache and an episodic store both take."""
    ring = grilly.zeros((4, 2))
    pos = grilly.zeros(())
    value = grilly.ones((1, 2))

    def write(pos, value):
        slot = grilly.remainder(pos, 4.0).long().reshape(1)
        ring.index_copy_(0, slot, value)
        return ring

    g = grilly.graphed(write)
    with grilly.no_grad():
        for step in range(6):
            pos.fill_(float(step))
            value.fill_(float(step + 1))
            g(pos, value)
    assert g.replays >= 3
    # steps 4 and 5 wrapped onto slots 0 and 1
    assert (ring.numpy()[:, 0] == [5.0, 6.0, 3.0, 4.0]).all()
