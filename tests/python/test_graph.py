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
