"""Decode-shaped submission stress, as a regression guard (CLAUDE.md §11).

A decode loop is thousands of tiny submits. Growth over such a run means
pooling or recycling is wrong: descriptor pools allocated per submit,
buffers not returned, fences not recycled. This runs ~2,400 small
submits (a decode-like step, one sync per op) and asserts, from the shim's
counters rather than timing:
  - no new descriptor pools and no new buffer allocations after warm-up;
  - outstanding buffers return to where they started;
  - the per-submit wait does not grow: the median of the last fifth is
    within 2x of the first fifth's (measured flat, 45-53 us per fifth over
    18,000 submits on the RX 6750 XT; the 2x margin absorbs noise, a leak
    would not stay under it).
Throughput is measured by tools/decode_stress.py, not asserted here.
"""

import statistics

import numpy as np

import grilly
from grilly import _C

F = grilly.nn.functional


def test_many_small_submits_do_not_grow_pools_or_waits():
    rng = np.random.default_rng(0)
    D, V = 512, 8000
    w = grilly.tensor((rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32))
    head = grilly.tensor((rng.standard_normal((1, V)) * 0.01).astype(np.float32))
    bias = grilly.tensor(rng.standard_normal(D).astype(np.float32))
    x = grilly.tensor(rng.standard_normal((1, D)).astype(np.float32))

    def step(x):
        yield F.linear(x, w)
        x = F.linear(x, w)
        yield x * grilly.sigmoid(x)
        x = x * grilly.sigmoid(x) + bias
        yield x
        p = (head * x[:, :1] + 0.5).softmax(-1)
        yield p
        yield p.amax(-1)

    for _ in range(20):  # warm-up: compile pipelines, fill the pool
        for _ in step(x):
            _C.synchronize()
    s0 = _C.submit_stats()
    waits = []
    for _ in range(480):
        for _ in step(x):
            _C.synchronize()
            waits.append(_C.submit_stats()["wait_ns_last"])
    s1 = _C.submit_stats()

    assert s1["submits"] - s0["submits"] >= 2400
    assert s1["desc_pools"] == s0["desc_pools"], "descriptor pools grew with submits"
    assert s1["buffer_allocations"] - s0["buffer_allocations"] <= 2, "buffers are allocated per submit"
    assert s1["buffers_outstanding"] == s0["buffers_outstanding"], "buffers are not returned"
    fifth = len(waits) // 5
    first, last = statistics.median(waits[:fifth]), statistics.median(waits[-fifth:])
    assert last <= 2 * first, f"per-submit wait grew over the run: {first / 1e3:.1f} -> {last / 1e3:.1f} us"
