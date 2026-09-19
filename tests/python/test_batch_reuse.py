"""Buffers freed while a batch is recording are reused by later commands in
the same batch (the shim's batch-local free list; CLAUDE.md §11, the root
cause of the fence timeout: without it a batch's footprint was the sum of
all its intermediates and paged to system RAM).

What makes that safe is ownership, not ordering: a buffer reaches the free
list only when the Rust `Buffer` owning it is dropped (the only call site of
`gc_buffer_release`), which happens when no tensor, view or autograd-saved
tensor refers to it; the handle is gone from the shim's table before the
buffer is listed, and handles are never reused. These tests check that live
data survives recycling around it. Every one asserts the reuse counter
moved: a test in which nothing was recycled would pass vacuously.

Only exactly-rounded ops are used (float32 add, sub, mul: `precise` on the
GPU, one rounding each in numpy), so results are compared with numpy for
exact equality.
"""

import numpy as np
import torch

import grilly
from grilly import _C

RNG = np.random.default_rng(99)
SIZES = [37, 1000, 4096, 1000, 37, 70_000, 4096, 1000, 300, 70_000]  # mixed sizes, shared buckets


def reuses() -> int:
    return _C.submit_stats()["batch_reuses"]


def settle(*tensors) -> None:
    """Close any open batch, so what follows records into a fresh one."""
    grilly.zeros(1).sum().item()


def test_live_tensors_survive_recycling_in_one_batch():
    xs_np = [RNG.standard_normal(n).astype(np.float32) for n in SIZES]
    xs = [grilly.tensor(a) for a in xs_np]  # uploads happen here, before the batch
    settle()
    before = reuses()

    canaries, views = [], []
    for step in range(30):
        for i, x in enumerate(xs):
            t = x * 2.0 + 1.0      # temporaries: each rebinding frees the previous buffer
            t = t * t - x
            t = t * 0.5
            if step == 0:
                canaries.append(t * 1.0)             # stays alive, unsynced, all batch long
            if step == 1:
                base = t + 3.0
                views.append(base.reshape(-1, 1))    # a view sharing storage;
                del base                              # its base's handle is dropped
    got_canaries = [c.numpy() for c in canaries]     # the only sync
    got_views = [v.numpy()[:, 0] for v in views]
    reused = reuses() - before

    for a, got, view in zip(xs_np, got_canaries, got_views):
        t = (a * np.float32(2.0) + np.float32(1.0))
        t = (t * t - a) * np.float32(0.5)
        np.testing.assert_array_equal(got, t * np.float32(1.0))
        np.testing.assert_array_equal(view, t + np.float32(3.0))
    assert reused > 100, f"only {reused} in-batch reuses: the test did not exercise recycling"


def test_autograd_saved_tensors_survive_recycling():
    """The forward saves tensors for backward; the backward runs in the same
    batch, amid recycled temporaries. Gradients equal torch's exactly (the
    same exactly-rounded ops on both sides)."""
    a_np = RNG.standard_normal(5000).astype(np.float32)
    w_np = RNG.standard_normal(5000).astype(np.float32)
    x, w = grilly.tensor(a_np, requires_grad=True), grilly.tensor(w_np, requires_grad=True)
    settle()
    before = reuses()
    h = x
    for _ in range(12):
        h = h * w + x          # saves h, w for the backward; frees nothing they need
        h = h * 0.5
    h.backward(grilly.tensor(np.ones(5000, np.float32)))
    gx, gw = x.grad.numpy(), w.grad.numpy()
    reused = reuses() - before

    xt, wt = torch.tensor(a_np, requires_grad=True), torch.tensor(w_np, requires_grad=True)
    ht = xt
    for _ in range(12):
        ht = ht * wt + xt
        ht = ht * 0.5
    ht.backward(torch.ones(5000))
    np.testing.assert_array_equal(gx, xt.grad.numpy())
    np.testing.assert_array_equal(gw, wt.grad.numpy())
    assert reused > 10, f"only {reused} in-batch reuses: the test did not exercise recycling"


def test_buffers_return_to_the_pool_after_the_batch():
    """What a batch freed goes back to v1's pool when it closes: a second
    identical batch allocates (almost) nothing new."""
    x = grilly.tensor(RNG.standard_normal(4096).astype(np.float32))
    settle()

    def batch():
        t = x
        for _ in range(50):
            t = t * 1.0001 + 0.5
        return t.numpy()

    batch()
    s0 = _C.submit_stats()
    batch()
    s1 = _C.submit_stats()
    assert s1["batch_reuses"] > s0["batch_reuses"]
    assert s1["buffer_allocations"] - s0["buffer_allocations"] <= 2
