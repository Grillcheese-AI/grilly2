"""Decode-shaped submission stress (CLAUDE.md §11).

A decode loop is thousands of tiny submits. This runs a decode-like step
(an F.linear projection (1, D) -> (1, D) as nn.Linear does it, elementwise
ops of mixed small sizes, a softmax over a vocabulary, a reduction) many
times in three patterns:

  per-op     a sync after every op: one submit per op
  per-step   a sync after every step: the shape of a real decode loop
  batched    everything recorded, one sync at the end (the device still
             flushes every 1024 launches)

and reports throughput, whether the per-submit wait grows over the run
(growth means pooling or recycling is wrong), and the shim's counters.

    python tools/decode_stress.py [--steps N] [--dim D] [--vocab V]
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import grilly
from grilly import _C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=32000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    D, V = args.dim, args.vocab
    w = grilly.tensor((rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32))
    head = grilly.tensor((rng.standard_normal((1, V)) * 0.01).astype(np.float32))
    x0 = grilly.tensor(rng.standard_normal((1, D)).astype(np.float32))
    bias = grilly.tensor(rng.standard_normal((D,)).astype(np.float32))

    def ops(x):
        """One decode-like step as a list of thunks, each one op."""
        state = {"x": x}

        def proj():
            state["x"] = grilly.nn.functional.linear(state["x"], w)  # weight (out, in), as nn.Linear
        def act():
            state["x"] = state["x"] * grilly.sigmoid(state["x"])
        def add_bias():
            state["x"] = state["x"] + bias
        def logits():
            state["l"] = head * state["x"][:, :1] + 0.5
        def soft():
            state["p"] = state["l"].softmax(-1)
        def pick():
            state["s"] = state["p"].amax(-1)
        return state, [proj, act, add_bias, logits, soft, pick]

    sync = _C.synchronize  # runs the open batch and waits; launches nothing itself

    # warm-up: compile pipelines, fill the pool
    state, thunks = ops(x0)
    for f in thunks:
        f()
    sync()

    def counters():
        return _C.submit_stats()

    # 1. per-op: one submit per op
    state, thunks = ops(x0)
    s0 = counters()
    waits, t0 = [], time.perf_counter()
    n_ops = 0
    for _ in range(args.steps):
        for f in thunks:
            f()
            sync()
            waits.append(counters()["wait_ns_last"])
            n_ops += 1
    per_op_s = time.perf_counter() - t0
    s1 = counters()

    # 2. per-step: one submit per decode step
    state, thunks = ops(x0)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for f in thunks:
            f()
        sync()
    per_step_s = time.perf_counter() - t0
    s2 = counters()

    # 3. batched: record everything, one sync at the end
    state, thunks = ops(x0)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        for f in thunks:
            f()
    t_record = time.perf_counter() - t0
    sync()
    batched_s = time.perf_counter() - t0
    s3 = counters()

    def row(name, n_ops, wall, a, b):
        wait = (b["wait_ns_total"] - a["wait_ns_total"]) / 1e9
        submits = b["submits"] - a["submits"]
        print(f"  {name:9s}: {n_ops} ops, {b['dispatches'] - a['dispatches']} dispatches, {submits} submits; "
              f"wall {wall:.3f} s = GPU wait {wait:.3f} s + host {wall - wait:.3f} s; "
              f"{n_ops / wall:,.0f} ops/s, host {1e6 * (wall - wait) / n_ops:.1f} us/op, "
              f"new allocations {b['buffer_allocations'] - a['buffer_allocations']}, "
              f"in-batch reuses {b['batch_reuses'] - a['batch_reuses']}")

    ops_n = args.steps * len(thunks)
    print(f"decode-shaped stress: {args.steps} steps x {len(thunks)} ops, D={D}, vocab={V}")
    row("per-op", n_ops, per_op_s, s0, s1)
    row("per-step", ops_n, per_step_s, s1, s2)
    row("batched", ops_n, batched_s, s2, s3)
    fifth = len(waits) // 5
    medians = [statistics.median(waits[i * fifth:(i + 1) * fifth]) / 1e3 for i in range(5)]
    print("  per-op median wait per submit, by fifth of the run (us): " + ", ".join(f"{m:.1f}" for m in medians))
    print(f"  counters after the run: desc pools {s3['desc_pools']}, outstanding buffers {s3['buffers_outstanding']} "
          f"(before the run {s0['buffers_outstanding']})")
    print(f"  throughput vs per-op: per-step {per_op_s / per_step_s * ops_n / n_ops:.2f}x, "
          f"batched {per_op_s / batched_s * ops_n / n_ops:.2f}x")

if __name__ == "__main__":
    main()
