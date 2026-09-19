# Submission: batching, its costs, and the fence timeout

Measurements behind CLAUDE.md §11's premise that batching tiny submits is
the win for a decode loop. All on the RX 6750 XT.

## The fence timeout's root cause (fb9d1cf)

It was not a long dispatch. Buffers freed while a batch was recording were
held until the submit, so a batch's footprint was the sum of all its
intermediates. Dense attention at S=4096 reached ~20 GB, WDDM paged half of
it to system RAM, the batch ran 12x slower, and v1's 2 s fence wait
expired.

Same 36 dispatches:

| | GPU wait | dedicated | shared (system RAM) |
|---|---:|---:|---:|
| one submit, before | 4.07 s | 10.0 GB | 10.1 GB |
| one submit per dispatch | 0.34 s | 6.6 GB | 0.1 GB |
| one submit, fixed | 0.34 s | 6.6 GB | 0.1 GB |

The fix: device buffers freed during recording are reused by later commands
of the same batch, and handed back to v1's pool only after the batch runs.
Why this is safe is in the commit message and `tests/python/test_batch_reuse.py`.

## Batching throughput (1a1a443, `tools/decode_stress.py`)

A decode-like step: an F.linear projection, elementwise ops of mixed sizes,
a softmax over the vocabulary, a max. That is 6 ops and 16 dispatches,
repeated 3,000 times.

| D, vocab | one submit per op | one per step | fully batched |
|---|---:|---:|---:|
| 1024, 32K | 11,370 ops/s | 23,982 (2.1x) | 35,796 (3.2x) |
| 2048, 50K | 9,167 ops/s | 18,201 (2.0x) | 25,168 (2.8x) |

- Each extra submit costs ~60 us.
- There is no growth over 18,000 submits: the per-submit wait is flat
  (45-53 us by fifth), there is one descriptor pool, and there are no new
  allocations after warm-up.
- The regression guard is `tests/python/test_decode_stress.py`.

## Where a dispatch's time goes (`crates/grilly-sys/tests/dispatch_cost.rs`)

This drives the shim directly (no Python, no tensor layer): 1,000
dispatches per batch, microseconds per dispatch.

| elements | GPU, barrier | GPU, no barrier | host recording |
|---:|---:|---:|---:|
| 1 | 0.92 | 0.26 | 1.1 |
| 256 | 0.96 | 0.28 | 1.1 |
| 4,096 | 1.00 | 0.25 | 1.1 |
| 65,536 | 1.40 | 0.54 | 1.1 |
| 262,144 | 3.30 | 1.70 | 1.1 |

- **The barrier** grilly_core records before each command costs ~0.65 us
  per tiny dispatch. For larger independent kernels it costs up to ~1.6
  us, because it stops them overlapping. A dependent chain needs it
  anyway. Alternating two pipelines costs nothing measurable.
- **The GPU fixed cost** of a tiny dispatch is under 1 us, so batching tiny
  work is cheap on the GPU side. In the decode step above, ~4.8 us of
  GPU per dispatch is mostly real memory traffic. The 1024x1024 linear
  alone reads 4 MB. The barrier is ~13% of that step's GPU time.
- **The host is the gap.** The shim records a dispatch in 1.1 us; grilly
  takes ~5.7 us from Python. The extra ~4.6 us is the Python op, PyO3 and
  the Rust tensor layer. A 400-dispatch decode step would spend ~2.3 ms
  per token on host recording, against ~0.36 ms of GPU fixed cost.

## What this says for §11

The premise holds: batching tiny submits is the win, and the GPU side of it
is cheap. The limit is host-side, so the levers, in order of payoff:

1. **Cut the per-op host overhead.** Profile the Python -> Rust path.
2. **Replay a recorded decode step,** as CUDA graphs do: record once,
   submit per token, which removes per-token recording.
3. **Fuse elementwise chains** (docs/fusion.md), for fewer dispatches.
4. **Skip the barrier between independent dispatches:** at most ~0.65 us
   each.
