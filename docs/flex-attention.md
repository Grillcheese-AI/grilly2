# flex_attention: from dense to a real kernel

**Status: options for the owner to decide.** Nothing below is built.

## Where it stands

`grilly.nn.attention.flex_attention` is dense (commit 421a9b7). It builds
the full (Q, KV) score matrix, applies `score_mod`, masks it, and runs a
softmax and two matmuls. That is the math torch's *uncompiled*
flex_attention does. It is correct against torch 2.10, and CubbyLLM's
suite passes, but a block mask saves no work.

Measured on the RX 6750 XT, forward + backward, one sequence, 16 heads of
dim 128, CubbyLLM's causal window of 512:

| S | time | one score matrix | fraction of scores the mask keeps |
|---:|---:|---:|---:|
| 1024 | 41 ms | 0.06 GiB | 0.38 |
| 2048 | 132 ms | 0.25 GiB | 0.22 |
| 4096 | did not finish: exceeded v1's 2 s fence timeout | 1.0 GiB | 0.12 |

The cost is quadratic in S, while the work the mask keeps is linear. At
S = 4096 the backward holds several score-sized tensors, each 1 GiB, on a
12 GB card. That is the failure CubbyLLM's `_window_block_mask` docstring
describes on CUDA ("MFU 40.3% at S=1024 collapsing to 21.8% at S=4096"),
and the reason flex_attention exists.

## What a real kernel has to do

torch compiles `mask_mod` and `score_mod` into a Triton kernel and keeps a
`BlockMask` of 128x128 blocks:

- `kv_indices`: for each query block, the key blocks it touches.
- `full_kv_indices`: the key blocks where every position is kept, so no
  mask is evaluated there.
- `q_indices`: the transpose of `kv_indices`, for the backward.

Work is proportional to the blocks kept.

Four constraints are specific to grilly:

1. **No JIT from Python to SPIR-V.** grilly cannot compile an arbitrary
   Python `score_mod` / `mask_mod` into a shader. A mask can still be
   evaluated once, densely, at `create_block_mask` time (as now) and
   stored as block indices plus bitmaps for partial blocks. A `score_mod`
   cannot be moved into a kernel unless it is one grilly recognizes.
2. **Float atomics are broken on this driver stack.** Float `atomicAdd`
   lands every add at index 0 (measured in v1 and again in grilly2). The
   standard FlashAttention-2 backward accumulates dQ with atomics. Here
   that would be a CAS loop, which is slow and nondeterministic.
3. **v1's fence timeout is 2 seconds per submitted batch.** Long work has
   to be split across submits. This bites dense attention at S = 4096
   today, and is its own fix (see "Also found").
4. **RDNA2 gives a workgroup 64 KiB of shared memory.** With head dim 128
   in float32, a 64-row Q tile plus 64-row K and V tiles is 96 KiB, too
   much. Tiles have to be 32x64 or 64x32, or head-dim-specific.

## Options

### A. Fused block-sparse attention (FlashAttention-2 style): recommended

- **Forward.** One workgroup per (batch, head, query tile). It loops only
  over the key tiles in `kv_indices`: S = Q K^T for the tile, a partial
  block's bitmap applied, then an online softmax (running max and sum)
  and O += P V. It writes O and the log-sum-exp. Scores never reach
  memory.
- **Backward, as two kernels, deterministically:**
  - one per key tile, over its `q_indices`, recomputing P from the saved
    log-sum-exp and accumulating dK and dV;
  - one per query tile, over its `kv_indices`, accumulating dQ.

  This takes 7 tile matmuls where FlashAttention-2 takes 5 (S and dP are
  recomputed for dQ), about 40% more backward matmul work, and avoids
  atomics entirely. That is consistent with grilly's deterministic
  reductions and sidesteps the broken float atomics.
- **Mask.** `create_block_mask` evaluates `mask_mod` densely once, as now,
  then stores torch's block structure: kv/q indices, full-block lists,
  and a bitmap for each partial block. For a window of 512 there are
  about two partial blocks per query row.
- **score_mod.** Not in the fused path. A `score_mod` falls back to the
  dense path. A small set of common mods could become kernel variants
  later: an additive per-head bias times (q - k), ALiBi, tanh soft-cap.
- **SDPA gets it too.** `is_causal`, no mask, or a bool mask converted to
  a block mask can use the same kernels. That fixes SDPA's memory for
  long sequences as well.
- **Cost.** About 3 shaders (~500 lines of GLSL), BlockMask
  construction, and tests against the dense path at every shape and
  mask. A week-scale piece of work.
- **Payoff.** Linear in S for windowed masks, and O(S) memory instead
  of O(S^2).

### B. Banded composition of existing kernels

For each query tile, gather the key/value blocks it keeps into a
contiguous slab, then run bmm, mask, softmax and bmm on the slab.
Autograd comes free, because every piece is an existing differentiable
op.

- **Good at:** banded masks (CubbyLLM's window). Every query tile keeps
  about the same number of blocks, so the slabs are regular. Memory is
  O(S x window).
- **Bad at:** causal and document masks. The number of kept blocks
  varies by row, so slabs pad to the longest row and causal attention
  saves nothing. K and V are also copied once per kept block.
- **Cost:** about 100 lines of Python, a day or two.

### C. Chunked dense

Run the dense math over chunks of queries so only a (chunk, S) slice of
scores exists at a time. This fixes the memory ceiling and nothing else.
Compute stays quadratic. It could also serve as the stopgap for the
fence timeout.

## Recommendation

**A, staged:**

1. **BlockMask with real block indices.** This is the structure A needs
   and B could use. `flex_attention` stays dense for now.
2. **Fused forward.** Test it against the dense path. Inference and
   evaluation speed up here.
3. **Fused backward.** Two deterministic kernels. Training speeds up
   here.
4. **Route SDPA.** Send `is_causal` / bool masks through the same
   kernels.

Each stage is a commit with its own tests against the dense path, which
stays as the reference and as the `score_mod` fallback.

## Questions for the owner

1. **A (the real fix) or B (quick, windowed masks only)?**
2. **What should happen to a `score_mod` in A?**
   - (a) Dense fallback, silently. This is correct and slow.
   - (b) Dense fallback with a one-time warning.
   - (c) A fixed menu of fused score_mods, with anything else dense.
3. **Are two deterministic backward kernels acceptable instead of an
   atomic dQ?** They cost about 40% more backward matmul work (7 matmuls
   instead of 5). An atomic dQ would be a CAS loop on this GPU and
   nondeterministic.

## Also found

v1's `CommandBatch` waits at most 2 seconds for each submit
(`kFenceTimeoutNs` in `cpp/src/command_batch.cpp`). grilly records work
lazily and submits it together, so any queued work longer than 2 s
fails with `vkWaitForFences timed out`. The device survives. Dense
attention at S = 4096 hits this.

v1 stays frozen, so the fix belongs in grilly2's shim or recorder. The
options:
- split a batch into submits before it gets long;
- or wait again after a timeout, while the fence has not signalled.

This is independent of the choice above.
