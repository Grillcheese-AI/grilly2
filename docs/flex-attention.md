# flex_attention: from dense to a real kernel

**Decided (owner, 2026-09-19):**
- Option A: fused block-sparse kernels.
- A `score_mod` falls back to the dense path with a one-time warning.
- The backward is two deterministic kernels, with no atomics.
- Shader fusion must build on grilly v1's (see "v1's attention kernels and
  shader fusion" below; one choice there is still open).

Built: all four stages.
- Tile lists (built on the GPU by g2-flash-tiles, no sync).
- The fused forward (1a71112).
- The deterministic two-kernel backward (335dec2).
- SDPA routed through the kernels: no dropout, bool or no mask,
  is_causal, 3-D or 4-D inputs.

flex_attention takes the fused path whenever there is no score_mod and
head dims are <= 128. Float masks and dropout keep SDPA dense.

Measured forward + backward at 16 heads x dim 128, window 512, on the
RX 6750 XT:

| S | flex_attention | SDPA, bool mask built per forward (CubbyLLM) |
|---:|---:|---:|
| 1K | 20 ms | 21 ms |
| 2K | 44 ms | 46 ms |
| 4K | 94 ms | 102 ms |
| 8K | 193 ms | 217 ms |

The cost is linear in S. The dense path took 41 ms and 132 ms at 1K and
2K and could not finish 4K. Building tiles on the GPU takes 23 ms at
8K; the earlier host build took 295 ms.

Left to optimize: the forward runs at about 0.6 TFLOP/s against 1.1
for g2-bmm.

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

1. **No general JIT from Python mods.** v1 has a JIT shader-fusion engine
   (below), but it generates elementwise chains of five known ops, not
   arbitrary Python. So a `mask_mod` is evaluated once, densely, at
   `create_block_mask` time (as now) and stored as block lists plus a bit
   mask. A `score_mod` stays on the dense path.
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

## Recommendation (A was chosen)

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

## v1's attention kernels and shader fusion

An earlier version of this document said v1 has no flash kernel. That was
wrong: the search that concluded it was cut short. v1 ships these,
compiled:

- **`flash-attention2`: not usable.** Every thread in a query row does
  unsynchronized read-modify-writes of the same running max, sum and
  accumulator (a race). It divides each weight by a partial sum and then
  again by the final sum. Its only mask is per-key padding.
- **`flash-attention3`: not usable here.** It has subgroup reductions and
  double-buffered tiles, but one workgroup per query row. It takes no
  mask (not even causal), writes no log-sum-exp for a backward, and its
  shared tiles cap the head dim at 64.
- **`chunked-sw-attention`: correct for one mask.** It computes the
  causal window [q-W+1, q] (CubbyLLM's), one workgroup per query row,
  walking the window a key at a time with a 7-barrier reduction per key.
  It uses native `exp` (measured 14 ULP off) and has no general masks,
  no GQA and no log-sum-exp.
- **`chunked-sw-attention-backward`: not usable on this GPU.**
  - It accumulates dK and dV with float `atomicAdd`, which lands every
    add at index 0 on this driver stack.
  - It keeps scores in `float scores[256]`, so CubbyLLM's window of 512
    writes past the array.

None of them does block-sparse attention with arbitrary masks, a saved
log-sum-exp and a deterministic backward, so A's kernels are new. They
follow `chunked-sw-attention`'s layout (BHSD) and online softmax.

**Shader fusion.** v1's `ShaderFusionEngine` (`cpp/src/shader_fusion.cpp`,
driven by `backend/jit.py`) works like this:
- It traces an op sequence and generates one GLSL compute shader that
  chains elementwise ops in registers.
- It compiles the shader at runtime with shaderc, and caches the pipeline
  by a BLAKE3 hash of the graph in v1's `PipelineCache`.
- shaderc is linked into grilly2's build of v1 (`SHADERC_LIB` found in
  the Vulkan SDK), so the engine is available.

Its code generator knows five ops: gelu, relu, silu, add/residual and
dropout. It silently skips any other op, and its dropout has its own RNG,
not grilly's Philox. Growing that list means editing v1's C++, which is
frozen.

**Open question: how grilly2 uses it.**

- **F1: call v1's engine through the shim, as is.**
  - Fuses chains of those five ops.
  - Smallest change.
  - The silent skip has to be guarded on grilly2's side.
- **F2: keep v1's design and machinery, generate in grilly2** (recommended).
  - One new shim entry compiles GLSL through the shaderc already linked
    into v1.
  - Pipelines are cached in v1's `PipelineCache`, so the one dispatch
    spine holds.
  - A Rust generator emits chains of grilly2's own elementwise ops from
    `g2-common`, so a fused result is bitwise the unfused one. Those ops
    are the torch-accurate ones (Cephes exp and tanh, sin and cos).
  - The entry point is torch's name, `torch.compile`, not a new API.

Either way, the flex kernels do not depend on it.

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
