# The GPU boundary

The answer to CLAUDE.md §11: what `grilly-core` needs from v1's C++ Vulkan layer,
how wide that surface is, and which language sits on the other side of it.

**Decision (owner, 2026-09-18): Rust core, with the shim written against v1's
Vulkan primitives (level A below).**

## Measured surface

v1's `grilly_core_lib` offers four levels a consumer could link against.

| level | what | size | what crosses |
|---|---|---|---|
| A | `GrillyDevice`, `BufferPool`, `PipelineCache`, `CommandBatch` | 4 classes, ~25 methods | raw `VkBuffer`, `VkPipeline`, `VkDescriptorSet`, `VmaAllocation` |
| B | `VulkanBackend` (`compute_backend.h`) | 13 methods, 222 lines | `u64` handles, shader-name strings, grid, push bytes |
| C | op functions, `src/ops/*.cpp` | 101 functions, ~9K lines | host pointers + per-op param structs |
| D | `autograd::TapeContext` | 1,683 lines, its own `forward_*` ops | `u32` buffer ids, `Node*`, arena |

**C is not a usable boundary.** Every op function stages all of its inputs
host → device, dispatches, stages the output back, and waits (`ops/linear.cpp`).
A `Linear` forward re-uploads its weights on every call and nothing stays resident
between ops. Fine for one-shot inference; unusable for training.

**D must not cross.** `TapeContext` is a second autograd with its own forward op
set — the second dispatch route §5 forbids. Autograd is what grilly2 rewrites; the
tape lives in Rust.

**B is the right shape but has two defects** that cannot be fixed without editing
v1: `dispatch` binds every buffer whole at offset 0 (no tensor views), and
`destroyBuffer` frees memory with `vmaDestroyBuffer` instead of returning it to the
pool.

So the shim is built on **A**, lives in this repository
(`crates/grilly-sys/shim/`), compiles against v1's headers, and links
`grilly_core_lib`. v1 is not modified.

## Answers to the three questions

1. **Entry points** — the `gc_*` functions in `crates/grilly-sys/shim/gc_shim.h`,
   all plain data or opaque handles: device create/destroy and capabilities,
   shader loading, buffer create/release, upload/download, and
   begin / dispatch / barrier / submit.
2. **What crosses** — one opaque context pointer and `u64` buffer handles.
   Descriptor sets, pipelines and VMA allocations never reach Rust.
   `TapeContext` does not cross at all.
3. **State** — everything long-lived (device, pool, pipeline cache, command
   buffer, handle table, descriptor pools) is inside the one C++ context object.
   Per call: shader name, bindings `(handle, offset, range)`, grid, push bytes.
   Ownership is one-directional: Rust owns handles and releases them on drop;
   C++ owns the memory.

Narrow and mostly opaque, so by §11's rule Rust stands.

## v1 behaviour the shim deliberately does not use

- **`PipelineCache::allocDescriptorSet`** caches descriptor sets keyed on
  `(VkBuffer, range)` with no offset. Two dispatches binding the same buffer at
  different offsets with the same range would share one set and read the wrong
  data. v1 never binds at an offset, so it never showed. The shim allocates
  descriptor sets from its own per-batch pool, reset after each submit.
  Regression test: `offset_bindings_do_not_alias`.
- **`PipelineCache::getOrCreate`** keys pipelines by shader name only, so a second
  call with a different binding count or push size silently gets the first
  layout. The shim records each shader's `(bindings, push bytes)` on first use and
  rejects a mismatch.
- **`CommandBatch::barrier`** makes shader writes visible to later reads but not to
  later writes. The shim records its own barrier with
  `dstAccessMask = READ | WRITE`.

## Cost accepted with this choice

The per-op knowledge now inside `src/ops/*.cpp` — shader name, binding order,
push-constant layout, grid, cooperative-matrix selection — is re-expressed once
in grilly2's op registry. That registry is the single dispatch spine of §5, so
this is work the project needs anyway, not a detour.

## Found during the analysis, not yet resolved

- **§7 baseline.** `backend/_bridge.py` calls `dev.blockcode_bind`, but that name
  exists nowhere in v1's C++ sources, in its git history (`git log -S`), or in the
  compiled `grilly_core` extension; any exception falls through silently to a
  numpy loop. Either the 1.83× measurement came from a different build or it
  measured the fallback. Must be settled before §7 is used as a regression
  baseline.
- **Milestone-1 coverage.** v1 has no `GRU` (minGRU only). The `t_*` tensor ops
  are CPU Eigen code, not GPU. No `gather`, `cat`, `stack` or `bmm` is bound in
  the torch-tier binding files.
