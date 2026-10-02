# The GPU boundary

The answer to CLAUDE.md §11: what `grilly-core` needs from v1's C++ Vulkan layer,
how wide that surface is, and which language sits on the other side of it.

**Decision (owner, 2026-09-18): Rust core, with the shim written against v1's
Vulkan primitives (level A below).**

**Superseded in one respect (owner, 2026-09-23): the link to v1 is cut.** The
level-A choice below is unchanged and is what made cutting it cheap — the
narrower the boundary, the less there is to bring over. What changed is
where level A's four classes live: they are grilly2's own files now, in
`crates/grilly-sys/vulkan/`, built by the same `cc` invocation as the shim.
Nothing about the shim's shape moved. The measurement is at the end of this
document.

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
(`crates/grilly-sys/shim/`), and compiles against the four classes in
`crates/grilly-sys/vulkan/` — which were v1's, and are now grilly2's own
(see "Cutting the link", below).

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


## Cutting the link (2026-09-23)

The owner's goal was no dependency on grilly v1 left in grilly2. The first
thing to do with a goal like that is measure it, because "depends on v1"
sounded like the whole of `grilly_core_lib` and was not:

| what | size |
|---|---|
| C++ the shim actually names | `device.cpp` 584, `buffer_pool.cpp` 548, `pipeline_cache.cpp` 371, `command_batch.cpp` 260, plus 4 headers, 387 — **2,629 lines** |
| SPIR-V dispatched by name | **6 of v1's 273** — relu, relu-backward, elementwise-add, linear, linear-backward, residual (277 lines of GLSL) |
| external, through those | VulkanMemoryAllocator (11 symbols) and shaderc, both available on their own |

Nothing else of v1 was reachable: the four `.cpp` files include only
`<vulkan/vulkan.h>`, `<vk_mem_alloc.h>`, the standard library and each
other. So this was a **move, not a rewrite** — which is the only reason it
was a sensible thing to do at all, and is a direct consequence of picking
the narrowest of the four levels above in the first place.

What it changed:

- `crates/grilly-sys/vulkan/` holds the four classes and their headers;
  `build.rs` compiles them with `cc` beside the shim. The CMake build, the
  `cmake` build-dependency, `GRILLY_V1_DIR` and `GRILLY_CORE_LIB_DIR` are
  all gone, and so is the couple of minutes the first build took.
- the six shaders are `crates/grilly-core/shaders/g2-relu`,
  `g2-relu-backward`, `g2-add-inplace`, `g2-linear`, `g2-linear-backward`
  and `g2-residual`, compiled by the existing `build.rs` like every other
  `g2-*`. v1 compiled them at `--target-env=vulkan1.0` without `-O`;
  grilly2 compiles them at `vulkan1.2` with `-O`, and the op tests that
  compare against torch are unchanged, so the optimizer did not move a
  number.
- `Device::open()` takes no directory. Every kernel is grilly2's own;
  `GRILLY_SHADER_DIR` is now an *override* loaded after them, via
  `Device::open_with`, rather than the place the real kernels came from.
- `grilly-sys`'s own tests compile their probe kernel through
  `gc_compile_shader` instead of borrowing one from a shader directory, so
  the crate is testable with no files at all.
- VulkanMemoryAllocator is a submodule at
  `crates/grilly-sys/third_party/VulkanMemoryAllocator` (v3.4.0, pinned;
  v1 carried 3.4.0-development), or `VMA_INCLUDE_DIR`. It is not vendored —
  CLAUDE.md §9's rule stands, and a submodule is how it stands.

What it did not change: the shim's contract, the one dispatch path, or any
measured number. 113 Rust tests and 1015 Python tests pass with no v1
checkout present and no environment variable set.

The §3 premise this all rested on — "the C++ Vulkan layer is finished, so
linking it is free" — is now moot rather than tested. The layer is
grilly2's, and an edit to it costs one `cargo build`.
