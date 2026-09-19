# grilly2

A torch-API-compatible tensor and autograd framework on Vulkan. Open source.

You are working in a **new, empty repository**. `grilly` (v1) stays where it is
and is not modified — it is the provider of the compiled C++ Vulkan layer that
grilly2 links against, and the reference for behaviour that must not regress.
Find it as a checkout beside this one.

---

## 1. What v1 proved, and what it got wrong

v1 was an experiment. It produced a Vulkan compute layer that is **efficient and
correct** — the kernels agree with the reference implementation to float32
precision and beat the Python path 1.83× on batched work. That part succeeded.

It was also **too hard to use, and that prevented adoption.** Nobody wants to
learn a novel tensor API to get a GPU working. An experiment that produces a fast
engine and a usability failure is a result, not a write-off: it tells you exactly
which half to keep.

**So grilly2 keeps the engine and replaces the interface with one every ML
developer already knows.** torch-API compatibility is not a feature of this
project; it is the fix for the specific thing that failed.

That yields one standing design rule:

> **Every time grilly2 is tempted to expose something novel, that is the v1
> failure coming back.** If torch has a name for it, use torch's name, torch's
> argument order, and torch's semantics. Novelty belongs in the modules
> (`grilly-vsa`, `grilly-snn`), never in the core.

## 2. The position

torch already runs wherever torch runs. The reason to adopt a torch-compatible
Vulkan backend is the hardware torch serves badly: **consumer AMD and Intel GPUs,
and Windows.** torch's AMD path is ROCm, which is Linux-only. The development
machine here is an RX 6750 XT with no CUDA and no ROCm.

Nothing in `tinygrad`, `candle`, `burn`, `ggml` or `llama.cpp` is a torch-API
drop-in that *trains* on a consumer AMD card under Windows.

**The v1 claim is "the torch API, on hardware torch will not train on" — not "a
torch replacement".** Scope is not shortfall. Do not widen the surface to chase a
comparison with torch.

## 3. Kept, and rewritten

**Kept, linked, not reimplemented:**

- the C++ Vulkan implementation in v1's `cpp/` — `CMakeLists.txt` already builds
  **`grilly_core_lib` as a STATIC library** separate from the Python module, so a
  second consumer links it with no restructuring;
- the **273 compiled SPIR-V modules**. SPIR-V is toolchain-independent. Do not
  recompile or port shaders;
- v1's `cpp/python/` binding layer as an **inventory and a seam map** (§4).

**Rewritten here:** the framework layer above the GPU code — tensor, autograd,
dispatch orchestration, `nn`, `optim`, and the torch-compatible surface. In v1
that layer is ~72,000 lines of Python that accreted. It is what made v1 hard to
use, and it is the only thing grilly2 is rewriting.

**Core language: Rust, pending the measurement in §11.** The reasoning: v1 failed
by accretion, and §4's constraint — `grilly-core` stays near 3–5K lines or the
split is cosmetic — is a matter of discipline in C++ and a compile error in Rust.
Crate boundaries make the thing this project is trying to enforce enforceable.
Second, this layer is buffer lifetimes and resource ownership across a GPU
boundary, which is where Rust pays; the kernels, where it would not, are staying
in C++. Python bindings via PyO3.

The cost is honest and accepted: three languages in a solo project — C++ for
Vulkan, Rust for the core, Python for the API.

**Linking detail.** `grilly_core_lib` exposes a **C++ ABI, not a C one** —
`extern "C"` and the usual export macros appear nowhere in v1's own sources (every
hit is vendored under its build tree). So a Rust core needs the `cxx` crate or a
thin `extern "C"` shim over only the entry points it uses. That shim is the first
deliverable and is a design decision, not plumbing — it is the one chance to fix
what the core may ask the GPU layer for.

**This decision reverses if the C++ Vulkan layer does not stay frozen.** The
premise is that it is finished and linked. If it turns out to need regular edits,
every change crosses the boundary twice and C++ wins on friction alone.

**Do not vendor dependencies into this repo.** v1 carries ~8M lines under
`external/` and `third_party/`. That is a large part of what made it unusable.

## 4. The module split — the seams already exist

v1's `cpp/python/` is already divided by domain: 26 `bindings_*.cpp` files and 5
`.inc` files over a shared `bindings_core.cpp` / `.h`. **The boundaries are not a
refactor to argue about; they are already drawn, in C++.** Follow them.

| package | v1 binding files that define it |
|---|---|
| `grilly-core` | `bindings_core` — device, memory, dispatch, shader registry, tensor, autograd |
| `grilly-torch` | `bindings_autograd`, `bindings_optim`, `bindings_linear`, `bindings_conv`, `bindings_normalization`, `bindings_activations`, `bindings_loss`, `bindings_attention`, `bindings_pooling`, `bindings_mingru`, `tensor_ops_bindings.inc`, `elementwise_bindings.inc` |
| `grilly-vsa` | `bindings_vsa_lm`, `vsa_explore_bindings.inc`, `block_ops_bindings.inc` |
| `grilly-snn` | `bindings_snn`, `bindings_eggroll`, `bindings_bandit` |
| `grilly-huggingface` | `bindings_siglip`, `bindings_distillation`, `bindings_perceiver` |

**The one thing that decides whether the split is real:** `grilly-core` must stay
thin. In v1, 42% of the Python layer was shared across all domains — four packages
on a fat shared base moves the coupling, it does not remove it. Hold yourself to:
*what is the minimum `grilly-core` that `grilly-vsa` alone needs?* Near 3–5K lines
and the split is real; near 30K and it is cosmetic. v1's `device.cpp` (585),
`buffer_pool.cpp` (549) and `pipeline_cache.cpp` (372) suggest the genuine core is
~1.5K lines of orchestration.

## 5. The structural fault to fix

v1 has **no single dispatch spine**. `.spv` names are known to individual op
families and headers — 25 files reference shaders directly — even though
`pipeline_cache.cpp`, `compute_backend.h` and `op_graph.h` all exist to be that
route. More than one way to reach the GPU, and every op added makes the second
path more expensive to remove.

**grilly2's core requirement: one dispatch path.** Shader registry → pipeline
cache → descriptor/bind → submit. Every op goes through it. No op knows a `.spv`
filename. If you find yourself adding a second route, stop and fix the first.

## 6. Conformance target and definition of done

The API surface is torch's. Milestone 1 is a real, already-green workload rather
than a reading of torch's docs:

| | |
|---|---:|
| distinct torch symbols used by the CubbyLLM repo | **110** |
| distinct torch symbols used by its `cubbyllm/` package | **59** |
| distinct `nn.Module` classes | **13** |

The 13: `Conv1d`, `Embedding`, `GELU`, `GRU`, `Identity`, `LayerNorm`, `Linear`,
`Module`, `ModuleList`, `Parameter`, `Sequential`, `Tanh`. The heavy tail is
ordinary — `no_grad`, `randn`, `Tensor`, `from_numpy`, `cat`, `arange`, `zeros`,
`stack`, `gather`, `bmm`, `cross_entropy`, `normalize`, `Adam`.

For scale, v1 already registers **107 pybind11 ops** — the target is the same
order as what is bound today. Prune against that list; not all 107 belong in a
public v1.

**Definition of done for milestone 1:** the CubbyLLM package test suite passes
under `import grilly as torch`. That suite exists, is maintained, and is green on
torch today. Do not build a new conformance harness before using the one that
already exists.

**The four hard entries, for anyone:**

- `torch.utils.checkpoint.checkpoint` — gradient checkpointing. Not optional for a
  2B model on 12 GB. Re-entrant autograd is the awkward part of any engine.
- `F.scaled_dot_product_attention`, `torch.nn.attention.flex_attention` —
  flash-attention-class kernels. v1 ships `attention-*.spv`, so this may be most
  of the way there already.
- `torch.einsum` — a general implementation is real work. Ten call sites in the
  consumer repo, five in the package; specialise the patterns in use first.
- `torch.Generator` / `manual_seed` — bit-exact reproducibility across a backend
  swap is where "drop-in" quietly stops being true. Decide early whether to
  promise it, because users will test it.

## 7. Behaviour that must not regress

The block-code VSA path has two downstream consumers. Measured on v1:

| | |
|---|---|
| bridge `blockcode_bind` vs the Python `BlockCodeOps.bind` | agree, max abs diff **1.19e-07** |
| bind → unbind exact block recovery, both paths | **100%** |
| speed at batch 1 / 64 / 512 | 1.01× / **1.74×** / **1.83×** |

## 8. Traps — read before measuring anything

Every one of these produced a confidently wrong conclusion during the analysis
behind this document. They cost hours.

1. **A `numpy` backend reading does not mean "no Vulkan tier".** It usually means
   the package is not importable from the interpreter you are testing in. Check
   installation before concluding anything about capability.
2. **Block codes are `(n, k*l)` batches of ONE-HOT vectors.** Not flat `(k*l,)`,
   not dense Gaussians. Feeding dense random vectors to `blockcode_bind` produces
   garbage that looks exactly like a semantics bug. v1's downstream consumer
   `ops/block_codes.py` has the correct reshape convention; copy it.
3. **`blockcode_*` and `vsa_*` are different algebras and must stay that way.**
   `blockcode_*` is sparse one-hot float32 taking `num_blocks` / `block_size`;
   `vsa_*` is bipolar int8 taking neither. Not two implementations of one thing.
   Do not unify them, and never let a capability check for one gate the other —
   v1's `_detect_backend()` gated the whole tier on
   `hasattr(bridge, "blockcode_bind")`, which is how the confusion started.
4. **The bridge returns `None` to mean "declined, fall through".** It does not
   raise. A caller that only catches exceptions will silently use a `None`.
5. **The GPU win is batch-only.** At batch 1 the tiers tie. Never quote a batched
   speedup without the batch size beside it.
6. **Counting symbols is not counting calls.** `VkDescriptorSet` in a signature is
   a type, not a device call. A regex that conflates them invents a layering
   violation that is not there.

**Standing rule: when a measurement contradicts what the code appears to say,
assume the measurement is wrong until proven otherwise.** Print shapes, dtypes,
and which interpreter you are in, before drawing a conclusion.

## 9. Repository conventions

- **Never put absolute local paths in committed files** — no drive letters, no
  user directories, no virtualenv names. Describe locations generically: "a
  `grilly` checkout beside this one", "the compiled extension at the repo root".
  Resolve by environment variable first, then a sibling-directory convention. v1
  hard-codes an absolute path to a data file on a drive that is no longer
  attached; do not reproduce that.
- Clean working tree: no `build/`, no vendored sources, no generated artifacts
  committed.
- Every op needs a test comparing it against torch on CPU within tolerance, and
  the tolerance recorded.
- Commit messages: what changed and why; name the measurement if one drove it.

## 10. Working agreement

- Small, reviewable commits. The owner reviews before anything merges.
- When a decision has two reasonable answers — core language, shim vs `cxx`, which
  ops make v1 — **write the options and the trade-off, and ask.** Do not pick
  silently and build on it for a day.
- Report progress against §6: how many of the 110 symbols are green.
- Usability is the acceptance criterion, not an afterthought. If a thing is
  awkward to call, that is a bug of the same severity as a wrong number — it is
  the failure mode that killed v1.

## 11. First task — size the boundary, then confirm the language

Before writing any core code, answer one question with a measurement:

> **What is the minimum surface `grilly-core` needs from the GPU layer, and how
> wide is it?**

Read v1's `cpp/python/bindings_core.h` and `bindings_core.cpp`, and whichever
`bindings_*.cpp` the first milestone actually needs (§6's list). Produce:

1. the list of entry points the core must call, with signatures;
2. which types cross the boundary — and in particular whether `Device`,
   `TapeContext` and descriptor handles have to be held on the Rust side or can
   stay opaque;
3. how much of the state is per-call versus long-lived.

**The decision rule.** If the surface is narrow and mostly opaque handles, the
shim is small and **Rust stands** (§3). If it is wide and stateful — descriptor
handles crossing constantly, lifetimes managed on both sides — then the bridge
stops being a day's work and starts being the project, and **C++ is the better
call**. Report the finding and the recommendation; do not start the core until
this is settled.

The language choice in §3 was made from a symbol count taken from outside the
headers. This task is what turns it into a decision.
