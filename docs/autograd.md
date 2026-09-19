# Autograd design

**Status: accepted by the owner 2026-09-18, with the three recommendations in §6,
and built (all four steps of §7).** The `use_deterministic_algorithms` switch
arrives with the first op that uses an atomic kernel; none of the ops built so
far does. One refinement while building: `record` and `SavedTensor::save` are
fallible, because a saved-tensor hook may fail. Checkpoint is written in Python
on top of the core's hooks, as torch's is.

Autograd belongs in `grilly-core` (CLAUDE.md §4). This document fixes its
structure before any code: the graph, the engine, grad mode, saved tensors,
in-place safety, and gradient checkpointing (§6's first hard entry).

## 1. What must work — measured, not guessed

From `cubbyllm/` and `tests/` at CubbyLLM df3ac55:

| feature | uses | notes |
|---|---:|---|
| `loss.backward()` | 7 | always a scalar root, no `gradient=` |
| `.grad` on parameters and inputs | 12 | read after backward; `zero_grad` in 2 places |
| `requires_grad` / `requires_grad_()` | 9 | leaf inputs, freezing params |
| `torch.no_grad()` | 23 | eval, init, and in-place parameter edits |
| `torch.is_grad_enabled()` | 2 | gates checkpointing |
| `.detach()` | 4 | `x.detach().clone().requires_grad_(True)` |
| `checkpoint(..., use_reentrant=False)` | 2 | both backbones; a test depends on it (§6) |
| `clip_grad_norm_` | 1 | training loop |
| in-place ops (`mul_`, `zero_`, `add_`, `normal_`) | 6 | on parameters, under `no_grad` |

Not used anywhere: `autograd.grad`, `autograd.Function`, `retain_graph`,
`create_graph`, hooks, `retain_grad`, `inference_mode`.

The test that constrains the design most is
`test_grad_checkpoint_with_memory_flows_grads_to_memory_params`. Parameters
*used inside* a checkpointed function but *not passed to it* must get the
same gradients (atol 1e-4) as without checkpointing.

## 2. Semantics promised

torch's, for everything in §1:

- define-by-run: an op records a node only when grad mode is on **and** an
  input requires grad; otherwise recording costs nothing;
- leaves accumulate into `.grad` (first gradient is stored, later ones are
  added); non-leaves get no `.grad`;
- the graph is owned by the tensors that produced it and freed when they
  die, not by a global context;
- after `backward()` the graph's saved tensors are released; a second
  `backward()` through it raises torch's error ("Trying to backward through
  the graph a second time");
- grad mode is **thread-local**, as in torch;
- modifying a tensor in place after it was saved for backward raises torch's
  error at backward time (version counters, §3.6);
- an in-place op on a leaf that requires grad, outside `no_grad`, raises
  torch's error.

## 3. Structure

### 3.1 Where the code lives

| piece | crate | why |
|---|---|---|
| graph, engine, grad mode, saved tensors, checkpoint mechanism | `grilly-core` | every module needs it; `grilly-vsa` trains too |
| each op's backward formula | the crate that owns the op (`grilly-torch`) | an op and its derivative change together |
| `Tensor.backward`, `no_grad`, `checkpoint` Python surface | `python/grilly` | torch names, thin |

The engine accumulates gradients, so the core needs one kernel of its own:
an elementwise add. It is declared as a core `Kernel` constant like any
other; there is still one dispatch path.

### 3.2 Data model

```rust
struct Tensor { storage, shape, offset, autograd: Option<Arc<AutogradMeta>> }

struct AutogradMeta {
    requires_grad: bool,
    grad: Mutex<Option<Tensor>>,     // leaves only
    grad_fn: Option<Edge>,           // None for leaves
}

struct Edge { node: Arc<Node>, input_nr: u32 }

struct Node {
    seq: u64,                        // creation order, for a deterministic engine
    name: &'static str,              // grad_fn name, e.g. "ReluBackward0" as in torch
    next: Vec<Option<Edge>>,         // one per differentiable input
    backward: Mutex<Box<dyn Backward>>,
}

trait Backward: Send {
    /// grads w.r.t. this node's outputs -> grads w.r.t. its inputs
    fn apply(&mut self, grad_out: &[Option<Tensor>]) -> Result<Vec<Option<Tensor>>>;
    fn release(&mut self);           // drop saved tensors (retain_graph=False)
}
```

A leaf's gradient goes to an `AccumulateGrad` node that holds a **weak**
reference to the leaf's `AutogradMeta`, so the graph never keeps a leaf alive.
The version counter lives on the storage `Buffer`, so views share it.

### 3.3 Recording

An op computes its output as today, then:

```rust
autograd::record(&[&input, &weight], &mut [&mut out], || LinearBackward {
    input: SavedTensor::save(&input), weight: SavedTensor::save(&weight), ...
})
```

`record` returns immediately if grad mode is off or no input requires grad,
so the closure (and its saves) never run. That is the zero-cost path used by
inference and by everything under `no_grad`.

### 3.4 Engine

`backward(roots, root_grads)`:

1. walk the graph from the roots, counting how many edges point at each node;
2. keep a ready set ordered by `seq`, highest first (last created first),
   seeded with the roots;
3. pop a node, sum the gradients buffered for it, call `apply`, release its
   saved tensors unless `retain_graph`, and add the resulting gradients into
   the buffers of `next`; a node whose count reaches zero becomes ready;
4. `AccumulateGrad` writes into `.grad` (store first, add later), under
   `no_grad`.

It runs on the calling thread. The backward kernels go through the same lazy
batch as forward work, so a whole backward pass is recorded without a host
sync. Nothing waits until someone reads `.grad`.

The order is a pure function of the graph, so gradients are summed in the
same order every run. With deterministic kernels, repeated runs give
bit-identical gradients (but see §6, question 1).

Backward formulas run with grad mode **off**. They are written with ordinary
grilly ops, so turning it on later gives double backward (`create_graph`)
without restructuring. Until then `create_graph=True` raises
`NotImplementedError`.

The engine holds no lock while a node runs. A backward that re-enters the
engine (re-entrant checkpoint, later) therefore cannot deadlock.

### 3.5 Grad mode

A thread-local flag in `grilly-core`: `no_grad`, `enable_grad`,
`set_grad_enabled` and `is_grad_enabled` in Python, as context managers and
decorators, like torch's.

### 3.6 Saved tensors, version counters and hooks

Every tensor a formula keeps goes through `SavedTensor`:

- at save it records the storage's version counter;
- on unpack, if the version has moved, it raises torch's error ("one of the
  variables needed for gradient computation has been modified by an inplace
  operation");
- save/unpack run through a thread-local stack of **saved-tensor hooks**
  (torch's `torch.autograd.graph.saved_tensors_hooks`). Checkpointing is
  built on this mechanism, not bolted on beside it.

Every in-place op (`add_`, `mul_`, `zero_`, `normal_`, `copy_`, `__setitem__`)
bumps the version. In-place ops on a tensor that is part of a graph (a
non-leaf with `grad_fn`) raise `NotImplementedError` at first. torch
"rebases" that tensor's history; CubbyLLM's in-place uses are all on leaves
under `no_grad`, so nothing needs rebasing yet.

## 4. Gradient checkpointing (`use_reentrant=False`)

torch's non-reentrant algorithm, which is what CubbyLLM uses:

1. **Forward** runs `fn` normally, with grad on, so the graph is recorded
   as usual. Edges to parameters captured by `fn` (the memory module's
   weights) exist exactly as without checkpointing. That is what the
   CubbyLLM test needs, and why re-entrant checkpointing gets it wrong. A
   saved-tensor hook is pushed for the call. It **packs** each saved tensor
   as just an index (the activation itself is not kept, which is where the
   memory saving comes from) and records the RNG state and grad mode.
2. **Backward**: the first time any of those saved tensors is unpacked, the
   hook reruns `fn` on the original inputs, with grad on, the RNG state
   restored and saving diverted into a table. The index then looks the
   tensor up in that table. The recomputed graph is discarded; only its
   saved tensors are used.
3. torch's checks carry over: if the recomputation saves a different number
   of tensors, or different shapes, raise its `CheckpointError`.

Restoring RNG state is cheap here: a Philox generator's state is
`(seed, block)`. It is what makes dropout inside a checkpointed block
reproducible.

`use_reentrant=True` raises `NotImplementedError` pointing at
`use_reentrant=False`. So does omitting the argument: torch 2.10 treats
omission as `True` (with a warning), and CubbyLLM always passes `False`.

## 5. Rejected alternatives

- **v1's `TapeContext`** (an arena tape with `u32` buffer ids and explicit
  `begin`/`end`). Its lifetime is a context rather than the tensors, it
  cannot hold two independent graphs, it has no place for saved-tensor
  hooks (so no non-reentrant checkpoint), and it carries its own forward ops,
  the second dispatch route §5 forbids.
- **A global linear tape** (a Wengert list). Simple, but `backward()` on one
  loss would walk unrelated work, freeing the graph when tensors die is
  awkward, and nested recording for checkpoint needs graph structure anyway.
  A node per op costs one small allocation beside a GPU dispatch; an arena
  can come later if a profile says so.
- **Autograd in Python.** It would re-create v1's Python accretion, and every
  op would pay Python overhead twice.

## 6. Questions for the owner

1. **Bit-identical gradients.** The engine sums in a fixed order, but five v1
   backward kernels use float atomics (`embedding-backward`,
   `chunked-sw-attention-backward`, `conv1x1-backward-weight`,
   `convd_col2im`, `fused-layernorm-linear`), which sum in hardware order.
   Promise bit-identical gradients run to run (use non-atomic variants where
   they exist, e.g. `convd_col2im_noatomic`, and write deterministic ones
   where they don't), or promise determinism only up to those kernels, as
   torch does unless `use_deterministic_algorithms(True)`?
   *Recommendation:* follow torch: fast by default, a
   `use_deterministic_algorithms` switch that picks the non-atomic kernels.
2. **In-place ops on graph tensors**: `NotImplementedError` until a consumer
   needs history rebasing? *Recommendation:* yes.
3. **Re-entrant checkpoint**: `NotImplementedError` for now? Nothing in the
   milestone uses it. *Recommendation:* yes.

## 7. Plan, in reviewable commits

1. Grad mode, `AutogradMeta`, `Node`, `record`, engine, `AccumulateGrad`;
   backward formulas for `relu`, `add` and `linear` (v1 ships
   `activation-relu-backward` and `fnn-linear-backward`). Tests: every
   gradient compared against torch's, K-scaled tolerances as for forward; a
   diamond graph (a gradient summed from two paths); a second backward
   through a freed graph; `no_grad`; `detach`; leaf accumulation across two
   backwards.
2. Version counters and in-place ops (`add_`, `mul_`, `zero_`, `normal_`),
   including torch's error cases.
3. Python surface: `backward`, `.grad`, `grad_fn`, `is_leaf`,
   `requires_grad_`, `detach`, `no_grad` and friends.
4. Saved-tensor hooks and non-reentrant `checkpoint`. Test: a Python
   version of CubbyLLM's test above (captured parameters get the same
   gradients with and without checkpointing).

Expected size: about 700–900 lines of Rust in `grilly-core`, which keeps the
core inside §4's 3–5K budget (it is 519 lines today).
