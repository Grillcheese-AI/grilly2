# Kernel fusion

**Decision (owner, 2026-09-19): F2.** grilly2 keeps grilly v1's fusion design
and machinery and generates the kernels itself. The entry point is torch's
name for it, `torch.compile`. The options are in `docs/flex-attention.md`
("v1's attention kernels and shader fusion").

## What is kept from v1

v1's `ShaderFusionEngine` (`cpp/src/shader_fusion.cpp`) did four things,
and grilly2 keeps all four:
- generate one GLSL compute shader for a graph of elementwise ops, with
  intermediates in registers;
- compile it at runtime with shaderc;
- register it in `PipelineCache`;
- reuse it by a key.

The shaderc is the one v1 already links from the Vulkan SDK. The shim's
`gc_compile_shader` compiles with the build's glslc settings (`-O`,
Vulkan 1.2) and calls v1's `PipelineCache::loadSPIRV`. A generated kernel
then goes through the same dispatch spine as every other kernel:
registry, pipeline cache, bind, submit.

## What is different

v1's code generator knows five ops, and it silently skips any other op in
a graph. grilly2 generates calls to `g2_unary` and `g2_binary`, the
functions in `g2-common.glsl` that the unfused `g2-unary` and `g2-binary`
kernels also run. Every elementwise op therefore has one definition, and
the accuracy work on it applies to fused code too: Cephes exp and tanh,
sin and cos to 1 ULP, square-and-multiply pow, torch's NaN rules.

Every intermediate is `precise`, so the compiler cannot contract or
reassociate across ops. A fused graph therefore returns what the same
ops return one kernel at a time, bit for bit. The exception is a NaN's
sign and payload, which IEEE 754 leaves unspecified. Tested in
`crates/grilly-torch/tests/fusion.rs` for every op, with broadcasting and
several outputs.

Scalars such as a learning rate or a dropout scale are runtime
parameters, not literals. A graph compiles once, whatever values it
later runs with. Kernels are cached per device by their generated
source.

## Layers

1. `grilly_core::fusion::Graph`: inputs, parameters, unary and binary
   nodes, and one or more outputs. `run(inputs, params, shape)` launches
   one kernel.
2. `torch.compile` (next). Inside a compiled function, float elementwise
   ops build a lazy graph. The first use that needs data (a matmul, a
   reduction, `.item()`) compiles and runs the graph as one kernel.
3. Autograd (after that). The graph's derivatives are themselves an
   elementwise graph, so the backward is a fused kernel too, followed by
   sums over broadcast dims.

## Measured

On the RX 6750 XT, the tanh-approximation GELU over 16M elements takes
28.0 ms as 8 separate kernels and 3.1 ms as 1 fused kernel, with the same
result.

## `torch.compile` (F2.2): lazy capture, forward only

`grilly.compile` builds no graph of its own and breaks nothing into
sub-graphs; it turns on lazy capture (`python/grilly/_lazy.py`) for the
duration of a call and materializes whatever comes out. Autograd is
untouched here — a captured graph is never recorded for backward, and
capture itself declines under grad mode (see below) — so gradients through
a compiled function are exactly today's, at the cost of falling back to
eager wherever a tensor requiring grad reaches an elementwise op. Fusing
the backward pass is item 3 of "Layers", above, and is not built yet.

**Capture rule.** Inside a compiled callable, `python/grilly/_ops.py`'s
`_unary`/`_binary` — the funnel every float elementwise op goes through —
return a `LazyTensor` instead of launching a kernel, when every tensor
operand is float32 (a `LazyTensor` counts as float32) **and** it is not the
case that grad mode is enabled and some operand requires grad. Otherwise
they run eagerly, exactly as before capture existed: any `LazyTensor`
operand is materialized for free, because `LazyTensor` is a `Tensor`
subclass and `_unary`/`_binary` read `x._t`.

**`LazyTensor`.** Holds one node of the expression — a unary or binary op,
its operand(s), and its scalar parameters — not a value. `.shape`, `.dim()`,
`.numel()`, `.dtype`, `.device`, `.requires_grad` (always `False`),
`.is_leaf` (always `True`, as for any tensor that doesn't require grad) and
`.grad_fn` (always `None`) all answer from the node, without materializing.
Every other op in grilly — matmul, a reduction, indexing, `.item()`,
`.numpy()`, printing, an in-place op — reads `._t`, which is a property: the
first read walks the expression into one `grilly_core::fusion::Graph`
(deduping real tensors by identity, and folding in any operand that is
itself an unmaterialized `LazyTensor`, so a chain that is never forced early
becomes one kernel end to end) and calls the `_C.fused` binding. An operand
that *was* already materialized independently — something forced it early,
e.g. a matmul mid-chain — is used as a real input instead of being
recomputed. The result is cached on that `LazyTensor`; nothing else needs a
hook.

At a compiled callable's exit, every `LazyTensor` still lazy in the return
value (walking tuples, lists and dicts) is materialized, so callers only
ever see real tensors.

**In-place writes.** A lazy node records the version counter of each real
tensor it reads. If one of them is written in place before the node runs
(`y = x * 2; x.add_(1); return y`), materializing raises instead of reading
the new values. torch.compile would return the old-x result here; grilly
refuses loudly rather than return a wrong one. The common optimizer shape
`p.add_(expr of p)` is unaffected: the argument is materialized before `p`
is written.

**Size.** An expression past 64 ops (`_lazy.MAX_NODES`) is materialized
before it is used again. That bounds recursion, generated-shader size and
compile time, and stops `x = x + x` in a loop from doubling each step.

**Scalars stay parameters.** A Python float threaded through as a
learning rate, a dropout scale, or an Adam `beta`/`eps` becomes a
`Node::Param`, not a literal in the generated GLSL (`fusion.rs`, already
true before `torch.compile`). Calling the same compiled function again with
different scalars reuses the cached kernel — `_C.generated_kernel_count()`
does not grow — as long as the *op sequence* is unchanged.

**The API.** `grilly.compile(model=None, *, fullgraph=False, dynamic=None,
backend="inductor", mode=None, options=None, disable=False)` matches
torch's signature and is usable bare, as a bare decorator, or as a
decorator with arguments, on a function or an `nn.Module`. Only `backend`
does anything: `"eager"`/`"aot_eager"` run the callable unchanged (capture
off, but `grilly.compiler.is_compiling()` is still `True` inside, matching
torch); an unrecognised name raises `RuntimeError`, as torch's does.
`fullgraph`, `dynamic`, `mode` and `options` are accepted and have no
effect — there is no graph-breaking tracer here to hand them to.
Compiling an `nn.Module` returns an `OptimizedModule` holding it as the
child `_orig_mod`, as torch's does: parameters are shared, `state_dict()`
keys carry torch's `"_orig_mod."` prefix, and other attributes are read
from the original. `Module.compile
(*args, **kwargs)` compiles `self.forward` in place instead (torch >= 2.2):
`self` keeps its own type. `grilly.compiler` provides `is_compiling()`,
`disable(fn=None, recursive=True)` and `reset()` (Hugging Face's usual
hooks); `disable`'s `recursive` argument is accepted but not distinguished
from the default, since suspending capture already covers every nested
call grilly makes, with no bytecode-level tracer to scope it to one frame.

**What was decided differently from a straight torch reading.** Comparisons
(`eq`, `lt`, ...) never reach `_binary`'s capture check at all: they are
bound to a separate `_C.compare` entry point that always returns a bool
tensor, so there was nothing to gate. `_binary`'s comparison codes
(`Binary::Eq` etc.) exist for library code that wants a float32 0/1 mask
(dropout); nothing in the current op set reaches them through `_unary`/
`_binary`, so this is a forward-compatibility note, not a behaviour change.
