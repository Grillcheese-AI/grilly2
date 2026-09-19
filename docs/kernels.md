# grilly2's own kernels

**Decision (owner, 2026-09-19):** v1's 273 SPIR-V modules cover
neural-network-level ops (activations, layernorm, softmax, attention,
cross-entropy, embedding, Adam) but no general arithmetic: no elementwise
multiply, subtract, divide or pow, no scalar operands, no broadcasting, no
elementwise exp/log/sqrt/clamp, and one reduction. grilly2 therefore owns a
small set of generic kernels. They live in `crates/grilly-core/shaders/`, are
compiled at build time with `glslc` from the Vulkan SDK, and never committed
as SPIR-V. v1 is untouched; CLAUDE.md's "do not recompile or port shaders"
applies to v1's modules.

| kernel | does | covers |
|---|---|---|
| `g2-unary` | `f(x; a, b)` per element, in place or not | neg, abs, sign, exp, log, sqrt, rsqrt, tanh, sigmoid, sin, cos, scalar + - * / pow, clamp, gelu, isfinite/isinf/isnan |
| `g2-binary` | `op(a, b)` with torch broadcasting (up to 6 dims) | + - * / pow, maximum, minimum, comparisons |
| `g2-reduce` | sum / mean / max / min along one axis | sum, mean, amax, amin, the backward of broadcasting |
| `g2-copy` | strided gather/scatter | transpose, permute, expand, slicing, cat, stack, `__setitem__` |
| `g2-cast` | dtype conversion with torch's rules | `.to(dtype)`, `.long()`, `.float()`, `.bool()` |
| `g2-binary-int` | integer / bool ops with broadcasting | int arithmetic, bitwise and logical ops, int comparisons |
| `g2-where` | `where(cond, a, b)` with broadcasting | where, masked_fill, bool-mask assignment |
| `g2-reduce-int` | integer sums and extremes along one axis | int sum, amax, amin, all, any |
| `g2-index` | gather along one dim by an int64 index | gather, index_select, advanced indexing, embedding |
| `g2-index-add` | backward of `g2-index`, one thread per destination, fixed order | the gradient under `use_deterministic_algorithms(True)` |
| `g2-index-add-atomic` | backward of `g2-index`, one thread per gradient element | the gradient by default (fast; last bits vary with duplicate indices) |

All are prefixed `g2-` so they cannot shadow a v1 kernel. Every tensor is
passed as its whole storage buffer plus an element offset, so views never
need aligned binding offsets.

Accuracy is measured against a float64 reference and bounded in
`crates/grilly-torch/tests/generic_kernels.rs`: exact ops are bit-identical to
torch, transcendental ops are within 2 ULP (GPU-native `exp` was 14 ULP off, so
`exp`/`tanh` use Cephes-style implementations), and sums use Kahan
compensation. Non-integer `pow` is the known weak spot (32 ULP bound).
`sin`/`cos` are within 1 ULP for |x| <= 8192 (Cephes reduction); larger
arguments would need Payne-Hanek reduction.

`g2-index-add-atomic` adds floats with a compare-and-swap loop on uint
atomics rather than `GL_EXT_shader_atomic_float`: on the RX 6750 XT driver
stack, float `atomicAdd` puts every add at flat index 0. v1 recorded the same
failure in `cpp/src/ops/loss.cpp`, and it was measured again here.
