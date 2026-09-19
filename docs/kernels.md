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
| `g2-unary` | `f(x; a, b)` per element, in place or not | neg, abs, sign, exp, log, sqrt, rsqrt, tanh, sigmoid, scalar + - * / pow, clamp, gelu |
| `g2-binary` | `op(a, b)` with torch broadcasting (up to 6 dims) | + - * / pow, maximum, minimum, comparisons |
| `g2-reduce` | sum / mean / max / min along one axis | sum, mean, amax, amin, the backward of broadcasting |
| `g2-copy` | strided gather/scatter | transpose, permute, expand, slicing, cat, stack, `__setitem__` |

All are prefixed `g2-` so they cannot shadow a v1 kernel. Every tensor is
passed as its whole storage buffer plus an element offset, so views never
need aligned binding offsets.

Accuracy is measured against a float64 reference and bounded in
`crates/grilly-torch/tests/generic_kernels.rs`: exact ops are bit-identical to
torch, transcendental ops are within 2 ULP (GPU-native `exp` was 14 ULP off, so
`exp`/`tanh` use Cephes-style implementations), and sums use Kahan
compensation. Non-integer `pow` is the known weak spot (32 ULP bound).
