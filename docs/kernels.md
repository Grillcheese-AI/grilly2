# grilly2's own kernels

**Decision (owner, 2026-09-19):** v1's 273 SPIR-V modules covered
neural-network-level ops (activations, layernorm, softmax, attention,
cross-entropy, embedding, Adam) but no general arithmetic: no elementwise
multiply, subtract, divide or pow, no scalar operands, no broadcasting, no
elementwise exp/log/sqrt/clamp, and one reduction. grilly2 therefore owns a
small set of generic kernels. They live in `crates/grilly-core/shaders/`, are
compiled at build time with `glslc` from the Vulkan SDK, and never committed
as SPIR-V.

**Update (owner, 2026-09-23): there is no longer a second set.** grilly2
dispatched exactly six of v1's 273, and when the link to v1 was cut those
six came over as `g2-relu`, `g2-relu-backward`, `g2-add-inplace`,
`g2-linear`, `g2-linear-backward` and `g2-residual`, compiled from their
GLSL by the same `build.rs`. v1 compiled them at `--target-env=vulkan1.0`
without `-O` and grilly2 compiles them at `vulkan1.2` with `-O`; the op
tests against torch are unchanged, so the optimizer moved nothing. v1 is
untouched.

| kernel | does | covers |
|---|---|---|
| `g2-unary` | `f(x; a, b)` per element, in place or not | neg, abs, sign, exp, log, sqrt, rsqrt, tanh, sigmoid, sin, cos, scalar + - * / pow, clamp, gelu, isfinite/isinf/isnan |
| `g2-binary` | `op(a, b)` with torch broadcasting (up to 6 dims) | + - * / pow, maximum, minimum, comparisons |
| `g2-reduce` | sum / mean / max / min along a strided axis, a thread per (output, chunk of the axis); few outputs split the axis and finish in a second pass | sum, mean, amax, amin over non-last dims, the backward of broadcasting, bias / weight gradients |
| `g2-reduce-rows` | the same along contiguous rows, a workgroup per row (or per chunk of a long row, finished by a second pass) | the last-dim reductions under softmax and log_softmax; full sums |
| `g2-copy` | strided gather/scatter | transpose, permute, expand, slicing, cat, stack, `__setitem__` |
| `g2-cast` | dtype conversion with torch's rules | `.to(dtype)`, `.long()`, `.float()`, `.bool()` |
| `g2-binary-int` | integer / bool ops with broadcasting | int arithmetic, bitwise and logical ops, int comparisons |
| `g2-where` | `where(cond, a, b)` with broadcasting | where, masked_fill, bool-mask assignment |
| `g2-reduce-int` | integer sums and extremes along one axis | int sum, amax, amin, all, any |
| `g2-index` | gather along one dim by an int64 index | gather, index_select, advanced indexing, embedding |
| `g2-index-add` | backward of `g2-index`, one thread per destination, fixed order | the gradient under `use_deterministic_algorithms(True)` |
| `g2-index-add-atomic` | backward of `g2-index`, one thread per gradient element | the gradient by default (fast; last bits vary with duplicate indices) |
| `g2-argreduce` | index of the first extreme along one axis (NaN is the extreme) | argmax, argmin, `max(dim)`, `min(dim)` |
| `g2-argreduce-rows` | argmax / argmin along contiguous rows of 256 or more, a workgroup per row, torch's tie and NaN rule | argmax, argmin, `max(dim)`, `min(dim)` over long rows; greedy decoding |
| `g2-sort-init`, `g2-sort-step`, `g2-sort-take` | bitonic sort of every row (padded to a power of two), float32 or int64 keys, ties by position | sort, argsort, topk |
| `g2-scatter` | `dst[c with c[dim] := index[c]] = src[c]`, any dtype | scatter, scatter_, the input gradient of scatter |
| `g2-scan` | inclusive scan along one axis, one thread per line, forward or reversed | cumsum (and its gradient), logcumsumexp (and its gradient) |
| `g2-layer-norm`, `g2-layer-norm-backward` | layer_norm forward and input gradient, a workgroup per row; (mean, rstd) saved per row | F.layer_norm, nn.LayerNorm (dw and db are column sums through g2-reduce) |
| `g2-flash-fwd` | block-sparse attention forward over 32x32 tiles, online softmax, writes O and lse | flex_attention, scaled_dot_product_attention (bool / no mask, no dropout) |
| `g2-flash-bwd-kv`, `g2-flash-bwd-q` | its backward as two kernels (dK, dV per key tile; dQ per query tile), no atomics | the gradients of both |
| `g2-flash-tiles` | a bool mask's bit mask and per-tile key / query lists, on the device | create_block_mask, SDPA masks |
| `g2-bmm` | batched `op(A) @ op(B)`, optional transposes, broadcast batch dims; 16x16 tiles through shared memory | matmul / bmm with batch dims, and both gradients (no transpose copies) |
| `g2-quantize-q8` | per-row int8 of a float32 weight, torchao's formula, with correctly rounded division (`div_rn`) | `quantize_(model, Int8WeightOnlyConfig())` (grilly.infer) |
| `g2-linear-q8` | `(x @ q.T) * scale + bias` with the int8 weight unpacked in the kernel; 64 threads per output feature, 16-byte weight loads, up to 8 rows of x per weight read | `F.linear` on an `Int8Tensor` weight |
| `g2-linear-q8-tiled` | the same for many rows: a 64x64 output tile per workgroup, x and the unpacked weight staged in shared memory, 4x4 per thread | `F.linear` on an `Int8Tensor` once the shape makes enough tiles |
| `g2-dequantize-q8` | `q * scale` back to float32 | any other use of an `Int8Tensor`; its linear's input gradient |
| `g2-rms-norm` | `x * rsqrt(mean(x^2) + eps) * w`, a workgroup per row, optionally adding a residual first and writing the sum out too; compensated sum, correctly rounded mean and reciprocal | `F.rms_norm`, `nn.RMSNorm`, a decoder layer's add-then-normalize |
| `g2-rope` | rotary embedding of `heads` heads read from one projection's columns, written as (batch, heads, T, dim) | `grilly.infer` decode and prefill; also straight into the KV cache |
| `g2-swiglu` | `silu(gate) * up`, each read from its own source and column (two projections, or the halves of a fused one) | the SwiGLU MLP of `grilly.infer.models` |
| `g2-quantize-q4` | group-wise asymmetric int4 of a float32 weight, torchao's formula, correctly rounded division | `quantize_(model, Int4WeightOnlyConfig())` |
| `g2-linear-q4` | `x @ dequantize(q)^T + bias`, a scale and zero point per group, unpacked in the kernel; 64 threads per output feature, 32 levels per 16-byte load | `F.linear` on an `Int4Tensor` weight |
| `g2-dequantize-q4` | `q * scale + min` back to float32 | any other use of an `Int4Tensor`; its linear's input gradient |
| `g2-attn-decode` | one query token against a KV cache's filled positions, a workgroup per (batch, head), grouped-query attention by index | `grilly.infer.cache.KVCache.attend` |
| `g2-attn-decode-split`, `g2-attn-decode-merge` | the same over a slice of the cache per workgroup, returning each slice's (max, sum, unnormalized output), then combining them ("flash decoding") | `KVCache.attend` once the cache is long enough to split |
| `g2-sample` | next-token sampling, a workgroup of 1024 per row: repetition penalty, temperature, top-k (radix select), top-p (binary search over values), inverse-CDF draw; no sort; candidates in shared memory when they fit | `grilly.infer.sampling`, `torch.multinomial` |

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

The sort runs log2(p)(log2(p)+1)/2 compare-exchange dispatches for rows
padded to p, all recorded in one batch (120 for a 32768-wide row). Its
order is total (equal values keep their original order), so `sort` is
always stable and `topk` breaks ties toward the lower index. torch's default
sort is not stable, so tests compare against `sort(stable=True)`.

`g2-scan`'s cumsum is Kahan-compensated: within 0.5 eps * cumsum(|x|) of a
float64 reference, as torch's CPU cumsum (which accumulates in double).
logcumsumexp carries the running max `m` and a compensated
`S = sum exp(x_j - m)`, giving `m + log S`: its absolute error stays within
about eps * (|m| + log n + 1) at any length. Chaining `logaddexp` step by step
was measured at 170 ULP after 8192 steps and was replaced.

`g2-bmm` accumulates each output's K products in order with fma: within
0.41 * sqrt(K) * eps * (|A| @ |B|) of float64 (measured), the bound the linear
tests use. It runs 64 products of 512^3 in 15 ms (1.1 TFLOP/s) on the
RX 6750 XT. 2-D @ 2-D still goes through v1's linear kernel.

Reductions pick a path by shape (`grilly_core::generic::reduce`): the rows
kernel when the reduced dim is last; otherwise the columns kernel, which
splits the reduced axis into chunks until there are ~16K threads
(neighbouring threads take neighbouring columns, so reads stay coalesced).
An earlier version copied few-output strided reductions to rows first;
that transpose copy cost 4.2 ms for (4096, 2048), more than the
reduction. Adjacent reduced
dims merge into one pass. The rows kernel sums with Kahan per thread and a
fixed two-sum tree across the workgroup: deterministic, and within
0.14 * eps * sum(|x|) of float64 (measured). Before it, one thread per output
made last-dim reductions uncoalesced (softmax over (8,8,512,512): 25.6 ms,
now 9.5) and put a whole-tensor sum on one thread (16M elements ran past the
fence timeout; now 0.1 ms).

Integer reductions took the same wrong turn and kept it longer.
`g2-reduce-int` gave each output one invocation, which is a single serial
thread whenever there is one long row: `any()` over a 127996-element mask
ran in **7.8 ms**, against 0.016 ms for the same-length float `sum()`, at
61 ns per element. `g2-reduce-int-rows` splits a contiguous row across a
workgroup the way the float path does — **0.032 ms**, 244x — and integer
partials are exact, so it needs none of the float version's compensated
pair. The columns kernel still takes every other axis.

## What is exact, and what is two ULP

grilly2's arithmetic is not uniformly correctly rounded, and the split is
worth knowing before writing `==` in a test. Measured against numpy over
200000 random float32 pairs on an RX 6750 XT:

| op | max ULP from numpy |
|---|---:|
| add, subtract, multiply | **0** (bit-exact) |
| divide | **2** |
| reciprocal, sqrt | **1** |

This is the hardware behaving to spec, not a kernel bug: Vulkan requires
`OpFDiv` to be within 2.5 ULP and `OpSqrt` within 3.0, and only add,
subtract and multiply are required to be correctly rounded. CUDA's `/` is
correctly rounded by default, so **an expression containing a divide will
not match torch bit for bit, and one built only from + - * will.**

Two consequences:

- a grilly-against-torch test (CLAUDE.md §9 case (b)) over anything with a
  divide in it cannot assert `0`, and the reason is this table rather than
  "float32";
- a grilly-against-grilly test (case (a)) is unaffected — the same kernel
  on the same hardware gives the same answer every time, whatever that
  answer's last bit is. Capture and replay stay bit-exact.

Where an exact result matters and the value is a function of a small
integer, compute the table on the host and gather from it. CubbyLLM's
repetition penalty does this: `penalty ** count` from a numpy-built table
indexed by the count, rather than a device `pow`, because the argmax over
128k logits can see an ulp.
