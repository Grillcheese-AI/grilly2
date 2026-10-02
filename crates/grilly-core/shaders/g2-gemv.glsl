#version 450
#extension GL_GOOGLE_include_directive : require
// `out = x @ W^T (+ b)` for a *few* rows: the decode shape, where x is one
// token and W is the whole weight.
//
// `g2-linear` tiles 16x16, which is right when there are rows to fill the
// tile and wrong when there is one: fifteen of every sixteen tile rows do
// nothing, and the kernel runs at a fraction of memory bandwidth on work
// that is purely a weight read. Measured on an RX 6750 XT at M=1, against
// the card's 432 GB/s: (512,512) **62 GB/s**, (2048,512) 136, (127996,512)
// 193. A decode step at d512/L8 spends 2.3 ms of its 5.5 ms token there.
//
// So: one workgroup per output feature, reading that weight row once,
// coalesced across the workgroup, and dotting it against every one of the
// M input rows while it is in registers. Weight traffic — the thing that
// dominates — is paid once for all M, which is the same reason a batched
// decode is cheap.
//
// The accumulator loop runs to a *constant* MAX_M rather than to `rows`,
// with out-of-range rows reading zero, so the compiler can unroll it and
// keep the accumulators in registers instead of spilling a dynamically
// indexed array to scratch. The wasted FMAs are free: this kernel is
// bandwidth-bound on W, not arithmetic-bound.
//
// Summation order is fixed — each thread sums its strided slice with fma,
// then a fixed tree across the workgroup — so it is deterministic. It is
// *not* the tiled kernel's order, so the two disagree within the error
// bound of a length-K dot product; `tests/python/test_gemv.py` measures
// both against float64 and prints the ratio.

layout(local_size_x = 64) in;
#include "g2-common.glsl"

const uint THREADS = 64u;
/// Rows this kernel serves. Above it the tiled kernel has tiles to fill.
const uint MAX_M = 8u;

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer W { float wt[]; };
layout(std430, binding = 2) readonly buffer B { float bias[]; };
layout(std430, binding = 3) writeonly buffer O { float o[]; };
layout(push_constant) uniform P { uint rows; uint k; uint n; uint has_bias; } p;

shared float partial[THREADS * MAX_M];

void main() {
    uint col = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (col >= p.n) return;                 // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;

    float acc[MAX_M];
    for (uint m = 0u; m < MAX_M; ++m) {
        acc[m] = 0.0;
    }
    uint row = col * p.k;
    for (uint e = tid; e < p.k; e += THREADS) {
        float weight = wt[row + e];         // coalesced: neighbours read neighbours
        for (uint m = 0u; m < MAX_M; ++m) {
            float v = m < p.rows ? x[m * p.k + e] : 0.0;
            acc[m] = fma(v, weight, acc[m]);
        }
    }
    for (uint m = 0u; m < MAX_M; ++m) {
        partial[tid * MAX_M + m] = acc[m];
    }
    barrier();
    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            for (uint m = 0u; m < MAX_M; ++m) {
                partial[tid * MAX_M + m] += partial[(tid + stride) * MAX_M + m];
            }
        }
        barrier();
    }
    if (tid == 0u) {
        float b = p.has_bias != 0u ? bias[col] : 0.0;
        for (uint m = 0u; m < p.rows; ++m) {
            o[m * p.n + col] = partial[m] + b;
        }
    }
}
