#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// int64 sum / max / min over contiguous rows: one workgroup per (row,
// chunk), the way g2-reduce-rows does it for floats.
//
// g2-reduce-int gives each output ONE invocation, which is right when
// there are many outputs and wrong when there is one long row: `any()`
// over a 127996-element mask reduced in a single thread and took 7.8 ms,
// against 0.016 ms for the same-length float `sum()`. Integers need none
// of the float version's machinery — no Kahan pair, no compensated tree,
// because every partial is exact — so the partial buffer is plain int64
// and the second pass is the same kernel over it.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { uint src[]; };
layout(std430, binding = 1) buffer O { uint dst[]; };
layout(push_constant) uniform P {
    uint rows; uint len; uint chunk; uint chunks; uint op; uint in_off; uint out_off;
} p;

// Op codes; keep in sync with grilly_core::generic::Reduce (Sum, _, Max, Min).
const uint SUM = 0, MAX = 2, MIN = 3;

shared int64_t s[256];

int64_t load(uint k) {
    return int64_t(uint64_t(src[2u * k]) | (uint64_t(src[2u * k + 1u]) << 32));
}

int64_t combine(int64_t a, int64_t b) {
    return p.op == SUM ? a + b : (p.op == MAX ? max(a, b) : min(a, b));
}

void main() {
    uint g = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (g >= p.rows * p.chunks) return;   // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    uint row = g / p.chunks, c = g % p.chunks;
    uint start = c * p.chunk, stop = min(p.len, start + p.chunk);
    uint base = p.in_off + row * p.len;

    int64_t r = p.op == SUM ? 0l : (p.op == MAX ? -9223372036854775807l - 1l : 9223372036854775807l);
    for (uint k = start + tid; k < stop; k += 256u) {
        r = combine(r, load(base + k));
    }
    s[tid] = r;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            s[tid] = combine(s[tid], s[tid + stride]);
        }
        barrier();
    }
    if (tid == 0u) {
        uint k = 2u * (p.out_off + g);
        dst[k] = uint(uint64_t(s[0]) & 0xffffffffUL);
        dst[k + 1u] = uint(uint64_t(s[0]) >> 32);
    }
}
