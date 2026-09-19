#version 450
#extension GL_GOOGLE_include_directive : require
// Pass 2 of decode attention over slices (see g2-attn-decode-split):
// combine each row's per-slice (m, l, o) into the row's output.
//   M = max_s m_s,  w_s = exp(m_s - M),  Z = sum_s l_s w_s,
//   out[d] = (sum_s o_s[d] w_s) / Z
// One workgroup per row, `splits` at most one per thread. Empty slices
// carry m = -inf, so w_s = 0 and they contribute nothing.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer M { float ml[]; };   // (rows, splits, 2)
layout(std430, binding = 1) readonly buffer P { float po[]; };   // (rows, splits, dim)
layout(std430, binding = 2) writeonly buffer O { float o[]; };   // (rows, dim)
layout(push_constant) uniform C { uint rows; uint splits; uint dim; } p;

shared float red[256];
shared float ws[256];

float sum_all(float v) {
    uint tid = gl_LocalInvocationID.x;
    red[tid] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}

float max_all(float v) {
    uint tid = gl_LocalInvocationID.x;
    red[tid] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        barrier();
    }
    float r = red[0];
    barrier();
    return r;
}

void main() {
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= p.rows) return;  // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;

    float m = tid < p.splits ? ml[(row * p.splits + tid) * 2u] : uintBitsToFloat(0xff800000u);
    float top = max_all(m);
    float w = tid < p.splits ? exp_acc(m - top) : 0.0;
    ws[tid] = w;
    float z = sum_all(tid < p.splits ? ml[(row * p.splits + tid) * 2u + 1u] * w : 0.0);
    barrier();

    for (uint d = tid; d < p.dim; d += 256u) {
        float t = 0.0, tc = 0.0;
        for (uint s = 0u; s < p.splits; ++s) kahan_add(t, tc, po[(row * p.splits + s) * p.dim + d] * ws[s]);
        o[row * p.dim + d] = div_rn(t + tc, z);
    }
}
