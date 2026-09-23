#version 450
#extension GL_GOOGLE_include_directive : require
// Attention of one new query token against a KV cache (the decode step):
//   out[b, h] = softmax(q[b, h] . K[b, g]^T * scale) @ V[b, g]
// over the cache's first `len` positions, g = h / (heads / kv_heads): grouped
// query attention by index, with no copy of the cache. One workgroup per
// (batch, head). The cache is (batch, kv_heads, max_len, dim), contiguous.
//
// 1. scores s_j = scale * q . k_j, one key per thread at a time, into a
//    scratch row (and the row maximum);
// 2. e_j = exp(s_j - max) in place, and Z = sum e_j (compensated);
// 3. out[d] = sum_j e_j v_j[d] / Z: G groups of `dim` lanes, lane d reading
//    v_j[d] (consecutive lanes, consecutive addresses), groups over j; the
//    groups' partial sums are then added in a fixed order.
// Every loop is bounded by len, dim or the workgroup; every sum runs in a
// fixed order.
//
// `len` is read from the cache's cursor buffer, not pushed: it changes every
// decode step, and a pushed value would be baked into a recorded command and
// replayed stale (grilly_core::graph).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer Q { float q[]; };
layout(std430, binding = 1) readonly buffer K { float kc[]; };
layout(std430, binding = 2) readonly buffer V { float vc[]; };
layout(std430, binding = 3) coherent buffer S { float sc[]; };   // (batch * heads, max_len) scratch
layout(std430, binding = 4) writeonly buffer O { float o[]; };   // (batch, heads, dim)
layout(std430, binding = 5) readonly buffer L { uint lens[]; };  // filled positions, per layer
layout(push_constant) uniform P {
    uint heads; uint kv_heads; uint max_len; uint layer;
    uint dim; float scale; uint q_off; uint rows;  // rows = batch * heads
} p;

shared float qs[256];
shared float red[256];
shared float part[256];

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
    uint b = row / p.heads, h = row % p.heads;
    uint g = h / (p.heads / p.kv_heads);
    uint kbase = (b * p.kv_heads + g) * p.max_len * p.dim;
    uint sbase = row * p.max_len;
    uint len = min(lens[p.layer], p.max_len);
    if (len == 0u) return;  // uniform per workgroup

    if (tid < p.dim) qs[tid] = q[p.q_off + row * p.dim + tid];
    barrier();

    // 1. scores
    float m = uintBitsToFloat(0xff800000u);
    for (uint j = tid; j < len; j += 256u) {
        uint kr = kbase + j * p.dim;
        float s = 0.0;
        for (uint d = 0u; d < p.dim; ++d) s = fma(qs[d], kc[kr + d], s);
        s *= p.scale;
        sc[sbase + j] = s;
        m = max(m, s);
    }
    m = max_all(m);

    // 2. exp and the normalizer
    float z = 0.0, zc = 0.0;
    for (uint j = tid; j < len; j += 256u) {
        float e = exp_acc(sc[sbase + j] - m);
        sc[sbase + j] = e;
        kahan_add(z, zc, e);
    }
    z = sum_all(z + zc);
    memoryBarrierBuffer();
    barrier();

    // 3. the weighted sum of values
    uint groups = max(1u, 256u / p.dim);
    uint lane = tid % p.dim, grp = tid / p.dim;
    float acc = 0.0, ac = 0.0;
    if (grp < groups) {
        for (uint j = grp; j < len; j += groups) kahan_add(acc, ac, sc[sbase + j] * vc[kbase + j * p.dim + lane]);
    }
    part[tid] = acc + ac;
    barrier();
    if (tid < p.dim) {
        float t = 0.0, tc = 0.0;
        for (uint k = 0u; k < groups; ++k) kahan_add(t, tc, part[k * p.dim + tid]);
        o[row * p.dim + tid] = div_rn(t + tc, z);
    }
}
