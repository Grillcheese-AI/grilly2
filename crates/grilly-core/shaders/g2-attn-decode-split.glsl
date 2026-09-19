#version 450
#extension GL_GOOGLE_include_directive : require
// Decode attention over a slice of the KV cache ("flash decoding", pass 1
// of 2). g2-attn-decode gives one workgroup the whole cache, which leaves
// most of the GPU idle when batch * heads is small: 14 workgroups for a
// 0.5B model at batch 1. This splits the positions instead, so a workgroup
// takes (row, split) and the card gets `splits` times as much to do.
//
// Each workgroup returns its slice's partial softmax, unnormalized:
//   m = max_j s_j, l = sum_j exp(s_j - m), o[d] = sum_j exp(s_j - m) v_j[d]
// for j in [split * chunk, min(len, (split + 1) * chunk)). g2-attn-decode-merge
// combines them. Within a slice the arithmetic is g2-attn-decode's, in the
// same order; across slices the merge rescales by exp(m_s - max_s m_s),
// which is the usual online-softmax identity.
//
// The score scratch is the same (rows, max_len) buffer: a workgroup writes
// only its own slice of its own row.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer Q { float q[]; };
layout(std430, binding = 1) readonly buffer K { float kc[]; };
layout(std430, binding = 2) readonly buffer V { float vc[]; };
layout(std430, binding = 3) coherent buffer S { float sc[]; };    // (rows, max_len) scratch
layout(std430, binding = 4) writeonly buffer M { float ml[]; };   // (rows, splits, 2): m, l
layout(std430, binding = 5) writeonly buffer O { float po[]; };   // (rows, splits, dim)
layout(push_constant) uniform P {
    uint heads; uint kv_heads; uint max_len; uint len;
    uint dim; float scale; uint q_off; uint splits;
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
    uint row = gl_WorkGroupID.x, split = gl_WorkGroupID.y;
    uint tid = gl_LocalInvocationID.x;
    uint b = row / p.heads, h = row % p.heads;
    uint g = h / (p.heads / p.kv_heads);
    uint kbase = (b * p.kv_heads + g) * p.max_len * p.dim;
    uint sbase = row * p.max_len;
    uint obase = (row * p.splits + split) * p.dim;

    uint chunk = (p.len + p.splits - 1u) / p.splits;
    uint start = split * chunk;
    uint end = min(p.len, start + chunk);
    if (start >= end) {  // uniform per workgroup: an empty slice contributes nothing
        if (tid == 0u) {
            ml[(row * p.splits + split) * 2u] = uintBitsToFloat(0xff800000u);
            ml[(row * p.splits + split) * 2u + 1u] = 0.0;
        }
        for (uint d = tid; d < p.dim; d += 256u) po[obase + d] = 0.0;
        return;
    }

    if (tid < p.dim) qs[tid] = q[p.q_off + row * p.dim + tid];
    barrier();

    // 1. scores
    float m = uintBitsToFloat(0xff800000u);
    for (uint j = start + tid; j < end; j += 256u) {
        uint kr = kbase + j * p.dim;
        float s = 0.0;
        for (uint d = 0u; d < p.dim; ++d) s = fma(qs[d], kc[kr + d], s);
        s *= p.scale;
        sc[sbase + j] = s;
        m = max(m, s);
    }
    m = max_all(m);

    // 2. exp and this slice's normalizer
    float z = 0.0, zc = 0.0;
    for (uint j = start + tid; j < end; j += 256u) {
        float e = exp_acc(sc[sbase + j] - m);
        sc[sbase + j] = e;
        kahan_add(z, zc, e);
    }
    z = sum_all(z + zc);
    memoryBarrierBuffer();
    barrier();
    if (tid == 0u) {
        ml[(row * p.splits + split) * 2u] = m;
        ml[(row * p.splits + split) * 2u + 1u] = z;
    }

    // 3. the weighted sum of values, left unnormalized for the merge
    uint groups = max(1u, 256u / p.dim);
    uint lane = tid % p.dim, grp = tid / p.dim;
    float acc = 0.0, ac = 0.0;
    if (grp < groups) {
        for (uint j = start + grp; j < end; j += groups) kahan_add(acc, ac, sc[sbase + j] * vc[kbase + j * p.dim + lane]);
    }
    part[tid] = acc + ac;
    barrier();
    if (tid < p.dim) {
        float t = 0.0, tc = 0.0;
        for (uint k = 0u; k < groups; ++k) kahan_add(t, tc, part[k * p.dim + tid]);
        po[obase + tid] = t + tc;
    }
}
