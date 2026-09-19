#version 450
#extension GL_GOOGLE_include_directive : require
// rms_norm over contiguous rows of n, torch's formula:
//   y = h * (1 / sqrt(mean(h^2) + eps)) * w
// with h = x, or h = x + r when a residual is given (a decoder layer's
// "add, then normalize"; h is then written out too). One workgroup per
// row. The sum of squares is compensated, and the mean and the reciprocal
// are correctly rounded.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

const uint HAS_WEIGHT = 1u, HAS_RESIDUAL = 2u;

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer R { float res[]; };
layout(std430, binding = 2) readonly buffer W { float w[]; };
layout(std430, binding = 3) writeonly buffer Y { float y[]; };
layout(std430, binding = 4) writeonly buffer H { float h[]; };
layout(push_constant) uniform P {
    uint rows; uint n; float eps; uint flags;
    uint x_off; uint r_off; uint w_off;
} p;

shared float s_hi[256];
shared float s_lo[256];

float row_total(float hi, float lo) {
    uint tid = gl_LocalInvocationID.x;
    s_hi[tid] = hi;
    s_lo[tid] = lo;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float a = s_hi[tid], b = s_lo[tid];
            pair_add(a, b, s_hi[tid + stride], s_lo[tid + stride]);
            s_hi[tid] = a;
            s_lo[tid] = b;
        }
        barrier();
    }
    return s_hi[0] + s_lo[0];
}

float value(uint row, uint k) {
    float v = x[p.x_off + row * p.n + k];
    if ((p.flags & HAS_RESIDUAL) != 0u) {
        precise float s = v + res[p.r_off + row * p.n + k];
        v = s;
    }
    return v;
}

void main() {
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= p.rows) return;  // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    float hi = 0.0, lo = 0.0;
    for (uint k = tid; k < p.n; k += 256u) {
        float v = value(row, k);
        precise float sq = v * v;
        kahan_add(hi, lo, sq);
    }
    float mean = div_rn(row_total(hi, lo), float(p.n));
    precise float shifted = mean + p.eps;
    float r = div_rn(1.0, sqrt(shifted));
    for (uint k = tid; k < p.n; k += 256u) {
        float v = value(row, k);
        if ((p.flags & HAS_RESIDUAL) != 0u) h[row * p.n + k] = v;
        precise float yv = v * r;
        if ((p.flags & HAS_WEIGHT) != 0u) yv = yv * w[p.w_off + k];
        y[row * p.n + k] = yv;
    }
}
