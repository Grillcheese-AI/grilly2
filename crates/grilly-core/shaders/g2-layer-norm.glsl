#version 450
#extension GL_GOOGLE_include_directive : require
// layer_norm forward over contiguous rows of n: one workgroup per row. The
// mean is a compensated sum; the (biased, as torch) variance is a second
// compensated pass over (x - mean)^2; then y = (x - mean) * rstd * w + b.
// (mean, rstd) per row are kept for the backward.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer X { float x[]; };
layout(std430, binding = 1) buffer W { float w[]; };
layout(std430, binding = 2) buffer B { float bias[]; };
layout(std430, binding = 3) buffer Y { float y[]; };
layout(std430, binding = 4) buffer S { float stats[]; };  // (mean, rstd) per row
layout(push_constant) uniform P {
    uint rows; uint n; float eps; uint affine;  // affine: 1 = weight, 2 = bias
    uint x_off; uint w_off; uint b_off;
} p;

shared float s_hi[256];
shared float s_lo[256];

// The workgroup's total of each thread's (hi, lo), in a fixed order.
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
    float total = s_hi[0] + s_lo[0];
    barrier();  // everyone has read it before the arrays are reused
    return total;
}

void main() {
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= p.rows) return;  // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    uint base = p.x_off + row * p.n;
    float hi = 0.0, lo = 0.0;
    for (uint k = tid; k < p.n; k += 256u) kahan_add(hi, lo, x[base + k]);
    float mean = row_total(hi, lo) / float(p.n);
    hi = 0.0;
    lo = 0.0;
    for (uint k = tid; k < p.n; k += 256u) {
        float d = x[base + k] - mean;
        kahan_add(hi, lo, d * d);
    }
    float rstd = 1.0 / sqrt(row_total(hi, lo) / float(p.n) + p.eps);
    for (uint k = tid; k < p.n; k += 256u) {
        float v = (x[base + k] - mean) * rstd;
        if ((p.affine & 1u) != 0u) v *= w[p.w_off + k];
        if ((p.affine & 2u) != 0u) v += bias[p.b_off + k];
        y[row * p.n + k] = v;
    }
    if (tid == 0u) {
        stats[2u * row] = mean;
        stats[2u * row + 1u] = rstd;
    }
}
