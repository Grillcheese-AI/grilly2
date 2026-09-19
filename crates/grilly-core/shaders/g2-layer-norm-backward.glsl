#version 450
#extension GL_GOOGLE_include_directive : require
// layer_norm backward over contiguous rows of n, one workgroup per row,
// from the forward's (mean, rstd):
//   xhat = (x - mean) * rstd,  gw = g * w
//   dx   = rstd * (gw - mean(gw) - xhat * mean(gw * xhat))
// and, when the weight needs a gradient, g * xhat per element (its column
// sums over rows are dw; db is the column sums of g).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer G { float g[]; };
layout(std430, binding = 1) buffer X { float x[]; };
layout(std430, binding = 2) buffer W { float w[]; };
layout(std430, binding = 3) buffer S { float stats[]; };
layout(std430, binding = 4) buffer DX { float dx[]; };
layout(std430, binding = 5) buffer GX { float gxhat[]; };
layout(push_constant) uniform P {
    uint rows; uint n; uint has_w; uint want_gxhat; uint g_off; uint x_off; uint w_off;
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
    float total = s_hi[0] + s_lo[0];
    barrier();
    return total;
}

void main() {
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= p.rows) return;
    uint tid = gl_LocalInvocationID.x;
    uint gb = p.g_off + row * p.n, xb = p.x_off + row * p.n;
    float mean = stats[2u * row], rstd = stats[2u * row + 1u];
    float a_hi = 0.0, a_lo = 0.0, c_hi = 0.0, c_lo = 0.0;
    for (uint k = tid; k < p.n; k += 256u) {
        float gw = g[gb + k] * (p.has_w != 0u ? w[p.w_off + k] : 1.0);
        float xhat = (x[xb + k] - mean) * rstd;
        kahan_add(a_hi, a_lo, gw);
        kahan_add(c_hi, c_lo, gw * xhat);
    }
    float mean_gw = row_total(a_hi, a_lo) / float(p.n);
    float mean_gwx = row_total(c_hi, c_lo) / float(p.n);
    for (uint k = tid; k < p.n; k += 256u) {
        float gk = g[gb + k];
        float gw = gk * (p.has_w != 0u ? w[p.w_off + k] : 1.0);
        float xhat = (x[xb + k] - mean) * rstd;
        dx[row * p.n + k] = rstd * (gw - mean_gw - xhat * mean_gwx);
        if (p.want_gxhat != 0u) gxhat[row * p.n + k] = gk * xhat;
    }
}
