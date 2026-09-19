#version 450
#extension GL_GOOGLE_include_directive : require
// y = x @ dequantize(q)^T + bias for an int4 group-wise weight
// (g2-quantize-q4), dequantized in the shader:
//
//   w[c][i] = q[c][i] * scale[c][g] + min[c][g],  min = zero - 8 * scale
//
// with g the group i belongs to. The scale cannot be pulled out of the
// whole dot product — it changes every `group` values — so each 16-byte
// vector's partial sum is scaled as it finishes, using
//   sum_i (q_i * s + m) * x_i = s * sum_i q_i x_i + m * sum_i x_i.
// A group is a multiple of 32 values and a vector is 32, so a vector lies
// in exactly one group and its scale is read once.
//
// Shaped like g2-linear-q8, whose structure this follows deliberately:
// 256 threads are 4 groups of 64 lanes, group t owns output feature
// 4 * workgroup + t, its lanes read that weight row as consecutive 16-byte
// vectors, and a workgroup takes up to ROWS rows of x. Only one
// accumulator per row stays live; holding a second pair of per-row sums
// spilled them to scratch and cost 40% (measured). Each lane sums in a
// fixed order, then a fixed tree over the 64 lanes: deterministic.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

#define ROWS 8u

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 0) readonly buffer X4 { vec4 x4[]; };  // the same buffer
layout(std430, binding = 1) readonly buffer Q { uvec4 q[]; };
layout(std430, binding = 2) readonly buffer S { float scale[]; };
layout(std430, binding = 3) readonly buffer Z { float zero[]; };
layout(std430, binding = 4) readonly buffer B { float bias[]; };
layout(std430, binding = 5) writeonly buffer Y { float y[]; };
layout(push_constant) uniform P {
    uint m; uint k; uint n; uint vecs;     // vecs: 16-byte vectors per weight row
    uint group; uint groups;               // values per group, groups per row
    uint x_off; uint b_off; uint has_bias; uint x_vec4;
} p;

shared float partial[ROWS][256];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint lane = tid & 63u;
    uint col = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 4u + (tid >> 6);
    uint row0 = gl_WorkGroupID.z * ROWS;
    uint rows = min(ROWS, p.m - row0);

    float acc[ROWS];
    for (uint r = 0u; r < ROWS; ++r) acc[r] = 0.0;

    if (col < p.n) {
        for (uint v = lane; v < p.vecs; v += 64u) {
            uint k0 = v * 32u;
            uint g = k0 / p.group;
            float s = scale[col * p.groups + g];
            float lo = zero[col * p.groups + g] - 8.0 * s;
            uvec4 packed = q[col * p.vecs + v];
            float lv[32];
            for (uint w = 0u; w < 4u; ++w) {
                uint word = packed[w];
                for (uint j = 0u; j < 8u; ++j) lv[8u * w + j] = float((word >> (j * 4u)) & 0xFu);
            }
            uint count = min(32u, p.k - k0);  // the row's padding is never multiplied
            for (uint r = 0u; r < ROWS; ++r) {
                if (r >= rows) continue;
                uint xb = p.x_off + (row0 + r) * p.k + k0;
                float dot = 0.0, sum = 0.0;
                if (p.x_vec4 != 0u) {
                    for (uint j = 0u; j < 8u; ++j) {
                        if (4u * j >= count) break;  // count is a multiple of 4 here
                        vec4 xv = x4[xb / 4u + j];
                        dot = fma(lv[4u * j], xv.x, dot);
                        dot = fma(lv[4u * j + 1u], xv.y, dot);
                        dot = fma(lv[4u * j + 2u], xv.z, dot);
                        dot = fma(lv[4u * j + 3u], xv.w, dot);
                        sum += (xv.x + xv.y) + (xv.z + xv.w);
                    }
                } else {
                    for (uint j = 0u; j < 32u; ++j) {
                        if (j >= count) break;
                        float xv = x[xb + j];
                        dot = fma(lv[j], xv, dot);
                        sum += xv;
                    }
                }
                acc[r] = fma(s, dot, acc[r]);
                acc[r] = fma(lo, sum, acc[r]);
            }
        }
    }

    for (uint r = 0u; r < ROWS; ++r) partial[r][tid] = acc[r];
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1) {
        if (lane < stride) {
            for (uint r = 0u; r < ROWS; ++r) partial[r][tid] += partial[r][tid + stride];
        }
        barrier();
    }
    if (lane == 0u && col < p.n) {
        for (uint r = 0u; r < ROWS; ++r) {
            if (r < rows) {
                float v = partial[r][tid];
                if (p.has_bias != 0u) v += bias[p.b_off + col];
                y[(row0 + r) * p.n + col] = v;
            }
        }
    }
}
