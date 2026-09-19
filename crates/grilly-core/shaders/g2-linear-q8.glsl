#version 450
#extension GL_GOOGLE_include_directive : require
// y = (x @ q^T) * scale + bias: linear with an int8 weight-only weight
// (g2-quantize-q8), dequantized in the shader. As in torchao's weight-only
// path, the row's scale multiplies the finished dot product.
//
// Shaped for decode, where x has few rows and the weight read is the whole
// cost. 256 threads are 4 groups of 64; group g owns output feature
// 4 * (workgroup) + g, and its lanes read that weight row as consecutive
// 16-byte vectors (16 int8 values each). A workgroup takes up to ROWS rows of
// x, so the weight is read once per ROWS rows. When x's rows start on 16-byte
// boundaries (x_vec4), x is read four floats per load. Each lane's sum runs
// in a fixed order, then a fixed tree over the 64 lanes: deterministic.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

#define ROWS 8u

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 0) readonly buffer X4 { vec4 x4[]; };  // the same buffer
layout(std430, binding = 1) readonly buffer Q { uvec4 q[]; };
layout(std430, binding = 2) readonly buffer S { float scale[]; };
layout(std430, binding = 3) readonly buffer B { float bias[]; };
layout(std430, binding = 4) writeonly buffer Y { float y[]; };
layout(push_constant) uniform P {
    uint m; uint k; uint n; uint vecs;  // vecs: 16-byte vectors per weight row
    uint x_off; uint b_off; uint has_bias; uint x_vec4;  // x_vec4: x_off and k are multiples of 4
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
            uvec4 packed = q[col * p.vecs + v];
            float wf[16];
            for (uint j = 0u; j < 4u; ++j) {
                int word = int(packed[j]);
                wf[4u * j] = float(bitfieldExtract(word, 0, 8));
                wf[4u * j + 1u] = float(bitfieldExtract(word, 8, 8));
                wf[4u * j + 2u] = float(bitfieldExtract(word, 16, 8));
                wf[4u * j + 3u] = float(bitfieldExtract(word, 24, 8));
            }
            uint k0 = v * 16u;
            uint count = min(16u, p.k - k0);  // the row's padding is never multiplied
            for (uint r = 0u; r < ROWS; ++r) {
                if (r < rows) {
                    uint xb = p.x_off + (row0 + r) * p.k + k0;
                    if (p.x_vec4 != 0u) {
                        for (uint j = 0u; j < 4u; ++j) {
                            if (4u * j < count) {  // count is a multiple of 4 here
                                vec4 xv = x4[xb / 4u + j];
                                acc[r] = fma(xv.x, wf[4u * j], acc[r]);
                                acc[r] = fma(xv.y, wf[4u * j + 1u], acc[r]);
                                acc[r] = fma(xv.z, wf[4u * j + 2u], acc[r]);
                                acc[r] = fma(xv.w, wf[4u * j + 3u], acc[r]);
                            }
                        }
                    } else {
                        for (uint j = 0u; j < 16u; ++j) {
                            if (j < count) acc[r] = fma(x[xb + j], wf[j], acc[r]);
                        }
                    }
                }
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
                float v = partial[r][tid] * scale[col];
                if (p.has_bias != 0u) v += bias[p.b_off + col];
                y[(row0 + r) * p.n + col] = v;
            }
        }
    }
}
