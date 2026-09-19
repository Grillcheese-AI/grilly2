#version 450
#extension GL_GOOGLE_include_directive : require
// dequantize(q) = q * scale + min for an int4 group-wise weight, where
// min = zero - 8 * scale (g2-quantize-q4). One invocation per weight value:
// anything but a linear sees an Int4Tensor this way.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer Q { uint q[]; };
layout(std430, binding = 1) readonly buffer S { float scale[]; };
layout(std430, binding = 2) readonly buffer Z { float zero[]; };
layout(std430, binding = 3) writeonly buffer W { float w[]; };
layout(push_constant) uniform P { uint rows; uint k; uint group; uint words; uint groups; } p;

void main() {
    uint id = flat_index();
    if (id >= p.rows * p.k) return;
    uint row = id / p.k, i = id % p.k;
    uint g = i / p.group;
    float s = scale[row * p.groups + g];
    float level = float((q[row * p.words + (i >> 3)] >> ((i & 7u) * 4u)) & 0xFu);
    w[id] = fma(level, s, zero[row * p.groups + g] - 8.0 * s);
}
