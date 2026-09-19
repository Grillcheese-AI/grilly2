#version 450
#extension GL_GOOGLE_include_directive : require
// w[row][i] = float(q[row][i]) * scale[row]: an int8 weight-only weight
// (g2-quantize-q8) back to float32, as torchao's dequantize (zero point 0).
// One invocation per element.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer Q { uint q[]; };
layout(std430, binding = 1) readonly buffer S { float scale[]; };
layout(std430, binding = 2) writeonly buffer W { float w[]; };
layout(push_constant) uniform P { uint rows; uint k; uint words; } p;

void main() {
    uint i = flat_index();
    if (i >= p.rows * p.k) return;
    uint row = i / p.k, col = i % p.k;
    int word = int(q[row * p.words + col / 4u]);
    w[i] = float(bitfieldExtract(word, int(8u * (col % 4u)), 8)) * scale[row];
}
