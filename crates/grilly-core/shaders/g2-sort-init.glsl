#version 450
#extension GL_GOOGLE_include_directive : require
// Sort, step 1: copy each row of a contiguous (rows, n) float32 or int64
// tensor into a (rows, padded) key buffer, padded to a power of two, and
// number the positions. Positions >= n are padding, and g2-sort-step orders
// them last. Keys keep their dtype's words (1 for float32, 2 for int64).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer S { uint src[]; };
layout(std430, binding = 1) buffer K { uint keys[]; };
layout(std430, binding = 2) buffer X { uint idx[]; };
layout(push_constant) uniform P { uint rows; uint n; uint padded; uint in_off; uint words; } p;

void main() {
    uint i = flat_index();
    if (i >= p.rows * p.padded) return;
    uint r = i / p.padded, c = i % p.padded;
    for (uint w = 0u; w < p.words; ++w) {
        keys[p.words * i + w] = c < p.n ? src[p.words * (p.in_off + r * p.n + c) + w] : 0u;
    }
    idx[i] = c;
}
