#version 450
#extension GL_GOOGLE_include_directive : require
// Sort, last step: the first `take` sorted positions of each row, as int64.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer X { uint idx[]; };
layout(std430, binding = 1) buffer O { uint dst[]; };
layout(push_constant) uniform P { uint rows; uint take; uint padded; } p;

void main() {
    uint i = flat_index();
    if (i >= p.rows * p.take) return;
    uint r = i / p.take, c = i % p.take;
    dst[2u * i] = idx[r * p.padded + c];
    dst[2u * i + 1u] = 0u;
}
