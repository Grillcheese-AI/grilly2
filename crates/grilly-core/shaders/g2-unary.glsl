#version 450
#extension GL_GOOGLE_include_directive : require
// y[i] = f(x[i]; a, b). x and y may be the same buffer (in-place ops).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer X { float x[]; };
layout(std430, binding = 1) buffer Y { float y[]; };
layout(push_constant) uniform P { uint n; uint op; float a; float b; uint x_off; uint y_off; } p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    float v = x[p.x_off + i];
    float r = g2_unary(p.op, v, p.a, p.b);
    y[p.y_off + i] = r;
}
