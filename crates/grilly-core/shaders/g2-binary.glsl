#version 450
#extension GL_GOOGLE_include_directive : require
// out[i] = op(a[...], b[...]) with broadcasting: each input is read through
// its own strides over the output's shape (stride 0 on broadcast dims).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 1) buffer B { float b[]; };
layout(std430, binding = 2) buffer O { float o[]; };
layout(push_constant) uniform P {
    uint n; uint op; uint ndim; uint a_off; uint b_off; uint o_off;
    uint shape[MAX_DIMS]; uint sa[MAX_DIMS]; uint sb[MAX_DIMS];
} p;


void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint rem = i, ia = p.a_off, ib = p.b_off;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        uint c = rem % p.shape[d];
        rem /= p.shape[d];
        ia += c * p.sa[d];
        ib += c * p.sb[d];
    }
    float r = g2_binary(p.op, a[ia], b[ib]);
    o[p.o_off + i] = r;
}
