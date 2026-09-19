#version 450
#extension GL_GOOGLE_include_directive : require
// out[i] = cond[...] ? a[...] : b[...], broadcasting all three. Elements are
// `w` 32-bit words each (any dtype), copied bit-exactly; cond is bool
// (one u32 per element). Offsets and strides are in elements.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer C { uint c[]; };
layout(std430, binding = 1) buffer A { uint a[]; };
layout(std430, binding = 2) buffer B { uint b[]; };
layout(std430, binding = 3) buffer O { uint o[]; };
layout(push_constant) uniform P {
    uint n; uint ndim; uint w; uint c_off; uint a_off; uint b_off; uint o_off;
    uint shape[MAX_DIMS]; uint sc[MAX_DIMS]; uint sa[MAX_DIMS]; uint sb[MAX_DIMS];
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint rem = i, ic = p.c_off, ia = p.a_off, ib = p.b_off;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        uint k = rem % p.shape[d];
        rem /= p.shape[d];
        ic += k * p.sc[d];
        ia += k * p.sa[d];
        ib += k * p.sb[d];
    }
    bool take_a = c[ic] != 0u;
    for (uint j = 0u; j < p.w; ++j) {
        o[(p.o_off + i) * p.w + j] = take_a ? a[ia * p.w + j] : b[ib * p.w + j];
    }
}
