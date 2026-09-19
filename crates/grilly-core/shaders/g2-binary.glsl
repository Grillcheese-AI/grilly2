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

// Op codes; keep in sync with grilly_core::generic::Binary.
const uint ADD = 0, SUB = 1, MUL = 2, DIV = 3, POW = 4, MAXIMUM = 5, MINIMUM = 6,
           EQ = 7, NE = 8, GT = 9, GE = 10, LT = 11, LE = 12;

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
    float x = a[ia], y = b[ib], r;
    switch (p.op) {
        case ADD: r = x + y; break;
        case SUB: r = x - y; break;
        case MUL: r = x * y; break;
        case DIV: r = x / y; break;
        case POW: r = power(x, y); break;
        case MAXIMUM: r = (isnan(x) || isnan(y)) ? NAN : max(x, y); break;
        case MINIMUM: r = (isnan(x) || isnan(y)) ? NAN : min(x, y); break;
        case EQ: r = x == y ? 1.0 : 0.0; break;
        case NE: r = x != y ? 1.0 : 0.0; break;
        case GT: r = x > y ? 1.0 : 0.0; break;
        case GE: r = x >= y ? 1.0 : 0.0; break;
        case LT: r = x < y ? 1.0 : 0.0; break;
        case LE: r = x <= y ? 1.0 : 0.0; break;
        default: r = NAN;
    }
    o[p.o_off + i] = r;
}
