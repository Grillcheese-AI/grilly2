#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// out[i] = op(a[...], b[...]) on int64 operands with broadcasting, exactly
// (no float round trip). The output is int64, or bool (one u32 0/1 per
// element) for comparisons and logical results.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer A { uint a[]; };
layout(std430, binding = 1) buffer B { uint b[]; };
layout(std430, binding = 2) buffer O { uint o[]; };
layout(push_constant) uniform P {
    uint n; uint op; uint ndim; uint a_off; uint b_off; uint o_off; uint out_bool;
    uint shape[MAX_DIMS]; uint sa[MAX_DIMS]; uint sb[MAX_DIMS];
} p;

// Op codes; keep in sync with grilly_core::generic::IntBinary.
const uint ADD = 0, SUB = 1, MUL = 2, MAXIMUM = 3, MINIMUM = 4, EQ = 5, NE = 6, GT = 7,
           GE = 8, LT = 9, LE = 10, AND = 11, OR = 12, XOR = 13;

int64_t load(uint k, bool from_a) {
    uint lo = from_a ? a[2u * k] : b[2u * k];
    uint hi = from_a ? a[2u * k + 1u] : b[2u * k + 1u];
    return int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
}

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
    int64_t x = load(ia, true), y = load(ib, false), r;
    switch (p.op) {
        case ADD: r = x + y; break;
        case SUB: r = x - y; break;
        case MUL: r = x * y; break;
        case MAXIMUM: r = max(x, y); break;
        case MINIMUM: r = min(x, y); break;
        case EQ: r = x == y ? 1l : 0l; break;
        case NE: r = x != y ? 1l : 0l; break;
        case GT: r = x > y ? 1l : 0l; break;
        case GE: r = x >= y ? 1l : 0l; break;
        case LT: r = x < y ? 1l : 0l; break;
        case LE: r = x <= y ? 1l : 0l; break;
        case AND: r = x & y; break;
        case OR: r = x | y; break;
        case XOR: r = x ^ y; break;
        default: r = 0l;
    }
    if (p.out_bool != 0u) {
        o[p.o_off + i] = r != 0l ? 1u : 0u;
    } else {
        uint k = 2u * (p.o_off + i);
        o[k] = uint(uint64_t(r) & 0xffffffffUL);
        o[k + 1u] = uint(uint64_t(r) >> 32);
    }
}
