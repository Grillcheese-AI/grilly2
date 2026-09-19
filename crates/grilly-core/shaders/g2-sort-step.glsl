#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// Sort, one compare-exchange stage (block k, distance j) of a bitonic sort
// of every row, float32 or int64 keys. The order is total, so the result is
// deterministic:
//   ascending:  smaller first, NaN last;  descending: larger first, NaN first
//   (torch's NaN placement); equal values by original position; padding last.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer K { uint keys[]; };
layout(std430, binding = 1) buffer X { uint idx[]; };
layout(push_constant) uniform P { uint rows; uint n; uint padded; uint j; uint k; uint descending; uint words; } p;

int64_t key_int(uint a) {
    return int64_t(uint64_t(keys[2u * a]) | (uint64_t(keys[2u * a + 1u]) << 32));
}

// Whether the element at buffer position a belongs before the one at b.
bool before(uint a, uint b) {
    uint ia = idx[a], ib = idx[b];
    if (ia >= p.n) return false;
    if (ib >= p.n) return true;
    bool desc = p.descending == 1u;
    if (p.words == 2u) {
        int64_t va = key_int(a), vb = key_int(b);
        if (va != vb) return desc ? va > vb : va < vb;
        return ia < ib;
    }
    float va = uintBitsToFloat(keys[a]), vb = uintBitsToFloat(keys[b]);
    bool na = isnan(va), nb = isnan(vb);
    if (na != nb) return desc ? na : nb;
    if (!na && va != vb) return desc ? va > vb : va < vb;
    return ia < ib;
}

void main() {
    uint i = flat_index();
    if (i >= p.rows * p.padded) return;
    uint c = i % p.padded, l = c ^ p.j;
    if (l <= c) return;
    uint a = i, b = i - c + l;
    bool forward = (c & p.k) == 0u;
    if (forward ? before(b, a) : before(a, b)) {
        for (uint w = 0u; w < p.words; ++w) {
            uint kv = keys[p.words * a + w];
            keys[p.words * a + w] = keys[p.words * b + w];
            keys[p.words * b + w] = kv;
        }
        uint iv = idx[a]; idx[a] = idx[b]; idx[b] = iv;
    }
}
