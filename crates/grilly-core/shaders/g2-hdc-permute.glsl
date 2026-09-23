#version 450
#extension GL_GOOGLE_include_directive : require
// Cyclic bit rotation of a packed binary hypervector — VSA's protection
// operator, which is how a position in a sequence gets its own role:
// permuting a hypervector produces one unrelated to it, reversibly.
//
//   out bit i = in bit (i - shift) mod dim
//
// the same as numpy's `roll` over the unpacked bits, which is v1's
// reference. Global bit i lives in word i / 32 at position i % 32, which is
// what `unpackbits(..., bitorder="little")` over little-endian words means.
//
// One invocation per output word, gathering its 32 bits. They come from at
// most two source words, but the general gather costs little and keeps the
// index arithmetic in one place.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer S { uint src[]; };
layout(std430, binding = 1) writeonly buffer O { uint o[]; };
layout(push_constant) uniform P { uint n; uint words; uint dim; uint shift; uint s_off; } p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint w = i % p.words;
    uint base = p.s_off + (i / p.words) * p.words;

    uint out_word = 0u;
    for (uint b = 0u; b < 32u; ++b) {
        uint bit = w * 32u + b;
        if (bit >= p.dim) break;                       // a partly used last word
        uint from = (bit + p.dim - p.shift) % p.dim;
        out_word |= ((src[base + from / 32u] >> (from % 32u)) & 1u) << b;
    }
    o[i] = out_word;
}
