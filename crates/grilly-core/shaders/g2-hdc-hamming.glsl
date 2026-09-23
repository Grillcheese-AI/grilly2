#version 450
#extension GL_GOOGLE_include_directive : require
// Hamming distance from one packed binary hypervector to every entry of a
// packed codebook: popcount of the XOR, summed over the words.
//
// A hypervector is `dim` bits held 32 to a word, so `words = dim / 32` —
// 32x smaller than one float per dimension, which is the whole reason the
// packed algebra exists. One invocation per codebook entry; each walks that
// entry's words once, which is the same memory the entry occupies.
//
// The result is a bit count, an integer, so it is exact. Similarity is
// `1 - distance / dim` and is computed from it on the way out.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer Q { uint q[]; };
layout(std430, binding = 1) readonly buffer C { uint cb[]; };
layout(std430, binding = 2) writeonly buffer O { int dist[]; };
layout(push_constant) uniform P { uint n; uint words; uint q_off; uint c_off; } p;

void main() {
    uint e = flat_index();
    if (e >= p.n) return;
    uint base = p.c_off + e * p.words;
    int d = 0;
    for (uint w = 0u; w < p.words; ++w) d += int(bitCount(q[p.q_off + w] ^ cb[base + w]));
    dist[e] = d;
}
