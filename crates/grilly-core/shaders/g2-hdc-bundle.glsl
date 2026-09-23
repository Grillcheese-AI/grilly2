#version 450
#extension GL_GOOGLE_include_directive : require
// Bundling of packed binary hypervectors: a bit-wise majority vote.
//
// Superposition in the packed algebra. Output bit is 1 where more than half
// the inputs have it set; a tie, which only an even count can produce, goes
// to 0. That is v1's rule (`counts > n / 2.0` in float), written here as
// `2 * count > n` so it is integer arithmetic and cannot land on the wrong
// side of a rounding.
//
// One invocation per output word, holding that word's 32 counters in
// registers and reading each input word once — rather than one invocation
// per bit, which would read the same word 32 times.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer V { uint vecs[]; };
layout(std430, binding = 1) writeonly buffer O { uint o[]; };
layout(push_constant) uniform P { uint n; uint words; uint v_off; } p;

void main() {
    uint w = flat_index();
    if (w >= p.words) return;

    uint count[32];
    for (uint b = 0u; b < 32u; ++b) count[b] = 0u;
    for (uint v = 0u; v < p.n; ++v) {
        uint word = vecs[p.v_off + v * p.words + w];
        for (uint b = 0u; b < 32u; ++b) count[b] += (word >> b) & 1u;
    }

    uint out_word = 0u;
    for (uint b = 0u; b < 32u; ++b) {
        if (2u * count[b] > p.n) out_word |= 1u << b;
    }
    o[w] = out_word;
}
