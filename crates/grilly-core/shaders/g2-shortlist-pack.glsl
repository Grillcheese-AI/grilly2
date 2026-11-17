#version 450
#extension GL_GOOGLE_include_directive : require
// A vocabulary shortlist head's codes (grilly_infer::shortlist): the sign
// bits of every row of the head's weight, 32 to a word. Bit b of word w of a
// row is `w[row, 32 w + b] > 0`, so 0.0, -0.0 and NaN give 0 — numpy's
// `packbits(x > 0, bitorder="little")` viewed as little-endian words.
//
// One invocation per (row, word). Built once, when the head is made.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer Wt { float w[]; };
layout(std430, binding = 1) writeonly buffer Cd { uint codes[]; };
layout(push_constant) uniform P { uint n; uint words; uint dim; uint w_off; } p;

void main() {
    uint e = flat_index();
    if (e >= p.n) return;
    uint row = e / p.words, word = e % p.words;
    uint base = p.w_off + row * p.dim + word * 32u;
    uint bits = 0u;
    for (uint b = 0u; b < 32u; ++b) {
        if (w[base + b] > 0.0) bits |= 1u << b;
    }
    codes[e] = bits;
}
