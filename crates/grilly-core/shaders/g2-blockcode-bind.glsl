#version 450
#extension GL_GOOGLE_include_directive : require
// Sparse block-code VSA binding: per-block circular convolution, and its
// inverse, correlation (IBM's NVSA block codes; Hersche et al. 2023).
//
// A vector is k blocks of l, `(..., k, l)`, each block a one-hot position
// (a discrete code) or a distribution over positions (a continuous one).
//
//   bind:   out[i] = sum_j a[j] * b[(i - j) mod l]
//   unbind: out[i] = sum_j a[(i + j) mod l] * b[j]          (a = composite)
//
// Done directly rather than through an FFT pair, which buys two things.
// For one-hot inputs exactly one term of the sum is non-zero and equals
// 1.0, so the result is an EXACT one-hot — bit-for-bit what v1's shader
// produced by finding the hot positions and adding them, and without its
// restriction, since the same code handles a continuous block. It is also
// more accurate than the FFT for a continuous one: no round trip through
// the frequency domain.
//
// One invocation per output element, l multiply-adds each. A stride of 0
// broadcasts that operand across the batch.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer O { float o[]; };
layout(push_constant) uniform P {
    uint n; uint k; uint l; uint unbind;
    uint a_stride; uint b_stride; uint a_off; uint b_off;
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint pos = i % p.l;
    uint rest = i / p.l;
    uint blk = rest % p.k;
    uint vec = rest / p.k;
    uint abase = p.a_off + vec * p.a_stride + blk * p.l;
    uint bbase = p.b_off + vec * p.b_stride + blk * p.l;

    float s = 0.0;
    if (p.unbind != 0u) {
        for (uint j = 0u; j < p.l; ++j) s += a[abase + (pos + j) % p.l] * b[bbase + j];
    } else {
        for (uint j = 0u; j < p.l; ++j) s += a[abase + j] * b[bbase + (pos + p.l - j) % p.l];
    }
    o[i] = s;
}
