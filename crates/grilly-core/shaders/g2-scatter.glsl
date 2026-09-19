#version 450
#extension GL_GOOGLE_include_directive : require
// torch's scatter: dst[c with c[dim] := index[c]] = src[c] for every
// position c of the index, any dtype (`words` 32-bit words per element).
// One invocation per index element. Indices are bounds-checked and
// non-negative (torch's scatter takes no negative indices); duplicates
// write in hardware order, which torch also leaves unspecified.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer S { uint src[]; };
layout(std430, binding = 1) buffer I { uint idx[]; };
layout(std430, binding = 2) buffer O { uint dst[]; };
layout(push_constant) uniform P {
    uint n; uint ndim; uint dim; uint words; uint src_off; uint idx_off; uint dst_off; uint unused;
    uint idx_shape[MAX_DIMS]; uint src_strides[MAX_DIMS]; uint dst_strides[MAX_DIMS];
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint rem = i, s = 0u, d = 0u;
    for (int k = int(p.ndim) - 1; k >= 0; --k) {
        uint c = rem % p.idx_shape[k];
        rem /= p.idx_shape[k];
        s += c * p.src_strides[k];
        if (uint(k) != p.dim) d += c * p.dst_strides[k];
    }
    d += idx[2u * (p.idx_off + i)] * p.dst_strides[p.dim];
    for (uint w = 0u; w < p.words; ++w) {
        dst[p.words * (p.dst_off + d) + w] = src[p.words * (p.src_off + s) + w];
    }
}
