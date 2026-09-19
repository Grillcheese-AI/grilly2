#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// Gather along `dim` with an int64 index, for any dtype (`w` words per
// element, copied bit-exactly). For every output coordinate c:
//   out[c] = src[c with c[dim] := index[pos]]
// where pos is c itself (mode 0, torch.gather: index has the output's shape)
// or c[dim] (mode 1, torch.index_select: index is 1-D). Negative indices
// count from the end (callers validate bounds first).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer S { uint src[]; };
layout(std430, binding = 1) buffer I { uint idx[]; };
layout(std430, binding = 2) buffer O { uint dst[]; };
layout(push_constant) uniform P {
    uint n; uint ndim; uint dim; uint w; uint mode; uint src_off; uint idx_off; uint out_off; uint dim_size;
    uint shape[MAX_DIMS]; uint src_strides[MAX_DIMS];
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint rem = i, at = p.src_off, cdim = 0u;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        uint c = rem % p.shape[d];
        rem /= p.shape[d];
        if (uint(d) == p.dim) cdim = c; else at += c * p.src_strides[d];
    }
    uint pos = p.idx_off + (p.mode == 0u ? i : cdim);
    int64_t v = int64_t(uint64_t(idx[2u * pos]) | (uint64_t(idx[2u * pos + 1u]) << 32));
    if (v < 0l) v += int64_t(p.dim_size);
    at += uint(v) * p.src_strides[p.dim];
    for (uint j = 0u; j < p.w; ++j) {
        dst[(p.out_off + i) * p.w + j] = src[at * p.w + j];
    }
}
