#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// The backward of g2-index (float32), deterministic: one invocation per
// element of the input-shaped result, summing, in a fixed order, the
// gradients of every output position whose index points at it.
//   mode 0 (gather): candidates are c with c[dim] := j, for j < len, when
//                    c is inside the index's shape in the other dims;
//   mode 1 (select): candidates are c with c[dim] := j, for j < len.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer G { float g[]; };
layout(std430, binding = 1) buffer I { uint idx[]; };
layout(std430, binding = 2) buffer O { float dst[]; };
layout(push_constant) uniform P {
    uint n; uint ndim; uint dim; uint mode; uint g_off; uint idx_off; uint out_off; uint dim_size; uint len;
    uint in_shape[MAX_DIMS]; uint g_shape[MAX_DIMS];
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    // coordinates of this destination element (input shape)
    uint c[MAX_DIMS];
    uint rem = i;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        c[d] = rem % p.in_shape[d];
        rem /= p.in_shape[d];
    }
    bool inside = true;
    for (uint d = 0u; d < p.ndim; ++d) {
        if (d != p.dim && c[d] >= p.g_shape[d]) inside = false;
    }
    precise float sum = 0.0;
    if (inside) {
        for (uint j = 0u; j < p.len; ++j) {
            // flat position of (c with c[dim] := j) in the gradient's shape
            uint pos = 0u;
            for (uint d = 0u; d < p.ndim; ++d) {
                pos = pos * p.g_shape[d] + (d == p.dim ? j : c[d]);
            }
            uint ipos = p.idx_off + (p.mode == 0u ? pos : j);
            int64_t v = int64_t(uint64_t(idx[2u * ipos]) | (uint64_t(idx[2u * ipos + 1u]) << 32));
            if (v < 0l) v += int64_t(p.dim_size);
            if (uint(v) == c[p.dim]) sum += g[p.g_off + pos];
        }
    }
    dst[p.out_off + i] = sum;
}
