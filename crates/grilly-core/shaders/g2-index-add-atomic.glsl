#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// The backward of g2-index (float32), fast: one invocation per gradient
// element, atomically added into the (zeroed) input-shaped result. Duplicate
// indices are summed in hardware order, so the last bits can vary run to
// run; torch.use_deterministic_algorithms(True) selects g2-index-add.
// The add is a compare-and-swap loop on core uint atomics, not
// GL_EXT_shader_atomic_float: float atomicAdd lands every add at flat index
// 0 on the RX 6750 XT driver stack (measured in v1, cpp/src/ops/loss.cpp,
// and again here).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer G { float g[]; };
layout(std430, binding = 1) buffer I { uint idx[]; };
layout(std430, binding = 2) buffer O { uint dst[]; };
layout(push_constant) uniform P {
    uint n; uint ndim; uint dim; uint mode; uint g_off; uint idx_off; uint out_off; uint dim_size; uint len;
    uint in_shape[MAX_DIMS]; uint g_shape[MAX_DIMS];
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint rem = i, at = 0u, stride = 1u;
    uint coords[MAX_DIMS];
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        coords[d] = rem % p.g_shape[d];
        rem /= p.g_shape[d];
    }
    uint ipos = p.idx_off + (p.mode == 0u ? i : coords[p.dim]);
    int64_t v = int64_t(uint64_t(idx[2u * ipos]) | (uint64_t(idx[2u * ipos + 1u]) << 32));
    if (v < 0l) v += int64_t(p.dim_size);
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        at += (uint(d) == p.dim ? uint(v) : coords[d]) * stride;
        stride *= p.in_shape[d];
    }
    uint slot = p.out_off + at;
    float add = g[p.g_off + i];
    uint seen = dst[slot], prev;
    do {
        prev = seen;
        seen = atomicCompSwap(dst[slot], prev, floatBitsToUint(uintBitsToFloat(prev) + add));
    } while (seen != prev);
}
