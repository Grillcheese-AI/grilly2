#version 450
#extension GL_GOOGLE_include_directive : require
// dst[dst_off + sum c_d * ds_d] = src[src_off + sum c_d * ss_d] for every
// coordinate c of `shape`. Materializes transposes, broadcasts and slices,
// and writes into slices (cat, stack, __setitem__).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer S { float src[]; };
layout(std430, binding = 1) buffer D { float dst[]; };
layout(push_constant) uniform P {
    uint n; uint ndim; uint src_off; uint dst_off;
    uint shape[MAX_DIMS]; uint ss[MAX_DIMS]; uint ds[MAX_DIMS];
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint rem = i, is = p.src_off, id = p.dst_off;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        uint c = rem % p.shape[d];
        rem /= p.shape[d];
        is += c * p.ss[d];
        id += c * p.ds[d];
    }
    dst[id] = src[is];
}
