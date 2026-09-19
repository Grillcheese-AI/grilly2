#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// int64 sum / max / min over the middle axis of a contiguous
// (outer, len, inner) view, exactly; one invocation per output.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { uint src[]; };
layout(std430, binding = 1) buffer O { uint dst[]; };
layout(push_constant) uniform P { uint outer; uint len; uint inner; uint op; uint in_off; uint out_off; } p;

// Op codes; keep in sync with grilly_core::generic::Reduce (Sum, _, Max, Min).
const uint SUM = 0, MAX = 2, MIN = 3;

int64_t load(uint k) {
    return int64_t(uint64_t(src[2u * k]) | (uint64_t(src[2u * k + 1u]) << 32));
}

void main() {
    uint i = flat_index();
    if (i >= p.outer * p.inner) return;
    uint o = i / p.inner, j = i % p.inner;
    uint base = p.in_off + o * p.len * p.inner + j;
    int64_t r = p.op == SUM ? 0l : load(base);
    for (uint k = (p.op == SUM ? 0u : 1u); k < p.len; ++k) {
        int64_t v = load(base + k * p.inner);
        r = p.op == SUM ? r + v : (p.op == MAX ? max(r, v) : min(r, v));
    }
    uint out_k = 2u * (p.out_off + i);
    dst[out_k] = uint(uint64_t(r) & 0xffffffffUL);
    dst[out_k + 1u] = uint(uint64_t(r) >> 32);
}
