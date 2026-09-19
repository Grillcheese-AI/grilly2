#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// argmax / argmin over the middle axis of a contiguous (outer, len, inner)
// view, as torch: the first extreme wins, and NaN is the extreme of both.
// float32 or int64 input, int64 index out; one invocation per output.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { uint src[]; };
layout(std430, binding = 1) buffer O { uint dst[]; };
layout(push_constant) uniform P { uint outer; uint len; uint inner; uint is_max; uint is_int; uint in_off; uint out_off; } p;

int64_t load_int(uint k) {
    return int64_t(uint64_t(src[2u * k]) | (uint64_t(src[2u * k + 1u]) << 32));
}

// Whether v replaces the running extreme r (strictly: ties keep the first).
bool beats(float v, float r) {
    if (isnan(r)) return false;
    if (isnan(v)) return true;
    return p.is_max == 1u ? v > r : v < r;
}

void main() {
    uint i = flat_index();
    if (i >= p.outer * p.inner) return;
    uint o = i / p.inner, j = i % p.inner;
    uint base = p.in_off + o * p.len * p.inner + j;
    uint best = 0u;
    if (p.is_int == 1u) {
        int64_t r = load_int(base);
        for (uint k = 1u; k < p.len; ++k) {
            int64_t v = load_int(base + k * p.inner);
            if (p.is_max == 1u ? v > r : v < r) { r = v; best = k; }
        }
    } else {
        float r = uintBitsToFloat(src[base]);
        for (uint k = 1u; k < p.len; ++k) {
            float v = uintBitsToFloat(src[base + k * p.inner]);
            if (beats(v, r)) { r = v; best = k; }
        }
    }
    dst[2u * (p.out_off + i)] = best;
    dst[2u * (p.out_off + i) + 1u] = 0u;
}
