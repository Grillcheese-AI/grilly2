#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// dst[i] = convert(src[i]) between float32, int64, int32 and bool, with
// torch's rules: float -> int truncates toward zero; anything -> bool is
// (value != 0); bool -> number is 0 or 1. Buffers are raw 32-bit words;
// int64 elements are two words, little-endian.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer S { uint src[]; };
layout(std430, binding = 1) buffer D { uint dst[]; };
layout(push_constant) uniform P { uint n; uint from; uint to; uint src_off; uint dst_off; } p;

// Codes; keep in sync with grilly_core::DType::code.
const uint F32 = 0, I64 = 1, I32 = 2, BOOL = 3;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    // Read as int64 (exact for every integer source) or float.
    bool is_float = p.from == F32;
    float f = 0.0;
    int64_t v = 0;
    if (p.from == F32) {
        f = uintBitsToFloat(src[p.src_off + i]);
    } else if (p.from == I64) {
        uint k = p.src_off + 2u * i;
        v = int64_t(uint64_t(src[k]) | (uint64_t(src[k + 1u]) << 32));
    } else if (p.from == I32) {
        v = int64_t(int(src[p.src_off + i]));
    } else {
        v = src[p.src_off + i] != 0u ? 1l : 0l;
    }
    if (p.to == F32) {
        dst[p.dst_off + i] = floatBitsToUint(is_float ? f : float(v));
    } else if (p.to == BOOL) {
        dst[p.dst_off + i] = (is_float ? f != 0.0 : v != 0l) ? 1u : 0u;
    } else {
        int64_t r = is_float ? int64_t(f) : v;   // conversion truncates toward zero
        if (p.to == I32) {
            dst[p.dst_off + i] = uint(int(r));
        } else {
            uint k = p.dst_off + 2u * i;
            dst[k] = uint(uint64_t(r) & 0xffffffffUL);
            dst[k + 1u] = uint(uint64_t(r) >> 32);
        }
    }
}
