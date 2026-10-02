#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// dst[i] = start + i * step, for float32, int64 or int32.
//
// A counting sequence is the one tensor that never needs to be sent: every
// element is a function of its own index. Building it on the host and
// uploading it cost a transfer per call — and, worse, it was a **host write**,
// which a capture refuses, so a `torch.arange` anywhere inside a decode step
// put the whole step outside `graphed`. Masks are written with arange
// (`arange(n) < length`), so that was most of them.
//
// float32 takes start/step as bit patterns; the integer dtypes take them as
// two words each, little-endian, so the whole int64 range survives the push.
//
// With `step` zero this is also a **fill**, which is the only device-side
// way to write an 8-byte constant: `vkCmdFillBuffer` takes a 4-byte
// pattern, so int64 fills used to upload one element and broadcast it —
// a host write, refused inside a capture. That made `torch.tensor(1)` and
// even `x % 512` on an int64 tensor uncapturable.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer D { uint dst[]; };
layout(push_constant) uniform P {
    uint n; uint dtype; uint start_lo; uint start_hi; uint step_lo; uint step_hi; uint dst_off;
} p;

// Codes; keep in sync with grilly_core::DType::code.
const uint F32 = 0, I64 = 1, I32 = 2;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    if (p.dtype == F32) {
        float start = uintBitsToFloat(p.start_lo), step = uintBitsToFloat(p.step_lo);
        dst[p.dst_off + i] = floatBitsToUint(start + float(i) * step);
        return;
    }
    int64_t start = int64_t(uint64_t(p.start_lo) | (uint64_t(p.start_hi) << 32));
    int64_t step = int64_t(uint64_t(p.step_lo) | (uint64_t(p.step_hi) << 32));
    int64_t v = start + int64_t(i) * step;
    if (p.dtype == I32) {
        dst[p.dst_off + i] = uint(int(v));
    } else {
        dst[2u * (p.dst_off + i)] = uint(uint64_t(v) & 0xffffffffUL);
        dst[2u * (p.dst_off + i) + 1u] = uint(uint64_t(v) >> 32);
    }
}
