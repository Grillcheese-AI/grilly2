#version 450
#extension GL_GOOGLE_include_directive : require
// int8 weight-only quantization of a float32 (rows, k) weight, per row, as
// torchao's Int8WeightOnlyConfig with its default PerRow granularity:
//   scale = max|w| / 127.5, at least float32 eps
//   q     = clamp(roundEven(w * (1 / scale)), -128, 127)
// Both divisions are correctly rounded (div_rn), as torch's are, so the
// values match torchao's exactly. One workgroup per row. A row is packed
// four values to a word, little-endian two's complement, and padded with
// zeros to `words` words (a multiple of 4: whole 16-byte vectors).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer W { float w[]; };
layout(std430, binding = 1) writeonly buffer Q { uint q[]; };
layout(std430, binding = 2) writeonly buffer S { float scale[]; };
layout(push_constant) uniform P { uint rows; uint k; uint words; uint w_off; } p;

shared float s_max[256];

void main() {
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= p.rows) return;  // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    uint base = p.w_off + row * p.k;

    float m = 0.0;
    for (uint i = tid; i < p.k; i += 256u) m = max(m, abs(w[base + i]));
    s_max[tid] = m;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) s_max[tid] = max(s_max[tid], s_max[tid + stride]);
        barrier();
    }
    // torchao: max(-min, max) / ((127 - (-128)) / 2), clamped to >= finfo(float32).eps
    float sc = max(div_rn(s_max[0], 127.5), 1.1920928955078125e-7);
    float inv = div_rn(1.0, sc);

    for (uint word = tid; word < p.words; word += 256u) {
        uint packed = 0u;
        for (uint j = 0u; j < 4u; ++j) {
            uint i = word * 4u + j;
            if (i < p.k) {
                precise float v = w[base + i] * inv;
                int qv = int(clamp(roundEven(v), -128.0, 127.0));
                packed |= (uint(qv) & 0xFFu) << (8u * j);
            }
        }
        q[row * p.words + word] = packed;
    }
    if (tid == 0u) scale[row] = sc;
}
