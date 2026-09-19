#version 450
#extension GL_GOOGLE_include_directive : require
// int4 weight-only quantization, group-wise and asymmetric, as torchao's
// Int4WeightOnlyConfig does it (group_size 128 by default):
//
//   scale = (max - min) / 15          over each group of `group` values
//   zero  = min + 8 * scale           (torchao's float zero-point domain)
//   q     = clamp(round_half_even((w - min) / scale), 0, 15)
//   w'    = (q - 8) * scale + zero    = q * scale + min
//
// One invocation per (row, group). A group is a whole number of 8-value
// words, so each invocation builds and writes its own words and nothing is
// shared. The division is correctly rounded (div_rn), as it is on the CPU,
// so the values are the formula's exactly.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

const float FLT_EPS = 1.1920928955078125e-07;

layout(std430, binding = 0) readonly buffer W { float w[]; };
layout(std430, binding = 1) writeonly buffer Q { uint q[]; };
layout(std430, binding = 2) writeonly buffer S { float scale[]; };
layout(std430, binding = 3) writeonly buffer Z { float zero[]; };
layout(push_constant) uniform P {
    uint rows; uint k; uint group; uint words;  // words: 32-bit words per quantized row
    uint groups; uint w_off;                    // groups: groups per row
} p;

void main() {
    uint id = flat_index();
    if (id >= p.rows * p.groups) return;
    uint row = id / p.groups, g = id % p.groups;
    uint start = g * p.group;
    uint end = min(p.k, start + p.group);
    uint base = p.w_off + row * p.k;
    if (start >= end) {  // a group past the end of a short last row
        scale[id] = 1.0;
        zero[id] = 0.0;
        return;
    }

    float lo = w[base + start], hi = lo;
    for (uint i = start + 1u; i < end; ++i) {
        float v = w[base + i];
        lo = min(lo, v);
        hi = max(hi, v);
    }
    float s = max(div_rn(hi - lo, 15.0), FLT_EPS);  // native / is up to 2.5 ulp off here
    scale[id] = s;
    zero[id] = lo + 8.0 * s;

    // the last group also clears the row's padding, so a whole 16-byte
    // vector is always defined
    if (end == p.k) {
        for (uint word = (p.k + 7u) >> 3; word < p.words; ++word) q[row * p.words + word] = 0u;
    }
    // whole words: the group starts on a word boundary and covers `group`
    // values, and only the row's last word can be partial
    for (uint word = start >> 3; word < (end + 7u) >> 3; ++word) {
        uint packed = 0u;
        for (uint j = 0u; j < 8u; ++j) {
            uint i = word * 8u + j;
            if (i >= end) break;  // the rest of the word stays zero
            int level = int(clamp(roundEven(div_rn(w[base + i] - lo, s)), 0.0, 15.0));
            packed |= uint(level) << (j * 4u);
        }
        q[row * p.words + word] = packed;
    }
}
