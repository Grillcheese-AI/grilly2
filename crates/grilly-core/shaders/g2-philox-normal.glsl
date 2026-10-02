#version 450
#extension GL_GOOGLE_include_directive : require
// Standard normals from a counter-based stream: Philox4x32-10 keyed by
// (key, stream), with the element index as the counter.
//
// Counter-based is the whole point. A sequential generator has to be
// *stepped* to reach a value, so reproducing one means either storing it
// or replaying everything before it; a counter-based one computes value i
// from i alone. That is what lets evolution strategies regenerate a
// population's perturbations at update time instead of holding N copies
// of every weight — the memory claim EGGROLL rests on.
//
// It also keeps the noise on the device. `Generator::normal` builds its
// values on the host and uploads them, which is a host write: refused
// inside a capture, and a transfer per call besides.
//
// **This stream is not `Generator`'s.** That one does Box-Muller in f64
// on the host; this does it in f32, because f64 on RDNA2 runs at 1/16
// rate and no caller needs the last bits of a Gaussian. Same algorithm,
// same key schedule, different rounding — so `randn` is unchanged and
// this is a second, documented stream. What it does guarantee is what it
// is for: the same (key, stream, index) gives the same value, on any
// machine, forever.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) writeonly buffer O { float dst[]; };
layout(push_constant) uniform P {
    uint n; uint out_off; uint key0; uint key1; uint stream0; uint stream1; uint std_bits;
} p;

const uint M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
const uint W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;

uvec2 mulhilo(uint a, uint b) {
    // 32x32 -> 64 without 64-bit ints: split b into halves.
    uint lo = a * b;
    uint a0 = a & 0xffffu, a1 = a >> 16;
    uint b0 = b & 0xffffu, b1 = b >> 16;
    uint t = a0 * b0;
    uint k = t >> 16;
    t = a1 * b0 + k;
    uint w1 = t & 0xffffu, w2 = t >> 16;
    t = a0 * b1 + w1;
    uint hi = a1 * b1 + w2 + (t >> 16);
    return uvec2(hi, lo);
}

uvec4 philox(uvec4 ctr, uvec2 key) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            key.x += W0;
            key.y += W1;
        }
        uvec2 p0 = mulhilo(M0, ctr.x);
        uvec2 p1 = mulhilo(M1, ctr.z);
        ctr = uvec4(p1.x ^ ctr.y ^ key.x, p1.y, p0.x ^ ctr.w ^ key.y, p0.y);
    }
    return ctr;
}

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    // four normals per block: (x, y) make one Box-Muller pair, (z, w) the
    // other, exactly as the host generator lays them out
    uvec4 r = philox(uvec4(i >> 2, p.stream0, p.stream1, 0u), uvec2(p.key0, p.key1));
    uint j = i & 3u;
    uint a = j < 2u ? r.x : r.z;
    uint b = j < 2u ? r.y : r.w;
    // u in (0, 1], t in [0, 1) — u never 0, so log(u) is finite
    float u = float((a >> 8) + 1u) * 5.9604645e-8;
    float t = float(b >> 8) * 5.9604645e-8;
    float radius = sqrt(-2.0 * log(u));
    float angle = 6.2831853071795864769 * t;
    float z = (j & 1u) == 0u ? radius * cos(angle) : radius * sin(angle);
    dst[p.out_off + i] = uintBitsToFloat(p.std_bits) * z;
}
