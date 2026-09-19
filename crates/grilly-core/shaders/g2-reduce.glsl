#version 450
#extension GL_GOOGLE_include_directive : require
// Reduce the middle axis of a contiguous (outer, len, inner) view:
// out[o * inner + j] = op over r of in[(o * len + r) * inner + j].
// One invocation per (output, chunk of the reduced axis); neighbouring
// invocations take neighbouring columns j, so reads are coalesced. With
// one chunk this is the whole reduction; with several, pass 1 writes a
// (hi, lo) pair per chunk, laid out (outer, chunks, inner), and pass 2
// reduces those along the chunk axis. Sums are Kahan-compensated and
// combined with two-sum (g2-common), in a fixed order: deterministic.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { float src[]; };
layout(std430, binding = 1) buffer O { float dst[]; };
layout(push_constant) uniform P {
    uint outer; uint len; uint inner; uint op; uint in_off; uint out_off;
    uint chunk; uint chunks;
    uint pairs_in;   // 1: the input holds (hi, lo) pairs from pass 1
    uint pairs_out;  // 1: write (hi, lo) per chunk for pass 2
    uint count;      // elements behind each output, for MEAN
} p;

// Op codes; keep in sync with grilly_core::generic::Reduce.
const uint SUM = 0, MEAN = 1, MAX = 2, MIN = 3;

float extreme(float r, float v) {
    // torch propagates NaN through max/min
    return (isnan(v) || isnan(r)) ? NAN : (p.op == MAX ? max(r, v) : min(r, v));
}

void main() {
    uint t = flat_index();
    if (t >= p.outer * p.chunks * p.inner) return;
    uint j = t % p.inner, rest = t / p.inner;
    uint c = rest % p.chunks, o = rest / p.chunks;
    uint start = c * p.chunk, stop = min(p.len, start + p.chunk);
    bool is_sum = p.op == SUM || p.op == MEAN;
    float hi = is_sum ? 0.0 : (p.op == MAX ? uintBitsToFloat(0xff800000u) : uintBitsToFloat(0x7f800000u));
    float lo = 0.0;
    for (uint k = start; k < stop; ++k) {
        uint at = (o * p.len + k) * p.inner + j;
        if (p.pairs_in == 1u) {
            float h = src[2u * at], l = src[2u * at + 1u];
            if (is_sum) pair_add(hi, lo, h, l); else hi = extreme(hi, h);
        } else {
            float v = src[p.in_off + at];
            if (is_sum) kahan_add(hi, lo, v); else hi = extreme(hi, v);
        }
    }
    if (p.pairs_out == 1u) {
        uint at = (o * p.chunks + c) * p.inner + j;
        dst[2u * at] = hi;
        dst[2u * at + 1u] = lo;
    } else {
        float r = is_sum ? hi + lo : hi;
        dst[p.out_off + o * p.inner + j] = p.op == MEAN ? r / float(p.count) : r;
    }
}
