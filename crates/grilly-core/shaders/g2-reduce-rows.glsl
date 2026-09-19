#version 450
#extension GL_GOOGLE_include_directive : require
// Reduce contiguous rows: one workgroup per (row, chunk), so neighbouring
// threads read neighbouring addresses (g2-reduce gives each output one
// thread, which is uncoalesced when the reduced dim is the last one).
// Each thread folds its elements (stride 256) in order, Kahan-compensated
// for sums; the 256 partials then combine in a fixed tree with two-sum, so
// the result is deterministic. A long row with few rows is split into
// chunks: pass 1 writes (hi, lo) pairs per chunk, pass 2 reduces those.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { float src[]; };
layout(std430, binding = 1) buffer O { float dst[]; };
layout(push_constant) uniform P {
    uint rows; uint len; uint chunk; uint chunks; uint op; uint in_off; uint out_off;
    uint pairs_in;   // 1: the row holds (hi, lo) pairs from pass 1, len pairs
    uint pairs_out;  // 1: write (hi, lo) per chunk for pass 2; 0: the finished value
    uint count;      // elements behind each row, for MEAN
} p;

// Op codes; keep in sync with grilly_core::generic::Reduce.
const uint SUM = 0, MEAN = 1, MAX = 2, MIN = 3;

shared float s_hi[256];
shared float s_lo[256];

// (hi, lo) += v, Kahan style (lo carries the rounding error, true ~ hi + lo)
void kahan_add(inout float hi, inout float lo, float v) {
    precise float y = v + lo;
    precise float t = hi + y;
    lo = isinf(t) ? 0.0 : y - (t - hi);  // an infinite sum carries no error term
    hi = t;
}

// (a_hi, a_lo) += (b_hi, b_lo) with two-sum on the high parts
void pair_add(inout float a_hi, inout float a_lo, float b_hi, float b_lo) {
    precise float t = a_hi + b_hi;
    precise float bp = t - a_hi;
    precise float err = (a_hi - (t - bp)) + (b_hi - bp);
    a_lo = isinf(t) ? 0.0 : a_lo + b_lo + err;
    a_hi = t;
}

float extreme(float r, float v) {
    // torch propagates NaN through max/min
    return (isnan(v) || isnan(r)) ? NAN : (p.op == MAX ? max(r, v) : min(r, v));
}

void main() {
    uint g = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (g >= p.rows * p.chunks) return;  // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    uint row = g / p.chunks, c = g % p.chunks;
    uint start = c * p.chunk, stop = min(p.len, start + p.chunk);
    bool is_sum = p.op == SUM || p.op == MEAN;
    uint width = p.pairs_in == 1u ? 2u : 1u;
    uint base = p.in_off + row * p.len * width;

    float hi = is_sum ? 0.0 : (p.op == MAX ? uintBitsToFloat(0xff800000u) : uintBitsToFloat(0x7f800000u));
    float lo = 0.0;
    for (uint k = start + tid; k < stop; k += 256u) {
        if (p.pairs_in == 1u) {
            float h = src[base + 2u * k], l = src[base + 2u * k + 1u];
            if (is_sum) pair_add(hi, lo, h, l); else hi = extreme(hi, h);
        } else {
            float v = src[base + k];
            if (is_sum) kahan_add(hi, lo, v); else hi = extreme(hi, v);
        }
    }
    s_hi[tid] = hi;
    s_lo[tid] = lo;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float a_hi = s_hi[tid], a_lo = s_lo[tid];
            if (is_sum) pair_add(a_hi, a_lo, s_hi[tid + stride], s_lo[tid + stride]);
            else a_hi = extreme(a_hi, s_hi[tid + stride]);
            s_hi[tid] = a_hi;
            s_lo[tid] = a_lo;
        }
        barrier();
    }
    if (tid == 0u) {
        float h = s_hi[0], l = s_lo[0];
        if (p.pairs_out == 1u) {
            dst[p.out_off + 2u * g] = h;
            dst[p.out_off + 2u * g + 1u] = l;
        } else {
            float r = is_sum ? h + l : h;
            dst[p.out_off + g] = p.op == MEAN ? r / float(p.count) : r;
        }
    }
}
