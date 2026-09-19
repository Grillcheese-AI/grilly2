#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// argmax / argmin along contiguous rows (g2-argreduce's inner == 1 case), a
// workgroup per row: each thread keeps the best (value, index) of a strided
// scan, then a tree combines them. torch's rule throughout: NaN is the
// extreme of both, and a tie goes to the lower index, so the answer is
// g2-argreduce's. float32 or int64 input, int64 index out.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { uint src[]; };
layout(std430, binding = 1) buffer O { uint dst[]; };
layout(push_constant) uniform P { uint rows; uint len; uint is_max; uint is_int; uint in_off; uint out_off; } p;

shared uint s_lo[256];  // value bits (float) or low word (int64)
shared uint s_hi[256];  // high word (int64)
shared uint s_at[256];  // index; NONE when the thread saw no element

const uint NONE = 0xFFFFFFFFu;

int64_t as_int(uint lo, uint hi) {
    return int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
}

// Whether candidate a (value, index) beats candidate b, as torch decides.
bool beats(uint alo, uint ahi, uint aat, uint blo, uint bhi, uint bat) {
    if (aat == NONE) return false;
    if (bat == NONE) return true;
    if (p.is_int == 1u) {
        int64_t a = as_int(alo, ahi), b = as_int(blo, bhi);
        if (a == b) return aat < bat;
        return p.is_max == 1u ? a > b : a < b;
    }
    float a = uintBitsToFloat(alo), b = uintBitsToFloat(blo);
    if (isnan(a) || isnan(b)) return isnan(a) && (!isnan(b) || aat < bat);
    if (a == b) return aat < bat;
    return p.is_max == 1u ? a > b : a < b;
}

void main() {
    uint row = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (row >= p.rows) return;  // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    uint base = p.in_off + row * p.len;

    uint lo = 0u, hi = 0u, at = NONE;
    for (uint k = tid; k < p.len; k += 256u) {
        uint vlo = p.is_int == 1u ? src[2u * (base + k)] : src[base + k];
        uint vhi = p.is_int == 1u ? src[2u * (base + k) + 1u] : 0u;
        if (beats(vlo, vhi, k, lo, hi, at)) {
            lo = vlo;
            hi = vhi;
            at = k;
        }
    }
    s_lo[tid] = lo;
    s_hi[tid] = hi;
    s_at[tid] = at;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            uint o = tid + stride;
            if (beats(s_lo[o], s_hi[o], s_at[o], s_lo[tid], s_hi[tid], s_at[tid])) {
                s_lo[tid] = s_lo[o];
                s_hi[tid] = s_hi[o];
                s_at[tid] = s_at[o];
            }
        }
        barrier();
    }
    if (tid == 0u) {
        dst[2u * (p.out_off + row)] = s_at[0] == NONE ? 0u : s_at[0];
        dst[2u * (p.out_off + row) + 1u] = 0u;
    }
}
