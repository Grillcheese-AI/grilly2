#version 450
#extension GL_GOOGLE_include_directive : require
// Top-k of every row for small k, in one or two dispatches.
//
// The general path is a bitonic sort of the whole row and then a slice,
// which costs log2(n)*(log2(n)+1)/2 dispatches whatever k is: **108 for
// topk(8) over 16384**, and 1.02 ms for topk(64) over 128k. That is the
// wrong shape of work for a decode step, which asks for k=8 four times a
// token (an episodic read) and k<=64 once (top-k sampling).
//
// Sorting n to keep k is also the wrong algorithm. Here each thread walks
// a strided slice of its chunk keeping its own sorted best-k in shared
// memory — an insertion that the list's tail rejects in one comparison
// almost every time — and the 64 lists are then merged by a tournament,
// k rounds of a 64-wide tree. One workgroup per (row, chunk); a second
// dispatch merges the chunks when there is more than one, and rows that
// are short enough for a single chunk need only the first.
//
// The order is the bitonic path's, exactly, so this is a faster route to
// the same answer and not a second one: descending puts NaN first
// (torch's placement) and breaks ties by original position; ascending
// puts NaN last. `better()` is that total order.

layout(local_size_x = 64) in;
#include "g2-common.glsl"

// Per-thread best-k lists live in shared memory: 64 threads x 64 entries.
const uint THREADS = 64u;
const uint K_MAX = 64u;

layout(std430, binding = 0) readonly buffer V { float src_val[]; };
layout(std430, binding = 1) readonly buffer I { uint src_idx[]; };   // int64 pairs
layout(std430, binding = 2) buffer OV { float out_val[]; };
layout(std430, binding = 3) buffer OI { uint out_idx[]; };           // int64 pairs
layout(push_constant) uniform P {
    uint rows; uint n; uint k; uint chunk; uint chunks;
    uint in_off; uint descending; uint merging;
} p;

shared float s_val[THREADS * K_MAX];
shared uint s_idx[THREADS * K_MAX];
shared uint s_len[THREADS];

shared float m_val[THREADS];
shared uint m_idx[THREADS];
shared uint m_src[THREADS];
shared bool m_has[THREADS];
shared uint m_pos[THREADS];

/// The bitonic path's total order: is (va, ia) earlier than (vb, ib)?
bool better(float va, uint ia, float vb, uint ib) {
    bool na = isnan(va), nb = isnan(vb);
    if (na != nb) {
        return p.descending != 0u ? na : nb;   // NaN first descending, last ascending
    }
    if (!na && va != vb) {
        return p.descending != 0u ? va > vb : va < vb;
    }
    return ia < ib;                            // equal values keep their order
}

void main() {
    uint g = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    if (g >= p.rows * p.chunks) return;        // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    uint row = g / p.chunks, c = g % p.chunks;
    uint base = tid * K_MAX;

    uint start = c * p.chunk;
    uint stop = min(p.n, start + p.chunk);
    uint row_base = p.in_off + row * p.n;

    // ── each thread's own best-k over its strided slice ──────────────────
    s_len[tid] = 0u;
    uint len = 0u;
    for (uint e = start + tid; e < stop; e += THREADS) {
        float v = src_val[row_base + e];
        // the element's position in the ORIGINAL row: itself in pass one,
        // the index recorded by pass one when merging
        uint i = p.merging != 0u ? src_idx[2u * (row_base + e)] : e;
        if (len == p.k && !better(v, i, s_val[base + p.k - 1u], s_idx[base + p.k - 1u])) {
            continue;                          // the common case, one comparison
        }
        uint pos = min(len, p.k - 1u);
        while (pos > 0u && better(v, i, s_val[base + pos - 1u], s_idx[base + pos - 1u])) {
            s_val[base + pos] = s_val[base + pos - 1u];
            s_idx[base + pos] = s_idx[base + pos - 1u];
            pos -= 1u;
        }
        s_val[base + pos] = v;
        s_idx[base + pos] = i;
        len = min(len + 1u, p.k);
    }
    s_len[tid] = len;
    m_pos[tid] = 0u;
    barrier();

    // ── merge the 64 sorted lists, one winner per round ──────────────────
    uint out_base = g * p.k;
    for (uint o = 0u; o < p.k; ++o) {
        uint pos = m_pos[tid];
        bool has = pos < s_len[tid];
        m_has[tid] = has;
        m_src[tid] = tid;
        m_val[tid] = has ? s_val[base + pos] : 0.0;
        m_idx[tid] = has ? s_idx[base + pos] : 0u;
        barrier();
        for (uint st = THREADS / 2u; st > 0u; st >>= 1) {
            if (tid < st) {
                bool a = m_has[tid], b = m_has[tid + st];
                if (b && (!a || better(m_val[tid + st], m_idx[tid + st], m_val[tid], m_idx[tid]))) {
                    m_val[tid] = m_val[tid + st];
                    m_idx[tid] = m_idx[tid + st];
                    m_src[tid] = m_src[tid + st];
                    m_has[tid] = true;
                }
            }
            barrier();
        }
        if (tid == 0u && m_has[0]) {
            out_val[out_base + o] = m_val[0];
            out_idx[2u * (out_base + o)] = m_idx[0];
            out_idx[2u * (out_base + o) + 1u] = 0u;   // positions are non-negative
        }
        uint winner = m_src[0];
        bool any = m_has[0];
        barrier();
        if (!any) break;                       // fewer than k elements in this chunk
        if (tid == winner) {
            m_pos[tid] += 1u;
        }
        barrier();
    }
}
