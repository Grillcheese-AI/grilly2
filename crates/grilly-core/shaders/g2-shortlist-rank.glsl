#version 450
// Shortlist head, dispatch 3 of 8 (grilly_infer::shortlist):
//
// - workgroups below `groups`: the clusters ranked by the context pass, by
//   counting rather than sorting. A cluster's rank is how many clusters go
//   before it — a higher score, or an equal one and a lower index, which is
//   numpy's stable argsort of the negated scores — so every cluster finds
//   its own rank at once over the scores held in shared memory: 64 clusters
//   a workgroup, four threads each counting a quarter of the scores, four
//   at a time. (One thread a cluster walking all 4,096 one at a time was
//   latency-bound: 0.2 ms on an RX 6750 XT.) Only ranks below `ranks` are
//   written: the plan's order.
// - the last workgroup: the k-th smallest Hamming distance, T, and how many
//   rows lie below it, from the histogram. Which rows at T are taken (ties
//   to the lower row) is settled by dispatches 4 and 6.
//
// An earlier version did all of this in one workgroup with a bitonic sort
// of the 4,096 scores and an ordered count over every row's distance, and
// that one workgroup was a third of the step's time on an RX 6750 XT.
//
// Plan buffer (uint): [0] T, [1] last tied row taken (dispatch 6;
// 0xFFFFFFFF = all), [2] rows below T, [3] context cut (dispatch 6),
// [4] short flag (dispatch 6); [16 + r] the cluster at rank r, r < ranks.

layout(local_size_x = 256) in;

const uint MAX_CLUSTERS = 4096u;
const uint PLAN_ORDER = 16u;

layout(std430, binding = 0) readonly buffer H { uint hist[]; };
layout(std430, binding = 1) readonly buffer S { float scores[]; };
layout(std430, binding = 2) writeonly buffer Pl { uint plan[]; };
layout(push_constant) uniform P { uint dim; uint near; uint clusters; uint ranks; uint groups; } p;

const uint PER = 64u;                           // clusters a workgroup

shared vec4 s_sc[MAX_CLUSTERS / 4u];
shared uint s_rank[PER];
shared uint s_scan[256];
shared uint s_found[2];

void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    if (g < p.groups) {
        // padding is -inf at an index past every cluster: never before one
        uint n4 = (p.clusters + 3u) / 4u;
        float ninf = uintBitsToFloat(0xff800000u);
        for (uint i = t; i < n4; i += 256u) {
            uint j = 4u * i;
            s_sc[i] = vec4(j < p.clusters ? scores[j] : ninf, j + 1u < p.clusters ? scores[j + 1u] : ninf,
                           j + 2u < p.clusters ? scores[j + 2u] : ninf, j + 3u < p.clusters ? scores[j + 3u] : ninf);
        }
        if (t < PER) s_rank[t] = 0u;
        barrier();
        uint c = g * PER + t / 4u, part = t % 4u;
        if (c < p.clusters) {
            float v = scores[c];
            uint q = (n4 + 3u) / 4u;
            uint lo = part * q, hi = min(n4, lo + q);
            uint rank = 0u;
            for (uint i = lo; i < hi; ++i) {
                vec4 u = s_sc[i];
                uint j = 4u * i;
                rank += (u.x > v || (u.x == v && j < c)) ? 1u : 0u;
                rank += (u.y > v || (u.y == v && j + 1u < c)) ? 1u : 0u;
                rank += (u.z > v || (u.z == v && j + 2u < c)) ? 1u : 0u;
                rank += (u.w > v || (u.w == v && j + 3u < c)) ? 1u : 0u;
            }
            atomicAdd(s_rank[t / 4u], rank);
        }
        barrier();
        uint mine = g * PER + t;
        if (t < PER && mine < p.clusters && s_rank[t] < p.ranks) plan[PLAN_ORDER + s_rank[t]] = mine;
        return;
    }

    // the k-th smallest distance, from the histogram
    uint bins = p.dim + 1u;
    uint chunk = (bins + 255u) / 256u;
    uint lo = min(bins, t * chunk), hi = min(bins, lo + chunk);
    uint sum = 0u;
    for (uint b = lo; b < hi; ++b) sum += hist[b];
    s_scan[t] = sum;
    if (t == 0u) { s_found[0] = bins; s_found[1] = 0u; }
    barrier();
    for (uint off = 1u; off < 256u; off <<= 1) {
        uint v = t >= off ? s_scan[t - off] : 0u;
        barrier();
        s_scan[t] += v;
        barrier();
    }
    uint incl = s_scan[t], excl = incl - sum;
    if (excl < p.near && p.near <= incl) {
        uint cum = excl;
        for (uint b = lo; b < hi; ++b) {
            uint h = hist[b];
            if (cum + h >= p.near) { s_found[0] = b; s_found[1] = cum; break; }
            cum += h;
        }
    }
    barrier();
    if (t == 0u) {
        plan[0] = s_found[0];
        plan[2] = s_found[1];
    }
}
