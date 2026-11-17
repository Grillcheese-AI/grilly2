#version 450
// Shortlist head, dispatch 3 of 8 (grilly_infer::shortlist):
//
// - workgroups below `groups`: the clusters ranked by the context pass, by
//   counting rather than sorting. A cluster's rank is how many clusters go
//   before it — a higher score, or an equal one and a lower index, which is
//   numpy's stable argsort of the negated scores — so every cluster finds
//   its own rank at once, one thread each, over the scores held in shared
//   memory. Only ranks below `ranks` are written: the plan's order.
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

shared float s_sc[MAX_CLUSTERS];
shared uint s_scan[256];
shared uint s_found[2];

void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    if (g < p.groups) {
        for (uint i = t; i < p.clusters; i += 256u) s_sc[i] = scores[i];
        barrier();
        uint c = g * 256u + t;
        if (c < p.clusters) {
            float v = s_sc[c];
            uint rank = 0u;
            for (uint j = 0u; j < p.clusters; ++j) {
                float u = s_sc[j];
                rank += (u > v || (u == v && j < c)) ? 1u : 0u;
            }
            if (rank < p.ranks) plan[PLAN_ORDER + rank] = c;
        }
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
