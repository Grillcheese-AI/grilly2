#version 450
// Shortlist head, dispatch 4 of 8 (grilly_infer::shortlist):
//
// - workgroups below `ranks`, one per ranked cluster: every token of rank
//   r's list gets its position in the ranked concatenation, start(r) + i,
//   and each token keeps the smallest it is given: its first position. The
//   context list is then "the tokens with the K smallest first positions",
//   which is the host loop's "the ranked clusters' new tokens until K"
//   without visiting the clusters one after another. Ranks past the one
//   where K is reached only lower positions already past the cut, so
//   scanning a few more than needed changes nothing.
// - the workgroups after them, one per block of `tie_block` rows: how many
//   rows of the block are at the k-th distance T, for dispatch 6 to find
//   which tied rows are taken (the lowest, in row order).
//
// start(r), where rank r's list begins, is summed here from the lengths of
// the lists ranked before it rather than read from a prefix sum.

layout(local_size_x = 256) in;

const uint PLAN_ORDER = 16u;

layout(std430, binding = 0) readonly buffer Pl { uint plan[]; };
layout(std430, binding = 1) readonly buffer Co { uint coff[]; };
layout(std430, binding = 2) readonly buffer Ci { uint cids[]; };
layout(std430, binding = 3) buffer Fp { uint firstpos[]; };
layout(std430, binding = 4) readonly buffer D { uint dist[]; };
layout(std430, binding = 5) writeonly buffer Tc { uint tiecnt[]; };
layout(push_constant) uniform P { uint ranks; uint rows; uint tie_block; } p;

shared uint s_red[256];

/// Where rank r's list starts: the lengths of ranks 0..r-1, summed.
uint list_start(uint r, uint t) {
    uint sum = 0u;
    for (uint j = t; j < r; j += 256u) {
        uint c = plan[PLAN_ORDER + j];
        sum += coff[c + 1u] - coff[c];
    }
    s_red[t] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) s_red[t] += s_red[t + s];
        barrier();
    }
    return s_red[0];
}

void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    if (g < p.ranks) {
        uint base = list_start(g, t);
        uint c = plan[PLAN_ORDER + g];
        uint lo = coff[c], len = coff[c + 1u] - lo;
        for (uint i = t; i < len; i += 256u) atomicMin(firstpos[cids[lo + i]], base + i);
        return;
    }
    uint b = g - p.ranks, T = plan[0];
    uint lo = b * p.tie_block, hi = min(p.rows, lo + p.tie_block);
    uint n = 0u;
    for (uint r = lo + t; r < hi; r += 256u) n += dist[r] == T ? 1u : 0u;
    s_red[t] = n;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) s_red[t] += s_red[t + s];
        barrier();
    }
    if (t == 0u) tiecnt[b] = s_red[0];
}
