#version 450
// Shortlist head, dispatch 4 of 8 (grilly_infer::shortlist): one workgroup
// per ranked cluster, up to the head's rank bound. Every token of rank r's
// list gets its position in the ranked concatenation, start[r] + i, and
// each token keeps the smallest it is given: its first position.
//
// The context list is then "the tokens with the K smallest first
// positions", which is the host loop's "the ranked clusters' new tokens
// until K" without visiting the clusters one after another. Ranks past the
// one where K is reached only lower positions that are already past the
// cut, so scanning a few more than needed changes nothing.

layout(local_size_x = 256) in;

const uint PLAN_ORDER = 16u;

layout(std430, binding = 0) readonly buffer Pl { uint plan[]; };
layout(std430, binding = 1) readonly buffer Co { uint coff[]; };
layout(std430, binding = 2) readonly buffer Ci { uint cids[]; };
layout(std430, binding = 3) buffer Fp { uint firstpos[]; };
layout(push_constant) uniform P { uint clusters; } p;

void main() {
    uint r = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint c = plan[PLAN_ORDER + r];
    uint base = plan[PLAN_ORDER + p.clusters + r];
    uint lo = coff[c], len = coff[c + 1u] - lo;
    for (uint i = t; i < len; i += 256u) atomicMin(firstpos[cids[lo + i]], base + i);
}
