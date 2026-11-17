#version 450
// Shortlist head, dispatch 5 of 8 (grilly_infer::shortlist): one workgroup
// per ranked cluster. How many of rank r's tokens first appear in rank r's
// list (their first position is their own), which is how many new tokens
// that cluster adds to the context list.

layout(local_size_x = 256) in;

const uint PLAN_ORDER = 16u;

layout(std430, binding = 0) readonly buffer Pl { uint plan[]; };
layout(std430, binding = 1) readonly buffer Co { uint coff[]; };
layout(std430, binding = 2) readonly buffer Ci { uint cids[]; };
layout(std430, binding = 3) readonly buffer Fp { uint firstpos[]; };
layout(std430, binding = 4) writeonly buffer Cn { uint cnt[]; };
layout(push_constant) uniform P { uint clusters; } p;

shared uint s_n;

void main() {
    uint r = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    if (t == 0u) s_n = 0u;
    barrier();
    uint c = plan[PLAN_ORDER + r];
    uint base = plan[PLAN_ORDER + p.clusters + r];
    uint lo = coff[c], len = coff[c + 1u] - lo;
    uint mine = 0u;
    for (uint i = t; i < len; i += 256u) mine += firstpos[cids[lo + i]] == base + i ? 1u : 0u;
    if (mine != 0u) atomicAdd(s_n, mine);
    barrier();
    if (t == 0u) cnt[r] = s_n;
}
