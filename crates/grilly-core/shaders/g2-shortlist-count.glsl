#version 450
// Shortlist head, dispatch 5 of 8 (grilly_infer::shortlist): one workgroup
// per ranked cluster. How many of rank r's tokens first appear in rank r's
// list (their first position is their own), which is how many new tokens
// that cluster adds to the context list. start(r) as in g2-shortlist-mark.

layout(local_size_x = 256) in;

const uint PLAN_ORDER = 16u;

layout(std430, binding = 0) readonly buffer Pl { uint plan[]; };
layout(std430, binding = 1) readonly buffer Co { uint coff[]; };
layout(std430, binding = 2) readonly buffer Ci { uint cids[]; };
layout(std430, binding = 3) readonly buffer Fp { uint firstpos[]; };
layout(std430, binding = 4) writeonly buffer Cn { uint cnt[]; };
layout(push_constant) uniform P { uint ranks; } p;

shared uint s_red[256];

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
    uint r = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint base = list_start(r, t);
    uint c = plan[PLAN_ORDER + r];
    uint lo = coff[c], len = coff[c + 1u] - lo;
    uint mine = 0u;
    for (uint i = t; i < len; i += 256u) mine += firstpos[cids[lo + i]] == base + i ? 1u : 0u;
    barrier();                                  // s_red is reused
    s_red[t] = mine;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) s_red[t] += s_red[t + s];
        barrier();
    }
    if (t == 0u) cnt[r] = s_red[0];
}
