#version 450
// Shortlist head, dispatch 6 of 8 (grilly_infer::shortlist), one workgroup:
// where the context list stops. A prefix sum over the clusters' counts of
// new tokens finds the rank J where the K-th distinct token appears; an
// ordered count inside J's list finds which of its tokens that is. Every
// token whose first position is before the cut is in the list, which is
// exactly the host's "new tokens in rank order, the last cluster cut short
// at K".
//
// If the scanned ranks hold fewer than K distinct tokens, everything they
// hold is in and the short flag says so; the caller decides what a short
// list means (the head reports it with every token).
//
// Then the previous token's successors join the union: their first position
// is set to 0, which is before any cut.

layout(local_size_x = 1024) in;

const uint PLAN_ORDER = 16u;

layout(std430, binding = 0) buffer Pl { uint plan[]; };
layout(std430, binding = 1) readonly buffer Co { uint coff[]; };
layout(std430, binding = 2) readonly buffer Ci { uint cids[]; };
layout(std430, binding = 3) buffer Fp { uint firstpos[]; };
layout(std430, binding = 4) readonly buffer Cn { uint cnt[]; };
layout(std430, binding = 5) readonly buffer Su { int succ[]; };
layout(push_constant) uniform P {
    uint clusters; uint ranks; uint context; uint rows; uint succ_width; uint prev;
} p;

shared uint s_scan[1024];
shared uint s_found[4];

void scan(uint t) {
    for (uint off = 1u; off < 1024u; off <<= 1) {
        uint v = t >= off ? s_scan[t - off] : 0u;
        barrier();
        s_scan[t] += v;
        barrier();
    }
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint n = t < p.ranks ? cnt[t] : 0u;
    s_scan[t] = n;
    if (t == 0u) { s_found[0] = 0xFFFFFFFFu; s_found[1] = 0u; s_found[2] = 0xFFFFFFFFu; }
    barrier();
    scan(t);
    uint incl = s_scan[t], excl = incl - n;
    bool short_list = s_scan[1023] < p.context;
    if (!short_list && excl < p.context && p.context <= incl) {
        s_found[0] = t;                        // J
        s_found[1] = p.context - excl;         // new tokens of J to take
    }
    barrier();
    uint J = s_found[0], need = s_found[1];

    // the need-th new token of J's list, by an ordered count
    uint c = short_list ? 0u : plan[PLAN_ORDER + J];
    uint base = short_list ? 0u : plan[PLAN_ORDER + p.clusters + J];
    uint lo = coff[c], len = short_list ? 0u : coff[c + 1u] - lo;
    uint chunk = (len + 1023u) / 1024u;
    uint ilo = min(len, t * chunk), ihi = min(len, ilo + chunk);
    uint mine = 0u;
    for (uint i = ilo; i < ihi; ++i) mine += firstpos[cids[lo + i]] == base + i ? 1u : 0u;
    barrier();                                 // s_scan is reused
    s_scan[t] = mine;
    barrier();
    scan(t);
    uint mincl = s_scan[t], mexcl = mincl - mine;
    if (!short_list && mexcl < need && need <= mincl) {
        uint k = mexcl;
        for (uint i = ilo; i < ihi; ++i) {
            if (firstpos[cids[lo + i]] == base + i) {
                k += 1u;
                if (k == need) { s_found[2] = base + i + 1u; break; }
            }
        }
    }
    barrier();
    if (t == 0u) {
        // short: every position the scanned ranks gave is before the cut
        plan[3] = short_list ? plan[PLAN_ORDER + p.clusters + p.ranks] : s_found[2];
        plan[4] = short_list ? 1u : 0u;
    }
    if (p.prev < p.rows && t < p.succ_width) {
        int s = succ[p.prev * p.succ_width + t];
        if (s >= 0 && uint(s) < p.rows) firstpos[uint(s)] = 0u;
    }
}
