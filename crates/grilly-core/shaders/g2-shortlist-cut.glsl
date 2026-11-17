#version 450
// Shortlist head, dispatch 6 of 8 (grilly_infer::shortlist), one workgroup.
//
// Where the context list stops: a prefix sum over the clusters' counts of
// new tokens finds the rank J where the K-th distinct token appears, and an
// ordered count inside J's list finds which of its tokens that is. Every
// token whose first position is before the cut is in the list, which is
// exactly the host's "new tokens in rank order, the last cluster cut short
// at K". If the scanned ranks hold fewer than K distinct tokens, everything
// they hold is in and the short flag says so.
//
// Which rows at the k-th distance T are taken: the lowest `near - below` of
// them in row order, found from dispatch 4's count per block of rows and an
// ordered count inside the block where the last one falls.
//
// Then the previous token's successors join the union: their first position
// is set to 0, which is before any cut.

layout(local_size_x = 1024) in;

const uint PLAN_ORDER = 16u;
const uint NONE = 0xFFFFFFFFu;

layout(std430, binding = 0) buffer Pl { uint plan[]; };
layout(std430, binding = 1) readonly buffer Co { uint coff[]; };
layout(std430, binding = 2) readonly buffer Ci { uint cids[]; };
layout(std430, binding = 3) buffer Fp { uint firstpos[]; };
layout(std430, binding = 4) readonly buffer Cn { uint cnt[]; };
layout(std430, binding = 5) readonly buffer Su { int succ[]; };
layout(std430, binding = 6) readonly buffer D { uint dist[]; };
layout(std430, binding = 7) readonly buffer Tc { uint tiecnt[]; };
layout(push_constant) uniform P {
    uint ranks; uint context; uint rows; uint succ_width; uint prev; uint near; uint tie_block; uint blocks;
} p;

shared uint s_scan[1024];
shared uint s_found[6];

void scan(uint t) {
    for (uint off = 1u; off < 1024u; off <<= 1) {
        uint v = t >= off ? s_scan[t - off] : 0u;
        barrier();
        s_scan[t] += v;
        barrier();
    }
}

uint sum_all(uint v, uint t) {
    barrier();
    s_scan[t] = v;
    barrier();
    for (uint s = 512u; s > 0u; s >>= 1) {
        if (t < s) s_scan[t] += s_scan[t + s];
        barrier();
    }
    return s_scan[0];
}

void main() {
    uint t = gl_LocalInvocationID.x;

    // ── the context cut: J, and how many of J's new tokens to take ──────
    uint n = t < p.ranks ? cnt[t] : 0u;
    s_scan[t] = n;
    if (t == 0u) {
        s_found[0] = 0u; s_found[1] = 0u; s_found[2] = NONE;
        s_found[3] = 0u; s_found[4] = NONE; s_found[5] = 0u;
    }
    barrier();
    scan(t);
    uint incl = s_scan[t], excl = incl - n;
    bool short_list = s_scan[1023] < p.context;
    if (!short_list && excl < p.context && p.context <= incl) {
        s_found[0] = t;                        // J
        s_found[1] = p.context - excl;         // new tokens of J to take
    }
    barrier();
    uint J = short_list ? p.ranks : s_found[0], need = s_found[1];
    // where J's list starts (or, short, where the scanned ranks end)
    uint len_before = 0u;
    if (t < J) {
        uint c = plan[PLAN_ORDER + t];
        len_before = coff[c + 1u] - coff[c];
    }
    uint base = sum_all(len_before, t);

    uint c = short_list ? 0u : plan[PLAN_ORDER + J];
    uint lo = coff[c], len = short_list ? 0u : coff[c + 1u] - lo;
    uint chunk = (len + 1023u) / 1024u;
    uint ilo = min(len, t * chunk), ihi = min(len, ilo + chunk);
    uint mine = 0u;
    for (uint i = ilo; i < ihi; ++i) mine += firstpos[cids[lo + i]] == base + i ? 1u : 0u;
    barrier();
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

    // ── the tied rows taken: the lowest `near - below` at T ─────────────
    uint T = plan[0], tneed = p.near - plan[2];
    uint tn = t < p.blocks ? tiecnt[t] : 0u;
    barrier();
    s_scan[t] = tn;
    barrier();
    scan(t);
    uint tincl = s_scan[t], texcl = tincl - tn;
    bool all_ties = s_scan[1023] == tneed;
    if (!all_ties && texcl < tneed && tneed <= tincl) {
        s_found[3] = t;                        // the block
        s_found[5] = tneed - texcl;            // tied rows to take inside it
    }
    barrier();
    uint b = s_found[3], bneed = s_found[5];
    uint blo = b * p.tie_block, bhi = min(p.rows, blo + p.tie_block);
    uint bchunk = (bhi - blo + 1023u) / 1024u;
    uint rlo = min(bhi, blo + t * bchunk), rhi = min(bhi, rlo + bchunk);
    uint ties = 0u;
    if (!all_ties) {
        for (uint r = rlo; r < rhi; ++r) ties += dist[r] == T ? 1u : 0u;
    }
    barrier();
    s_scan[t] = ties;
    barrier();
    scan(t);
    uint rincl = s_scan[t], rexcl = rincl - ties;
    if (!all_ties && rexcl < bneed && bneed <= rincl) {
        uint k = rexcl;
        for (uint r = rlo; r < rhi; ++r) {
            if (dist[r] == T) {
                k += 1u;
                if (k == bneed) { s_found[4] = r; break; }
            }
        }
    }
    barrier();
    if (t == 0u) {
        plan[1] = all_ties ? NONE : s_found[4];
        // short: every position the scanned ranks gave is before the cut
        plan[3] = short_list ? base : s_found[2];
        plan[4] = short_list ? 1u : 0u;
    }
    if (p.prev < p.rows && t < p.succ_width) {
        int s = succ[p.prev * p.succ_width + t];
        if (s >= 0 && uint(s) < p.rows) firstpos[uint(s)] = 0u;
    }
}
