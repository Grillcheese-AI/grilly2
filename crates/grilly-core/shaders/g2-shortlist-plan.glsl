#version 450
// Shortlist head, dispatch 3 of 8 (grilly_infer::shortlist), one workgroup:
//
// 1. the k nearest rows without a sort. The histogram gives T, the k-th
//    smallest distance, and how many rows lie below it; the rows at T are
//    then taken in id order up to k (ties to the lower id, as
//    `torch.topk(largest=False)` and numpy's stable argsort break them), by
//    an ordered count over contiguous chunks of `dist`. The result is two
//    numbers, T and the last tied id taken, not a list.
// 2. the clusters ranked by the context pass: score descending, then index
//    ascending (a stable sort of the negated scores), by a bitonic sort of
//    up to 4,096 in shared memory;
// 3. where each ranked cluster's list starts in their concatenation, which
//    is the position dispatch 4 gives each of its tokens.
//
// Plan buffer (uint): [0] T, [1] last tied id taken (0xFFFFFFFF = all),
// [2] rows below T, [3] context cut (dispatch 6), [4] short flag
// (dispatch 6); [16 + r] the cluster at rank r; [16 + C + r] where rank r's
// list starts, r = 0..C.

layout(local_size_x = 1024) in;

const uint MAX_CLUSTERS = 4096u;
const uint PLAN_ORDER = 16u;

layout(std430, binding = 0) readonly buffer H { uint hist[]; };
layout(std430, binding = 1) readonly buffer D { uint dist[]; };
layout(std430, binding = 2) readonly buffer S { float scores[]; };
layout(std430, binding = 3) readonly buffer Co { uint coff[]; };
layout(std430, binding = 4) writeonly buffer Pl { uint plan[]; };
layout(push_constant) uniform P { uint rows; uint dim; uint near; uint clusters; uint padded; } p;

shared uint s_key[MAX_CLUSTERS];
shared uint s_idx[MAX_CLUSTERS];
shared uint s_scan[1024];
shared uint s_found[4];

/// Inclusive prefix sum of s_scan over the workgroup.
void scan(uint t) {
    for (uint off = 1u; off < 1024u; off <<= 1) {
        uint v = t >= off ? s_scan[t - off] : 0u;
        barrier();
        s_scan[t] += v;
        barrier();
    }
}

/// Float bits as a uint that sorts in the same order (finite values).
uint orderable(float f) {
    uint b = floatBitsToUint(f);
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

/// Does entry a go before entry b: higher score first, lower index on a tie.
bool before(uint a, uint b) {
    return s_key[a] < s_key[b] || (s_key[a] == s_key[b] && s_idx[a] < s_idx[b]);
}

void main() {
    uint t = gl_LocalInvocationID.x;

    // ── 1. the k-th smallest distance, T ─────────────────────────────────
    uint bins = p.dim + 1u;
    uint chunk = (bins + 1023u) / 1024u;
    uint lo = min(bins, t * chunk), hi = min(bins, lo + chunk);
    uint sum = 0u;
    for (uint b = lo; b < hi; ++b) sum += hist[b];
    s_scan[t] = sum;
    if (t == 0u) { s_found[0] = bins; s_found[1] = 0u; s_found[2] = 0xFFFFFFFFu; }
    barrier();
    scan(t);
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
    uint T = s_found[0], below = s_found[1];
    uint need = p.near - below;                 // rows at T to take, >= 1

    // ... and the last tied row taken: an ordered count of dist == T
    uint rchunk = (p.rows + 1023u) / 1024u;
    uint rlo = min(p.rows, t * rchunk), rhi = min(p.rows, rlo + rchunk);
    uint ties = 0u;
    for (uint r = rlo; r < rhi; ++r) ties += dist[r] == T ? 1u : 0u;
    barrier();                                  // s_scan is reused
    s_scan[t] = ties;
    barrier();
    scan(t);
    uint tincl = s_scan[t], texcl = tincl - ties;
    bool all_ties = s_scan[1023] == need;
    if (!all_ties && texcl < need && need <= tincl) {
        uint c = texcl;
        for (uint r = rlo; r < rhi; ++r) {
            if (dist[r] == T) {
                c += 1u;
                if (c == need) { s_found[2] = r; break; }
            }
        }
    }

    // ── 2. the clusters ranked ───────────────────────────────────────────
    for (uint i = t; i < p.padded; i += 1024u) {
        // ascending on ~orderable is descending on the score; padding last
        s_key[i] = i < p.clusters ? ~orderable(scores[i]) : 0xFFFFFFFFu;
        s_idx[i] = i;
    }
    barrier();
    for (uint k = 2u; k <= p.padded; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            for (uint i = t; i < p.padded; i += 1024u) {
                uint l = i ^ j;
                if (l > i) {
                    bool up = (i & k) == 0u;
                    if (up ? before(l, i) : before(i, l)) {
                        uint a = s_key[i]; s_key[i] = s_key[l]; s_key[l] = a;
                        uint b = s_idx[i]; s_idx[i] = s_idx[l]; s_idx[l] = b;
                    }
                }
            }
            barrier();
        }
    }
    for (uint r = t; r < p.clusters; r += 1024u) plan[PLAN_ORDER + r] = s_idx[r];

    // ── 3. where each ranked list starts ─────────────────────────────────
    uint cchunk = (p.clusters + 1023u) / 1024u;
    uint clo = min(p.clusters, t * cchunk), chi = min(p.clusters, clo + cchunk);
    uint len = 0u;
    for (uint r = clo; r < chi; ++r) len += coff[s_idx[r] + 1u] - coff[s_idx[r]];
    barrier();
    s_scan[t] = len;
    barrier();
    scan(t);
    uint run = s_scan[t] - len;
    for (uint r = clo; r < chi; ++r) {
        plan[PLAN_ORDER + p.clusters + r] = run;
        run += coff[s_idx[r] + 1u] - coff[s_idx[r]];
    }
    if (t == 1023u) plan[PLAN_ORDER + 2u * p.clusters] = s_scan[1023];
    if (t == 0u) {
        plan[0] = T;
        plan[1] = all_ties ? 0xFFFFFFFFu : s_found[2];
        plan[2] = below;
    }
}
