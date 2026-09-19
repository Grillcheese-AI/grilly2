#version 450
#extension GL_GOOGLE_include_directive : require
// Next-token sampling, one workgroup per row of logits, with Hugging Face's
// processors in generate()'s order (transformers 5.5):
//   repetition penalty on the tokens so far:  x < 0 ? x * r : x / r
//   temperature:                               x / t
//   top_k: drop x below the k-th largest value (ties with it stay)
//   top_p: in ascending order, drop tokens whose cumulative probability is
//          <= 1 - p (the largest always stays)
// then draw from the softmax of what is left, by the inverse CDF in token
// order (as torch.multinomial), with one uniform u per row drawn on the host.
//
// No sort. top_k is a radix select of the k-th largest value: four passes
// of 8-bit digits over order-preserving integer keys. top_p is a binary
// search over keys for the smallest value whose cumulative mass passes
// (1 - p) * Z: at most 32 masked sums. Tokens tied at that value are kept
// in the number HF keeps, lowest indices dropped first: HF's order among
// ties is torch.sort's, which is not stable, so only the count can match.
// Divisions are correctly rounded
// (div_rn), as torch's are, so penalized and scaled scores are torch's bits.
// A thread's sums over its ~V/WG elements are compensated (Kahan): plain
// float32 sums over a 152K vocabulary put the softmax normalizer 1.3e-5
// off. Every sum runs in a fixed order: the draw is deterministic given u.
//
// Fast path: when the candidates left by top_k (or, with top_k off and
// top_p on, the CAP largest scores, used only when they hold at least p of
// the mass) fit in shared memory, they are copied there in token order and
// top_p, the ties, the draw and the distribution run on at most CAP
// entries. Otherwise every pass reads the whole row.
//
// Modes (flags): SCORES_ONLY stops after the penalty (greedy decoding takes
// the argmax of the scores); WEIGHTS draws from given non-negative weights
// with no processing (torch.multinomial); PROBS also writes the filtered
// distribution. Binding 4 always holds exp(x - max) during the draw.

// 1024 threads: sixteen waves per workgroup to hide memory latency (with
// 256, every pass over a 152K row was latency-bound at ~0.1 ms)
#define WG 1024u
layout(local_size_x = 1024) in;
#include "g2-common.glsl"

const uint F_PENALTY = 1u, F_TEMPERATURE = 2u, F_TOP_P = 4u, F_PROBS = 8u, F_WEIGHTS = 16u, F_SCORES_ONLY = 32u;

layout(std430, binding = 0) readonly buffer L { float logits[]; };
layout(std430, binding = 1) readonly buffer H { uint hist[]; };         // int64 token ids, (lo, hi) words
layout(std430, binding = 2) coherent buffer S { float scores[]; };     // (rows, v)
layout(std430, binding = 3) writeonly buffer T { uint tokens[]; };     // int64, (lo, hi) words
layout(std430, binding = 4) coherent buffer E { float ex[]; };         // (rows, v)
layout(push_constant) uniform Push {
    uint v; uint hist_len; uint flags; uint top_k;
    float temperature; float penalty; float one_minus_p; uint logits_off;
    uint hist_off; uint row0; uint rows;
    float u[8];
} p;

shared float red_f[1024];
shared uint red_u[1024];
shared uint bins[256];  // radix digits, not threads
shared uint pick[2];
shared float pick_f;
shared uint tie_at;  // index of the first tie kept at the top_p cut

// the fast path's candidates, in token order
const uint CAP = 2048u;
shared uint c_key[CAP];
shared uint c_idx[CAP];
shared float c_ex[CAP];

// order-preserving: a < b as floats  <=>  key(a) < key(b) as uints
uint fkey(float f) {
    uint b = floatBitsToUint(f);
    return (b >> 31) != 0u ? ~b : (b | 0x80000000u);
}
float kfloat(uint k) {
    return uintBitsToFloat((k >> 31) != 0u ? (k & 0x7FFFFFFFu) : ~k);
}

float sum_all(float v) {
    uint tid = gl_LocalInvocationID.x;
    red_f[tid] = v;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) red_f[tid] += red_f[tid + s];
        barrier();
    }
    float r = red_f[0];
    barrier();
    return r;
}

uint sum_all_u(uint v) {
    uint tid = gl_LocalInvocationID.x;
    red_u[tid] = v;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) red_u[tid] += red_u[tid + s];
        barrier();
    }
    uint r = red_u[0];
    barrier();
    return r;
}

// Kept after top_k and top_p: above the cut value, or tied with it at or
// past the first tie kept.
bool kept_key(uint k, uint i, uint kcut, uint tie_start) {
    return k > kcut || (k == kcut && i >= tie_start);
}

uint max_all(uint v) {
    uint tid = gl_LocalInvocationID.x;
    red_u[tid] = v;
    barrier();
    for (uint s = WG / 2u; s > 0u; s >>= 1) {
        if (tid < s) red_u[tid] = max(red_u[tid], red_u[tid + s]);
        barrier();
    }
    uint r = red_u[0];
    barrier();
    return r;
}

float scaled(float x) {
    return (p.flags & F_TEMPERATURE) != 0u ? div_rn(x, p.temperature) : x;
}

// The key of the need-th largest score of the row (1 <= need <= n): a radix
// select, four passes of 8-bit digits from the top.
uint radix_kth(uint sb, uint n, uint need) {
    uint tid = gl_LocalInvocationID.x;
    uint prefix = 0u, mask = 0u;
    for (int d = 3; d >= 0; --d) {
        uint shift = uint(8 * d);
        if (tid < 256u) bins[tid] = 0u;
        barrier();
        for (uint i = tid; i < n; i += WG) {
            uint k = fkey(scores[sb + i]);
            if ((k & mask) == prefix) atomicAdd(bins[(k >> shift) & 0xFFu], 1u);
        }
        barrier();
        if (tid == 0u) {
            uint cum = 0u;
            for (int b = 255; b >= 0; --b) {
                uint c = bins[b];
                if (cum + c >= need) {
                    pick[0] = uint(b);
                    pick[1] = need - cum;
                    break;
                }
                cum += c;
            }
        }
        barrier();
        prefix |= pick[0] << shift;
        mask |= 0xFFu << shift;
        need = pick[1];
        barrier();
    }
    return prefix;
}

void main() {
    uint row = p.row0 + gl_WorkGroupID.x;
    if (row >= p.rows) return;  // uniform per workgroup
    uint tid = gl_LocalInvocationID.x;
    uint n = p.v;
    uint lb = p.logits_off + row * n;
    uint sb = row * n;
    bool weights = (p.flags & F_WEIGHTS) != 0u;

    // ── scores: penalty, then temperature ────────────────────────────────
    uint kmin = 0u;  // lowest key kept
    if (!weights) {
        for (uint i = tid; i < n; i += WG) scores[sb + i] = scaled(logits[lb + i]);
        memoryBarrierBuffer();
        barrier();
        if ((p.flags & F_PENALTY) != 0u) {
            // repeats of a token write the same value, as gather + scatter
            for (uint h = tid; h < p.hist_len; h += WG) {
                uint at = 2u * (p.hist_off + row * p.hist_len + h);
                uint tok = hist[at];
                if (hist[at + 1u] == 0u && tok < n) {
                    float x = logits[lb + tok];
                    precise float mul = x * p.penalty;
                    scores[sb + tok] = scaled(x < 0.0 ? mul : div_rn(x, p.penalty));
                }
            }
            memoryBarrierBuffer();
            barrier();
        }
        if ((p.flags & F_SCORES_ONLY) != 0u) return;

        // ── top_k: the key of the k-th largest score ─────────────────────
        if (p.top_k > 0u && p.top_k < n) kmin = radix_kth(sb, n, p.top_k);
        // -inf scores have no probability: never kept
        kmin = max(kmin, fkey(uintBitsToFloat(0xff800000u)) + 1u);
    }

    // ── the largest score ────────────────────────────────────────────────
    uint kmax = 0u;
    if (!weights) {
        uint mk = 0u;
        for (uint i = tid; i < n; i += WG) mk = max(mk, fkey(scores[sb + i]));
        kmax = max_all(mk);
    }
    float m = kfloat(kmax);
    uint chunk = (n + WG - 1u) / WG;
    uint start = min(tid * chunk, n), end = min(start + chunk, n);

    // ── fast path: the candidates fit in shared memory ───────────────────
    // Every branch below depends only on push constants and shared sums, so
    // the whole workgroup takes it together.
    if (!weights && kmax >= kmin) {
        bool top_p = (p.flags & F_TOP_P) != 0u;
        uint kc = kmin;
        bool pool = false;
        if (top_p && !(p.top_k > 0u && p.top_k < n) && n > CAP) {
            kc = max(kmin, radix_kth(sb, n, CAP));
            pool = true;
        }
        uint cnt = 0u;
        for (uint i = start; i < end; ++i) {
            if (fkey(scores[sb + i]) >= kc) cnt++;
        }
        uint total = sum_all_u(cnt);
        if (total <= CAP) {
            // each thread copies its chunk's candidates, in token order
            red_u[tid] = cnt;
            barrier();
            if (tid == 0u) {
                uint acc = 0u;
                for (uint t = 0u; t < WG; ++t) {
                    uint c = red_u[t];
                    red_u[t] = acc;
                    acc += c;
                }
            }
            barrier();
            uint at = red_u[tid];
            barrier();
            for (uint i = start; i < end; ++i) {
                float s = scores[sb + i];
                uint k = fkey(s);
                if (k >= kc) {
                    c_key[at] = k;
                    c_idx[at] = i;
                    c_ex[at] = exp_acc(s - m);
                    at++;
                }
            }
            barrier();

            // A pool stands in for the row only if everything below it is
            // dropped by top_p: the mass below it is <= (1 - p) * Z.
            float base = 0.0, target = 0.0;
            bool ok = true;
            if (top_p) {
                float cm = 0.0, cmc = 0.0;
                for (uint j = tid; j < total; j += WG) kahan_add(cm, cmc, c_ex[j]);
                float z = sum_all(cm + cmc);
                if (pool) {
                    float zr = 0.0, zc = 0.0;
                    for (uint i = tid; i < n; i += WG) {
                        float s = scores[sb + i];
                        if (fkey(s) >= kmin) kahan_add(zr, zc, exp_acc(s - m));
                    }
                    float whole = sum_all(zr + zc);
                    base = whole - z;
                    z = whole;
                }
                target = p.one_minus_p * z;
                ok = base <= target;
            }
            if (ok) {
                uint kcut = kc, tie_start = 0u;
                if (top_p) {
                    uint lo = kc, hi = kmax;
                    while (lo < hi) {
                        uint mid = lo + (hi - lo) / 2u;
                        float part = 0.0, pc = 0.0;
                        for (uint j = tid; j < total; j += WG) {
                            if (c_key[j] <= mid) kahan_add(part, pc, c_ex[j]);
                        }
                        if (base + sum_all(part + pc) > target) hi = mid; else lo = mid + 1u;
                    }
                    kcut = lo;
                    // ties at the cut, as in the full path below
                    float bm = 0.0, bmc = 0.0;
                    uint ties = 0u;
                    for (uint j = tid; j < total; j += WG) {
                        if (c_key[j] < kcut) kahan_add(bm, bmc, c_ex[j]);
                        if (c_key[j] == kcut) ties++;
                    }
                    float below = base + sum_all(bm + bmc);
                    uint g = sum_all_u(ties);
                    if (g > 1u) {
                        float q = exp_acc(kfloat(kcut) - m);
                        uint drop = uint(clamp(floor(div_rn(target - below, q)), 0.0, float(g - 1u)));
                        if (tid == 0u) {
                            uint seen = 0u;
                            tie_at = 0u;
                            for (uint j = 0u; j < total; ++j) {
                                if (c_key[j] == kcut) {
                                    if (seen == drop) {
                                        tie_at = c_idx[j];
                                        break;
                                    }
                                    seen++;
                                }
                            }
                        }
                        barrier();
                        tie_start = tie_at;
                        barrier();
                    }
                }
                // the draw: inverse CDF in token order over the candidates
                if (tid == 0u) {
                    float tot = 0.0, tc = 0.0;
                    for (uint j = 0u; j < total; ++j) {
                        if (kept_key(c_key[j], c_idx[j], kcut, tie_start)) kahan_add(tot, tc, c_ex[j]);
                    }
                    tot += tc;
                    float want = p.u[row - p.row0] * tot;
                    float acc = 0.0, ac = 0.0;
                    uint chosen = 0xFFFFFFFFu, lastk = 0xFFFFFFFFu;
                    for (uint j = 0u; j < total; ++j) {
                        if (kept_key(c_key[j], c_idx[j], kcut, tie_start)) {
                            lastk = c_idx[j];
                            kahan_add(acc, ac, c_ex[j]);
                            if (acc + ac > want) {
                                chosen = c_idx[j];
                                break;
                            }
                        }
                    }
                    if (chosen == 0xFFFFFFFFu) chosen = lastk;  // rounding put the target at the very end
                    tokens[2u * row] = chosen;
                    tokens[2u * row + 1u] = chosen == 0xFFFFFFFFu ? 0xFFFFFFFFu : 0u;
                    pick_f = tot;
                }
                if ((p.flags & F_PROBS) != 0u) {
                    barrier();
                    float tot = pick_f;
                    for (uint i = tid; i < n; i += WG) ex[sb + i] = 0.0;
                    memoryBarrierBuffer();
                    barrier();
                    for (uint j = tid; j < total; j += WG) {
                        if (kept_key(c_key[j], c_idx[j], kcut, tie_start)) ex[sb + c_idx[j]] = div_rn(c_ex[j], tot);
                    }
                }
                return;
            }
        }
    }

    // ── full path: exp(x - max) over what top_k kept, the whole row ──────
    if (!weights) {
        for (uint i = tid; i < n; i += WG) {
            float s = scores[sb + i];
            ex[sb + i] = fkey(s) >= kmin ? exp_acc(s - m) : 0.0;
        }
    } else {
        for (uint i = tid; i < n; i += WG) ex[sb + i] = logits[lb + i];
    }
    memoryBarrierBuffer();
    barrier();

    // ── top_p: the smallest key whose cumulative mass passes (1 - p) * Z ─
    uint kcut = kmin, tie_start = 0u;
    if (!weights && (p.flags & F_TOP_P) != 0u && kmax >= kmin) {
        float z = 0.0, zc = 0.0;
        for (uint i = tid; i < n; i += WG) kahan_add(z, zc, ex[sb + i]);
        float target = p.one_minus_p * sum_all(z + zc);
        uint lo = kmin, hi = kmax;
        while (lo < hi) {  // uniform: every thread sees the same sums
            uint mid = lo + (hi - lo) / 2u;
            float part = 0.0, pc = 0.0;
            for (uint i = tid; i < n; i += WG) {
                if (fkey(scores[sb + i]) <= mid) kahan_add(part, pc, ex[sb + i]);
            }
            if (sum_all(part + pc) > target) hi = mid; else lo = mid + 1u;
        }
        kcut = lo;

        // Ties at the cut: HF, in its ascending order, drops each while the
        // cumulative mass is still <= target. With `below` the mass under
        // the cut and q each tie's mass, that drops floor((target - below) / q)
        // of them, never all.
        float bm = 0.0, bmc = 0.0;
        uint ties = 0u;
        for (uint i = start; i < end; ++i) {
            uint k = fkey(scores[sb + i]);
            if (k >= kmin && k < kcut) kahan_add(bm, bmc, ex[sb + i]);
            if (k == kcut) ties++;
        }
        float below = sum_all(bm + bmc);
        uint g = sum_all_u(ties);
        if (g > 1u) {
            float q = exp_acc(kfloat(kcut) - kfloat(kmax));
            uint drop = uint(clamp(floor(div_rn(target - below, q)), 0.0, float(g - 1u)));
            if (drop > 0u) {
                // the index of tie number `drop` in index order
                red_u[tid] = ties;
                barrier();
                if (tid == 0u) {
                    uint seen = 0u;
                    for (uint t = 0u; t < WG; ++t) {
                        if (seen + red_u[t] > drop) {
                            pick[0] = t;
                            pick[1] = drop - seen;
                            break;
                        }
                        seen += red_u[t];
                    }
                }
                barrier();
                if (tid == pick[0]) {
                    uint j = 0u;
                    for (uint i = start; i < end; ++i) {
                        if (fkey(scores[sb + i]) == kcut) {
                            if (j == pick[1]) {
                                tie_at = i;
                                break;
                            }
                            j++;
                        }
                    }
                }
                barrier();
                tie_start = tie_at;
                barrier();
            }
        }
    }

    // ── draw: inverse CDF in token order, a contiguous chunk per thread ──
    float cs = 0.0, cc = 0.0;
    for (uint i = start; i < end; ++i) {
        if (weights || kept_key(fkey(scores[sb + i]), i, kcut, tie_start)) kahan_add(cs, cc, ex[sb + i]);
    }
    red_f[tid] = cs + cc;
    barrier();
    if (tid == 0u) {
        float total = 0.0, tc = 0.0;
        for (uint t = 0u; t < WG; ++t) kahan_add(total, tc, red_f[t]);
        total += tc;
        float target = p.u[row - p.row0] * total;
        // the chunk holding the target, with the mass before it (compensated)
        float before = 0.0, bc = 0.0;
        uint owner = 0xFFFFFFFFu, last = 0xFFFFFFFFu;
        for (uint t = 0u; t < WG; ++t) {
            if (red_f[t] > 0.0) {
                last = t;
                if ((before + bc) + red_f[t] > target) {
                    owner = t;
                    break;
                }
                kahan_add(before, bc, red_f[t]);
            }
        }
        // Rounding put the target at the very end. `last` is the sentinel too
        // when no chunk has any mass (all -inf, or zero weights): that row
        // draws nothing, below. Without this check the loop ran to 2^32 - 1
        // on one thread and held the GPU for minutes.
        if (owner == 0xFFFFFFFFu && last != 0xFFFFFFFFu) {
            owner = last;
            before = 0.0;
            bc = 0.0;
            for (uint t = 0u; t < last; ++t) kahan_add(before, bc, red_f[t]);
            target = (before + bc) + red_f[last];
        }
        pick[0] = owner;
        pick_f = target - (before + bc);
        red_f[0] = total;  // kept mass, for PROBS
    }
    barrier();
    uint owner = pick[0];
    if (owner == 0xFFFFFFFFu) {  // nothing to draw from (all -inf, or zero weights)
        if (tid == 0u) {
            tokens[2u * row] = 0xFFFFFFFFu;
            tokens[2u * row + 1u] = 0xFFFFFFFFu;
        }
        return;
    }
    if (tid == owner) {
        float acc = 0.0, ac = 0.0;
        uint chosen = start, lastkept = start;
        bool found = false;
        for (uint i = start; i < end; ++i) {
            if (weights || kept_key(fkey(scores[sb + i]), i, kcut, tie_start)) {
                float e = ex[sb + i];
                if (e > 0.0) lastkept = i;
                kahan_add(acc, ac, e);
                if (acc + ac > pick_f) {
                    chosen = i;
                    found = true;
                    break;
                }
            }
        }
        if (!found) chosen = lastkept;
        tokens[2u * row] = chosen;
        tokens[2u * row + 1u] = 0u;
    }

    // ── the filtered distribution, when asked ────────────────────────────
    if ((p.flags & F_PROBS) != 0u) {
        float total = red_f[0];
        barrier();
        for (uint i = tid; i < n; i += WG) {
            bool kept = weights || kept_key(fkey(scores[sb + i]), i, kcut, tie_start);
            ex[sb + i] = kept ? div_rn(ex[sb + i], total) : 0.0;
        }
    }
}
