#version 450
#extension GL_GOOGLE_include_directive : require
// Block-sparse attention forward (flex_attention's fused path),
// FlashAttention-2 style: one workgroup per (batch, head, 32-query tile).
// It walks only the key tiles in its tile list (ascending), scores them
// from shared-memory Q/K tiles, masks partial tiles with the bit mask,
// and folds them into an online softmax (running max m, sum l, with
// Cephes exp) and an output accumulator. No score matrix reaches memory.
// Writes O and the log-sum-exp (natural log) per query row; a row with no
// kept key gets O = 0 and lse = -inf, as torch's flex_attention does.
// Layout: Q (B, H, L, D), K (B, H, S, D), V (B, H, S, Dv), contiguous.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

#define T 32
#define MAX_D 128
#define FULL 0x40000000u

layout(std430, binding = 0) readonly buffer Qb { float Qd[]; };
layout(std430, binding = 1) readonly buffer Kb { float Kd[]; };
layout(std430, binding = 2) readonly buffer Vb { float Vd[]; };
layout(std430, binding = 3) readonly buffer Bits { uint bits[]; };        // (mb, mh, L, nk)
layout(std430, binding = 4) readonly buffer KvCount { uint kv_count[]; }; // (mb, mh, nq)
layout(std430, binding = 5) readonly buffer KvList { uint kv_list[]; };   // (mb, mh, nq, nk)
layout(std430, binding = 6) writeonly buffer Ob { float Od[]; };
layout(std430, binding = 7) writeonly buffer Lse { float lse[]; };
layout(push_constant) uniform P {
    uint B; uint H; uint L; uint S; uint D; uint Dv; float scale;
    uint mb; uint mh; uint nq; uint nk; uint q_off; uint k_off; uint v_off;
} p;

shared float Qs[T][MAX_D + 1];  // +1: rows start in different banks
shared float Ks[T][MAX_D + 1];
shared float Vs[T][MAX_D + 1];
shared float Ps[T][T + 1];
shared float m_s[T];
shared float l_s[T];
shared float alpha_s[T];

void main() {
    uint qt = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint tid = gl_LocalInvocationID.x;
    uint mask_bh = (p.mb == 1u ? 0u : b) * p.mh + (p.mh == 1u ? 0u : h);
    uint bh = b * p.H + h;
    uint q0 = qt * T;
    uint q_base = p.q_off + bh * p.L * p.D;
    uint k_base = p.k_off + bh * p.S * p.D;
    uint v_base = p.v_off + bh * p.S * p.Dv;
    const float NEG_INF = uintBitsToFloat(0xff800000u);

    for (uint i = tid; i < T * p.D; i += 256u) {
        uint r = i / p.D, d = i % p.D;
        Qs[r][d] = q0 + r < p.L ? Qd[q_base + (q0 + r) * p.D + d] : 0.0;
    }
    if (tid < T) {
        m_s[tid] = NEG_INF;
        l_s[tid] = 0.0;
    }
    float acc[16];  // this thread's share of the (T, Dv) output tile
    for (uint j = 0u; j < 16u; ++j) acc[j] = 0.0;
    uint per = (T * p.Dv + 255u) / 256u;
    barrier();

    uint list = mask_bh * p.nq + qt;
    uint count = kv_count[list];
    for (uint e = 0u; e < count; ++e) {
        uint entry = kv_list[list * p.nk + e];
        uint kt = entry & (FULL - 1u);
        bool full = (entry & FULL) != 0u;
        uint k0 = kt * T;
        for (uint i = tid; i < T * p.D; i += 256u) {
            uint r = i / p.D, d = i % p.D;
            Ks[r][d] = k0 + r < p.S ? Kd[k_base + (k0 + r) * p.D + d] : 0.0;
        }
        for (uint i = tid; i < T * p.Dv; i += 256u) {
            uint r = i / p.Dv, d = i % p.Dv;
            Vs[r][d] = k0 + r < p.S ? Vd[v_base + (k0 + r) * p.Dv + d] : 0.0;
        }
        barrier();
        // scores of the tile, -inf where masked
        for (uint i = tid; i < T * T; i += 256u) {
            uint r = i / T, c = i % T;
            bool keep = k0 + c < p.S && q0 + r < p.L;
            if (keep && !full) keep = ((bits[(mask_bh * p.L + q0 + r) * p.nk + kt] >> c) & 1u) != 0u;
            float score = NEG_INF;
            if (keep) {
                float dot = 0.0;
                for (uint d = 0u; d < p.D; ++d) dot = fma(Qs[r][d], Ks[c][d], dot);
                score = dot * p.scale;
            }
            Ps[r][c] = score;
        }
        barrier();
        // online softmax, one thread per row, columns in order
        if (tid < T) {
            float m_old = m_s[tid], mx = m_old;
            for (uint c = 0u; c < T; ++c) mx = max(mx, Ps[tid][c]);
            float alpha = 1.0, sum = 0.0;
            if (mx == NEG_INF) {
                for (uint c = 0u; c < T; ++c) Ps[tid][c] = 0.0;
            } else {
                alpha = exp_acc(m_old - mx);  // 0 on the first kept tile (m_old = -inf)
                for (uint c = 0u; c < T; ++c) {
                    float pv = exp_acc(Ps[tid][c] - mx);
                    Ps[tid][c] = pv;
                    sum += pv;
                }
            }
            l_s[tid] = l_s[tid] * alpha + sum;
            m_s[tid] = mx;
            alpha_s[tid] = alpha;
        }
        barrier();
        for (uint j = 0u; j < 16u; ++j) {
            uint i = tid + j * 256u;
            if (j >= per || i >= T * p.Dv) break;
            uint r = i / p.Dv, d = i % p.Dv;
            float a = acc[j] * alpha_s[r];
            for (uint c = 0u; c < T; ++c) a = fma(Ps[r][c], Vs[c][d], a);
            acc[j] = a;
        }
        barrier();  // before the next tile overwrites Ks, Vs, Ps
    }

    for (uint j = 0u; j < 16u; ++j) {
        uint i = tid + j * 256u;
        if (j >= per || i >= T * p.Dv) break;
        uint r = i / p.Dv, d = i % p.Dv;
        if (q0 + r < p.L) {
            float l = l_s[r];
            Od[(bh * p.L + q0 + r) * p.Dv + d] = l > 0.0 ? acc[j] / l : 0.0;
        }
    }
    if (tid < T && q0 + tid < p.L) {
        float l = l_s[tid];
        lse[bh * p.L + q0 + tid] = l > 0.0 ? m_s[tid] + log(l) : NEG_INF;
    }
}
