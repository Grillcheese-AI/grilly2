#version 450
#extension GL_GOOGLE_include_directive : require
// Block-sparse attention backward, part 1 of 2: dK and dV. One workgroup
// per (batch, head, 32-key tile); it walks the query tiles that touch it
// (the transposed tile list, ascending), 16 query rows at a time so its
// shared tiles fit in 64 KiB, and recomputes from the forward's lse:
//   P  = exp(q.k * scale - lse)          (0 where masked)
//   dS = P * (dO.v - Drow),  Drow = rowsum(dO * O)
//   dV += P^T dO,  dK += scale * dS^T Q
// Every sum runs in a fixed order and nothing is atomic: deterministic.
// dQ is g2-flash-bwd-q.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

#define T 32
#define U 16
#define MAX_D 128
#define FULL 0x40000000u

layout(std430, binding = 0) readonly buffer Qb { float Qd[]; };
layout(std430, binding = 1) readonly buffer Kb { float Kd[]; };
layout(std430, binding = 2) readonly buffer Vb { float Vd[]; };
layout(std430, binding = 3) readonly buffer dOb { float dOd[]; };
layout(std430, binding = 4) readonly buffer LseB { float lse[]; };       // (B, H, L)
layout(std430, binding = 5) readonly buffer DrowB { float drow[]; };     // (B, H, L)
layout(std430, binding = 6) readonly buffer Bits { uint bits[]; };       // (mb, mh, L, nk)
layout(std430, binding = 7) readonly buffer QCount { uint q_count[]; };  // (mb, mh, nk)
layout(std430, binding = 8) readonly buffer QList { uint q_list[]; };    // (mb, mh, nk, nq)
layout(std430, binding = 9) writeonly buffer dKb { float dKd[]; };
layout(std430, binding = 10) writeonly buffer dVb { float dVd[]; };
layout(push_constant) uniform P {
    uint B; uint H; uint L; uint S; uint D; uint Dv; float scale;
    uint mb; uint mh; uint nq; uint nk; uint q_off; uint k_off; uint v_off; uint do_off;
} p;

shared float Ks[T][MAX_D + 1];
shared float Vs[T][MAX_D + 1];
shared float Qs[U][MAX_D + 1];
shared float dOs[U][MAX_D + 1];
shared float Ps[U][T + 1];
shared float dSs[U][T + 1];
shared float lse_s[U];
shared float drow_s[U];

void main() {
    uint kt = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint tid = gl_LocalInvocationID.x;
    uint mask_bh = (p.mb == 1u ? 0u : b) * p.mh + (p.mh == 1u ? 0u : h);
    uint bh = b * p.H + h;
    uint k0 = kt * T;
    uint q_base = p.q_off + bh * p.L * p.D;
    uint k_base = p.k_off + bh * p.S * p.D;
    uint v_base = p.v_off + bh * p.S * p.Dv;
    uint do_base = p.do_off + bh * p.L * p.Dv;

    for (uint i = tid; i < T * p.D; i += 256u) {
        uint r = i / p.D, d = i % p.D;
        Ks[r][d] = k0 + r < p.S ? Kd[k_base + (k0 + r) * p.D + d] : 0.0;
    }
    for (uint i = tid; i < T * p.Dv; i += 256u) {
        uint r = i / p.Dv, d = i % p.Dv;
        Vs[r][d] = k0 + r < p.S ? Vd[v_base + (k0 + r) * p.Dv + d] : 0.0;
    }
    float dk[16], dv[16];  // this thread's share of the (T, D) and (T, Dv) tiles
    for (uint j = 0u; j < 16u; ++j) {
        dk[j] = 0.0;
        dv[j] = 0.0;
    }

    uint list = mask_bh * p.nk + kt;
    uint count = q_count[list];
    for (uint e = 0u; e < count; ++e) {
        uint entry = q_list[list * p.nq + e];
        uint qt = entry & (FULL - 1u);
        bool full = (entry & FULL) != 0u;
        for (uint half_ = 0u; half_ < T / U; ++half_) {
            uint q0 = qt * T + half_ * U;
            if (q0 >= p.L) break;
            barrier();  // the previous rows are consumed
            for (uint i = tid; i < U * p.D; i += 256u) {
                uint r = i / p.D, d = i % p.D;
                Qs[r][d] = q0 + r < p.L ? Qd[q_base + (q0 + r) * p.D + d] : 0.0;
            }
            for (uint i = tid; i < U * p.Dv; i += 256u) {
                uint r = i / p.Dv, d = i % p.Dv;
                dOs[r][d] = q0 + r < p.L ? dOd[do_base + (q0 + r) * p.Dv + d] : 0.0;
            }
            if (tid < U) {
                lse_s[tid] = q0 + tid < p.L ? lse[bh * p.L + q0 + tid] : 0.0;
                drow_s[tid] = q0 + tid < p.L ? drow[bh * p.L + q0 + tid] : 0.0;
            }
            barrier();
            for (uint i = tid; i < U * T; i += 256u) {
                uint r = i / T, c = i % T;
                bool keep = q0 + r < p.L && k0 + c < p.S;
                if (keep && !full) keep = ((bits[(mask_bh * p.L + q0 + r) * p.nk + kt] >> c) & 1u) != 0u;
                float pv = 0.0, ds = 0.0;
                if (keep) {
                    float dot = 0.0;
                    for (uint d = 0u; d < p.D; ++d) dot = fma(Qs[r][d], Ks[c][d], dot);
                    pv = exp_acc(dot * p.scale - lse_s[r]);
                    float dp = 0.0;
                    for (uint d = 0u; d < p.Dv; ++d) dp = fma(dOs[r][d], Vs[c][d], dp);
                    ds = pv * (dp - drow_s[r]);
                }
                Ps[r][c] = pv;
                dSs[r][c] = ds;
            }
            barrier();
            for (uint j = 0u; j < 16u; ++j) {
                uint i = tid + j * 256u;
                if (i < T * p.Dv) {
                    uint c = i / p.Dv, d = i % p.Dv;
                    float a = dv[j];
                    for (uint r = 0u; r < U; ++r) a = fma(Ps[r][c], dOs[r][d], a);
                    dv[j] = a;
                }
                if (i < T * p.D) {
                    uint c = i / p.D, d = i % p.D;
                    float a = dk[j];
                    for (uint r = 0u; r < U; ++r) a = fma(dSs[r][c], Qs[r][d], a);
                    dk[j] = a;
                }
            }
        }
    }

    for (uint j = 0u; j < 16u; ++j) {
        uint i = tid + j * 256u;
        if (i < T * p.Dv) {
            uint c = i / p.Dv, d = i % p.Dv;
            if (k0 + c < p.S) dVd[(bh * p.S + k0 + c) * p.Dv + d] = dv[j];
        }
        if (i < T * p.D) {
            uint c = i / p.D, d = i % p.D;
            if (k0 + c < p.S) dKd[(bh * p.S + k0 + c) * p.D + d] = dk[j] * p.scale;
        }
    }
}
