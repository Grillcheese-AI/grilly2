#version 450
#extension GL_GOOGLE_include_directive : require
// Block-sparse attention backward, part 2 of 2: dQ. One workgroup per
// (batch, head, 32-query tile); it walks its key tiles (ascending), 16 keys
// at a time so its shared tiles fit in 64 KiB, and recomputes
//   P  = exp(q.k * scale - lse)          (0 where masked)
//   dS = P * (dO.v - Drow),  Drow = rowsum(dO * O)
//   dQ += scale * dS K
// in a fixed order, without atomics: deterministic. dK and dV are
// g2-flash-bwd-kv.

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
layout(std430, binding = 4) readonly buffer LseB { float lse[]; };
layout(std430, binding = 5) readonly buffer DrowB { float drow[]; };
layout(std430, binding = 6) readonly buffer Bits { uint bits[]; };
layout(std430, binding = 7) readonly buffer KvCount { uint kv_count[]; };
layout(std430, binding = 8) readonly buffer KvList { uint kv_list[]; };
layout(std430, binding = 9) writeonly buffer dQb { float dQd[]; };
layout(push_constant) uniform P {
    uint B; uint H; uint L; uint S; uint D; uint Dv; float scale;
    uint mb; uint mh; uint nq; uint nk; uint q_off; uint k_off; uint v_off; uint do_off;
} p;

shared float Qs[T][MAX_D + 1];
shared float dOs[T][MAX_D + 1];
shared float Ks[U][MAX_D + 1];
shared float Vs[U][MAX_D + 1];
shared float dSs[T][U + 1];
shared float lse_s[T];
shared float drow_s[T];

void main() {
    uint qt = gl_WorkGroupID.x, h = gl_WorkGroupID.y, b = gl_WorkGroupID.z;
    uint tid = gl_LocalInvocationID.x;
    uint mask_bh = (p.mb == 1u ? 0u : b) * p.mh + (p.mh == 1u ? 0u : h);
    uint bh = b * p.H + h;
    uint q0 = qt * T;
    uint q_base = p.q_off + bh * p.L * p.D;
    uint k_base = p.k_off + bh * p.S * p.D;
    uint v_base = p.v_off + bh * p.S * p.Dv;
    uint do_base = p.do_off + bh * p.L * p.Dv;

    for (uint i = tid; i < T * p.D; i += 256u) {
        uint r = i / p.D, d = i % p.D;
        Qs[r][d] = q0 + r < p.L ? Qd[q_base + (q0 + r) * p.D + d] : 0.0;
    }
    for (uint i = tid; i < T * p.Dv; i += 256u) {
        uint r = i / p.Dv, d = i % p.Dv;
        dOs[r][d] = q0 + r < p.L ? dOd[do_base + (q0 + r) * p.Dv + d] : 0.0;
    }
    if (tid < T) {
        lse_s[tid] = q0 + tid < p.L ? lse[bh * p.L + q0 + tid] : 0.0;
        drow_s[tid] = q0 + tid < p.L ? drow[bh * p.L + q0 + tid] : 0.0;
    }
    float dq[16];  // this thread's share of the (T, D) tile
    for (uint j = 0u; j < 16u; ++j) dq[j] = 0.0;

    uint list = mask_bh * p.nq + qt;
    uint count = kv_count[list];
    for (uint e = 0u; e < count; ++e) {
        uint entry = kv_list[list * p.nk + e];
        uint kt = entry & (FULL - 1u);
        bool full = (entry & FULL) != 0u;
        for (uint half_ = 0u; half_ < T / U; ++half_) {
            uint k0 = kt * T + half_ * U;
            if (k0 >= p.S) break;
            barrier();  // the previous keys are consumed
            for (uint i = tid; i < U * p.D; i += 256u) {
                uint r = i / p.D, d = i % p.D;
                Ks[r][d] = k0 + r < p.S ? Kd[k_base + (k0 + r) * p.D + d] : 0.0;
            }
            for (uint i = tid; i < U * p.Dv; i += 256u) {
                uint r = i / p.Dv, d = i % p.Dv;
                Vs[r][d] = k0 + r < p.S ? Vd[v_base + (k0 + r) * p.Dv + d] : 0.0;
            }
            barrier();
            for (uint i = tid; i < T * U; i += 256u) {
                uint r = i / U, c = i % U;
                bool keep = q0 + r < p.L && k0 + c < p.S;
                if (keep && !full) {
                    keep = ((bits[(mask_bh * p.L + q0 + r) * p.nk + kt] >> (half_ * U + c)) & 1u) != 0u;
                }
                float ds = 0.0;
                if (keep) {
                    float dot = 0.0;
                    for (uint d = 0u; d < p.D; ++d) dot = fma(Qs[r][d], Ks[c][d], dot);
                    float pv = exp_acc(dot * p.scale - lse_s[r]);
                    float dp = 0.0;
                    for (uint d = 0u; d < p.Dv; ++d) dp = fma(dOs[r][d], Vs[c][d], dp);
                    ds = pv * (dp - drow_s[r]);
                }
                dSs[r][c] = ds;
            }
            barrier();
            for (uint j = 0u; j < 16u; ++j) {
                uint i = tid + j * 256u;
                if (i >= T * p.D) break;
                uint r = i / p.D, d = i % p.D;
                float a = dq[j];
                for (uint c = 0u; c < U; ++c) a = fma(dSs[r][c], Ks[c][d], a);
                dq[j] = a;
            }
        }
    }

    for (uint j = 0u; j < 16u; ++j) {
        uint i = tid + j * 256u;
        if (i >= T * p.D) break;
        uint r = i / p.D, d = i % p.D;
        if (q0 + r < p.L) dQd[(bh * p.L + q0 + r) * p.D + d] = dq[j] * p.scale;
    }
}
