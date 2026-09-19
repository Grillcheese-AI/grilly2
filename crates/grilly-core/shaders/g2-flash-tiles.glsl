#version 450
#extension GL_GOOGLE_include_directive : require
// Tile lists for the fused attention kernels, built on the GPU from a bool
// mask (mb, mh, L, S) (one u32 per element), in four passes (`stage`):
//   0  bits:     one thread per (mb, mh, q, key tile): 32 mask entries -> a word
//   1  flags:    one thread per (mb, mh, query tile, key tile): 0 untouched,
//                1 partial, 2 full (every in-range position kept)
//   2  kv lists: one thread per (mb, mh, query tile): its touched key tiles,
//                ascending, FULL-flagged, and their count
//   3  q lists:  one thread per (mb, mh, key tile): the transpose
// The kernels (g2-flash-fwd, -bwd-kv, -bwd-q) read these; no host sync.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

#define T 32
#define FULL 0x40000000u

layout(std430, binding = 0) readonly buffer Mask { uint mask[]; };
layout(std430, binding = 1) buffer Bits { uint bits[]; };          // (mb, mh, L, nk)
layout(std430, binding = 2) buffer Flags { uint flags[]; };        // (mb, mh, nq, nk)
layout(std430, binding = 3) buffer KvCount { uint kv_count[]; };   // (mb, mh, nq)
layout(std430, binding = 4) buffer KvList { uint kv_list[]; };     // (mb, mh, nq, nk)
layout(std430, binding = 5) buffer QCount { uint q_count[]; };     // (mb, mh, nk)
layout(std430, binding = 6) buffer QList { uint q_list[]; };       // (mb, mh, nk, nq)
layout(push_constant) uniform P { uint stage; uint planes; uint L; uint S; uint nq; uint nk; uint mask_off; } p;

void main() {
    uint i = flat_index();
    if (p.stage == 0u) {
        if (i >= p.planes * p.L * p.nk) return;
        uint kt = i % p.nk, row = i / p.nk;  // row = plane * L + q
        uint word = 0u;
        for (uint c = 0u; c < T; ++c) {
            uint key = kt * T + c;
            if (key < p.S && mask[p.mask_off + row * p.S + key] != 0u) word |= 1u << c;
        }
        bits[i] = word;
    } else if (p.stage == 1u) {
        if (i >= p.planes * p.nq * p.nk) return;
        uint kt = i % p.nk, rest = i / p.nk, qt = rest % p.nq, plane = rest / p.nq;
        uint cols = min(T, p.S - kt * T);
        uint valid = cols == 32u ? 0xffffffffu : (1u << cols) - 1u;
        bool any_ = false, all_ = true;
        for (uint r = 0u; r < T && qt * T + r < p.L; ++r) {
            uint word = bits[(plane * p.L + qt * T + r) * p.nk + kt];
            any_ = any_ || word != 0u;
            all_ = all_ && word == valid;
        }
        flags[i] = any_ ? (all_ ? 2u : 1u) : 0u;
    } else if (p.stage == 2u) {
        if (i >= p.planes * p.nq) return;  // i = plane * nq + qt
        uint n = 0u;
        for (uint kt = 0u; kt < p.nk; ++kt) {
            uint f = flags[i * p.nk + kt];
            if (f != 0u) kv_list[i * p.nk + n++] = kt | (f == 2u ? FULL : 0u);
        }
        for (uint e = n; e < p.nk; ++e) kv_list[i * p.nk + e] = 0u;
        kv_count[i] = n;
    } else {
        if (i >= p.planes * p.nk) return;  // i = plane * nk + kt
        uint plane = i / p.nk, kt = i % p.nk;
        uint n = 0u;
        for (uint qt = 0u; qt < p.nq; ++qt) {
            uint f = flags[(plane * p.nq + qt) * p.nk + kt];
            if (f != 0u) q_list[i * p.nq + n++] = qt | (f == 2u ? FULL : 0u);
        }
        for (uint e = n; e < p.nq; ++e) q_list[i * p.nq + e] = 0u;
        q_count[i] = n;
    }
}
