#version 450
#extension GL_GOOGLE_include_directive : require
// y = (x @ q^T) * scale + bias with an int8 weight-only weight, for many
// rows of x (prefill). Same arithmetic as g2-linear-q8, different data flow.
//
// g2-linear-q8 is shaped for decode: a workgroup takes 8 rows of x and reads
// the whole weight row for them. At 128 rows that reads the weight 16 times
// and runs at about 1% of this card's bandwidth (docs/inference.md). Here
// each workgroup owns a BM x BN tile of the output and walks k in steps of
// BK, staging x and the unpacked weight in shared memory, so every value
// loaded is used BN (for x) or BM (for the weight) times.
//
// 256 threads, each holding a 4x4 micro-tile of the output in registers.
// Every accumulator sums its k terms in order, so the result is
// deterministic; it is not g2-linear-q8's order, which sums per lane and
// then over 64 lanes, so the two agree to float32 reassociation, not
// bit-exactly (tests/quant.rs measures both against float64).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

#define BM 64u   // rows of x per workgroup
#define BN 64u   // output features per workgroup
#define BK 32u   // k per step: one 16-byte weight vector is 16 values, so 2 per row
#define TM 4u    // rows per thread
#define TN 4u    // columns per thread

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer Q { uvec4 q[]; };
layout(std430, binding = 2) readonly buffer S { float scale[]; };
layout(std430, binding = 3) readonly buffer B { float bias[]; };
layout(std430, binding = 4) writeonly buffer Y { float y[]; };
layout(push_constant) uniform P {
    uint m; uint k; uint n; uint vecs;
    uint x_off; uint b_off; uint has_bias; uint pad;
} p;

shared float xs[BK][BM + 1u];  // +1: stagger the rows so column reads do not collide
shared float ws[BK][BN + 1u];

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint row0 = gl_WorkGroupID.y * BM;
    uint col0 = gl_WorkGroupID.x * BN;
    // 16x16 threads, each owning a 4x4 micro-tile
    uint tr = (tid / 16u) * TM, tc = (tid % 16u) * TN;

    float acc[TM][TN];
    for (uint i = 0u; i < TM; ++i) {
        for (uint j = 0u; j < TN; ++j) acc[i][j] = 0.0;
    }

    for (uint k0 = 0u; k0 < p.k; k0 += BK) {
        // stage x: 64 rows x 32 k, one float per thread twice
        for (uint t = tid; t < BM * BK; t += 256u) {
            uint r = t / BK, c = t % BK;
            uint gr = row0 + r, gk = k0 + c;
            xs[c][r] = (gr < p.m && gk < p.k) ? x[p.x_off + gr * p.k + gk] : 0.0;
        }
        // stage the weight: 64 features x 32 k, unpacked from two 16-byte
        // vectors per feature; 128 vectors over 256 threads
        for (uint t = tid; t < BN * (BK / 16u); t += 256u) {
            uint f = t / (BK / 16u), v = t % (BK / 16u);
            uint gf = col0 + f, gv = k0 / 16u + v;
            uvec4 packed = (gf < p.n && gv < p.vecs) ? q[gf * p.vecs + gv] : uvec4(0u);
            for (uint j = 0u; j < 4u; ++j) {
                int word = int(packed[j]);
                uint base = v * 16u + 4u * j;
                ws[base][f] = float(bitfieldExtract(word, 0, 8));
                ws[base + 1u][f] = float(bitfieldExtract(word, 8, 8));
                ws[base + 2u][f] = float(bitfieldExtract(word, 16, 8));
                ws[base + 3u][f] = float(bitfieldExtract(word, 24, 8));
            }
        }
        barrier();

        uint steps = min(BK, p.k - k0);  // a row's padding is never multiplied
        for (uint c = 0u; c < steps; ++c) {
            float xv[TM], wv[TN];
            for (uint i = 0u; i < TM; ++i) xv[i] = xs[c][tr + i];
            for (uint j = 0u; j < TN; ++j) wv[j] = ws[c][tc + j];
            for (uint i = 0u; i < TM; ++i) {
                for (uint j = 0u; j < TN; ++j) acc[i][j] = fma(xv[i], wv[j], acc[i][j]);
            }
        }
        barrier();
    }

    for (uint i = 0u; i < TM; ++i) {
        uint gr = row0 + tr + i;
        if (gr >= p.m) continue;
        for (uint j = 0u; j < TN; ++j) {
            uint gc = col0 + tc + j;
            if (gc >= p.n) continue;
            float v = acc[i][j] * scale[gc];
            if (p.has_bias != 0u) v += bias[p.b_off + gc];
            y[gr * p.n + gc] = v;
        }
    }
}
