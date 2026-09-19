#version 450
#extension GL_GOOGLE_include_directive : require
// Batched matrix product C[b] = op(A[b]) @ op(B[b]), float32, where op is
// an optional transpose (A stored (K, M), B stored (N, K)), so gradients
// need no transpose copies. Batch dims broadcast through stride-0 batch
// strides. One 16x16 output tile per workgroup, K staged through shared
// memory 16 at a time; each output accumulates its K products in order with
// fma. Tiles are loaded so neighbouring threads read neighbouring addresses
// for either layout.

layout(local_size_x = 16, local_size_y = 16) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer A { float a[]; };
layout(std430, binding = 1) buffer B { float b[]; };
layout(std430, binding = 2) buffer C { float c[]; };
layout(push_constant) uniform P {
    uint M; uint N; uint K; uint trans_a; uint trans_b; uint a_off; uint b_off; uint c_off; uint batch_dims;
    uint batch_shape[4]; uint a_stride[4]; uint b_stride[4];
} p;

shared float As[16][17];  // As[i][k]: A(row0 + i, t + k)
shared float Bs[16][17];  // Bs[k][j]: B(t + k, col0 + j)

void main() {
    uint lx = gl_LocalInvocationID.x, ly = gl_LocalInvocationID.y;
    uint row0 = gl_WorkGroupID.y * 16u, col0 = gl_WorkGroupID.x * 16u;
    uint batch = gl_WorkGroupID.z;
    uint a_base = p.a_off, b_base = p.b_off, rem = batch;
    for (int d = int(p.batch_dims) - 1; d >= 0; --d) {
        uint idx = rem % p.batch_shape[d];
        rem /= p.batch_shape[d];
        a_base += idx * p.a_stride[d];
        b_base += idx * p.b_stride[d];
    }
    float acc = 0.0;
    for (uint t = 0u; t < p.K; t += 16u) {
        if (p.trans_a == 0u) {  // A(i, k) at i * K + k
            uint i = row0 + ly, k = t + lx;
            As[ly][lx] = (i < p.M && k < p.K) ? a[a_base + i * p.K + k] : 0.0;
        } else {                // A(i, k) at k * M + i
            uint i = row0 + lx, k = t + ly;
            As[lx][ly] = (i < p.M && k < p.K) ? a[a_base + k * p.M + i] : 0.0;
        }
        if (p.trans_b == 0u) {  // B(k, j) at k * N + j
            uint k = t + ly, j = col0 + lx;
            Bs[ly][lx] = (k < p.K && j < p.N) ? b[b_base + k * p.N + j] : 0.0;
        } else {                // B(k, j) at j * K + k
            uint k = t + lx, j = col0 + ly;
            Bs[lx][ly] = (k < p.K && j < p.N) ? b[b_base + j * p.K + k] : 0.0;
        }
        barrier();
        for (uint q = 0u; q < 16u; ++q) acc = fma(As[ly][q], Bs[q][lx], acc);
        barrier();
    }
    uint row = row0 + ly, col = col0 + lx;
    if (row < p.M && col < p.N) c[p.c_off + (batch * p.M + row) * p.N + col] = acc;
}
