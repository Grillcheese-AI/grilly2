#version 450
#extension GL_GOOGLE_include_directive : require
// SwiGLU: out = silu(gate) * up = (gate * sigmoid(gate)) * up, with the same
// op functions and order grilly's silu and mul use one at a time, so the
// result is theirs bit for bit. gate and up are read from (rows, row_stride)
// sources at their own columns: two separate projections, or the two
// halves of one fused gate_up projection. One invocation per output element.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer G { float gate[]; };
layout(std430, binding = 1) readonly buffer U { float up[]; };
layout(std430, binding = 2) writeonly buffer O { float o[]; };
layout(push_constant) uniform P {
    uint n; uint width;
    uint g_off; uint g_row; uint g_col;
    uint u_off; uint u_row; uint u_col;
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint r = i / p.width, j = i % p.width;
    float g = gate[p.g_off + r * p.g_row + p.g_col + j];
    float u = up[p.u_off + r * p.u_row + p.u_col + j];
    float s = g2_binary(BIN_MUL, g, g2_unary(UN_SIGMOID, g, 0.0, 0.0));
    o[i] = g2_binary(BIN_MUL, s, u);
}
