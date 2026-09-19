#version 450
#extension GL_GOOGLE_include_directive : require
// Rotary position embedding as transformers applies it:
//   y = x * cos + rotate_half(x) * sin,   rotate_half(x) = cat(-x2, x1)
// with the same two multiplies and one add, so it matches those ops
// computed one at a time, bit for bit. Heads are read straight from a
// projection's (batch, T, row) layout, starting at column `col`, and written
// to any strided destination: a (batch, heads, T, dim) tensor, or a KV cache
// at its next position. One invocation per output element.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer S { float src[]; };
layout(std430, binding = 1) readonly buffer C { float cosv[]; };
layout(std430, binding = 2) readonly buffer N { float sinv[]; };
layout(std430, binding = 3) writeonly buffer D { float dst[]; };
layout(push_constant) uniform P {
    uint n; uint t; uint heads; uint dim;
    uint row; uint col; uint src_off; uint pos0;
    uint dst_off; uint ds_b; uint ds_h; uint ds_t;
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    uint d = i % p.dim, rest = i / p.dim;
    uint h = rest % p.heads;
    rest /= p.heads;
    uint t = rest % p.t, b = rest / p.t;
    uint si = p.src_off + (b * p.t + t) * p.row + p.col + h * p.dim;
    uint half_dim = p.dim / 2u;
    float xv = src[si + d];
    float rv = d < half_dim ? -src[si + d + half_dim] : src[si + d - half_dim];
    uint ti = (p.pos0 + t) * p.dim + d;
    precise float a = xv * cosv[ti];
    precise float c = rv * sinv[ti];
    precise float y = a + c;
    dst[p.dst_off + b * p.ds_b + h * p.ds_h + t * p.ds_t + d] = y;
}
