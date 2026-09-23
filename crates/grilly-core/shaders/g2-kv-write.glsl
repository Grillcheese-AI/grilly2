#version 450
#extension GL_GOOGLE_include_directive : require
// Write a step's keys or values into a KV cache at the position the cache's
// own cursor holds, optionally rotating them on the way (g2-rope's exact
// arithmetic, so a rotated key matches the same rope computed separately,
// bit for bit).
//
// Why this exists rather than g2-copy or g2-rope with a pushed offset: the
// destination position changes every decode step, and a pushed value is
// baked into a recorded command and replayed stale (grilly_core::graph).
// Reading it from `lens` is what lets a decode step be captured once.
//
// Source and destination are addressed by strides, which covers both layouts
// the cache is fed from: a (batch, heads, T, dim) tensor, and `heads` columns
// of a (batch, T, row) projection read in place at their column.
//
// A key is rotated at its own ABSOLUTE position, which is the slot it lands
// in plus `pos_base` — the absolute position the cache's slot 0 holds. The
// two are usually equal (a cache that starts at the start of the sequence
// has base 0), but they are separate quantities and only the slot comes
// from the cursor. Keeping the base a push constant is safe: it is fixed
// for a cache's lifetime, where the slot changes every step.
//
// When `rope` is 0 the cos/sin bindings are not read; the caller binds any
// buffer there (the source, in practice) because a descriptor set has to be
// complete. One invocation per element written.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer S { float src[]; };
layout(std430, binding = 1) readonly buffer C { float cosv[]; };
layout(std430, binding = 2) readonly buffer N { float sinv[]; };
layout(std430, binding = 3) writeonly buffer D { float dst[]; };
layout(std430, binding = 4) readonly buffer L { uint lens[]; };
layout(push_constant) uniform P {
    uint n; uint t; uint heads; uint dim;
    uint layer; uint max_len; uint src_off; uint rope;
    uint ss_b; uint ss_h; uint ss_t;
    uint ds_b; uint ds_h; uint ds_t; uint pos_base;
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    // A full cache takes nothing more. The host refuses an overrun before it
    // records; a replay has no host, so it saturates here instead of writing
    // past the end (docs/capture.md).
    uint pos = lens[p.layer];
    if (pos + p.t > p.max_len) return;

    uint d = i % p.dim, rest = i / p.dim;
    uint h = rest % p.heads;
    rest /= p.heads;
    uint t = rest % p.t, b = rest / p.t;
    uint si = p.src_off + b * p.ss_b + h * p.ss_h + t * p.ss_t;

    float y;
    if (p.rope != 0u) {
        uint half_dim = p.dim / 2u;
        float xv = src[si + d];
        float rv = d < half_dim ? -src[si + d + half_dim] : src[si + d - half_dim];
        uint ti = (p.pos_base + pos + t) * p.dim + d;
        precise float a = xv * cosv[ti];
        precise float c = rv * sinv[ti];
        precise float r = a + c;
        y = r;
    } else {
        y = src[si + d];
    }
    dst[pos * p.dim + b * p.ds_b + h * p.ds_h + t * p.ds_t + d] = y;
}
