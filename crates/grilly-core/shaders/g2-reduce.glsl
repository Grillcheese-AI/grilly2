#version 450
#extension GL_GOOGLE_include_directive : require
// Reduce the middle axis of a contiguous (outer, len, inner) view:
// out[o * inner + j] = op over r of in[(o * len + r) * inner + j].
// One invocation per output, summing in order with Kahan compensation
// (`precise` keeps the compiler from reassociating it away).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { float src[]; };
layout(std430, binding = 1) buffer O { float dst[]; };
layout(push_constant) uniform P { uint outer; uint len; uint inner; uint op; uint in_off; uint out_off; } p;

// Op codes; keep in sync with grilly_core::generic::Reduce.
const uint SUM = 0, MEAN = 1, MAX = 2, MIN = 3;

void main() {
    uint i = flat_index();
    if (i >= p.outer * p.inner) return;
    uint o = i / p.inner, j = i % p.inner;
    uint base = p.in_off + o * p.len * p.inner + j;
    float r;
    if (p.op == MAX || p.op == MIN) {
        r = src[base];
        for (uint k = 1; k < p.len; ++k) {
            float v = src[base + k * p.inner];
            // torch propagates NaN through max/min
            r = (isnan(v) || isnan(r)) ? NAN : ((p.op == MAX) ? max(r, v) : min(r, v));
        }
    } else {
        precise float sum = 0.0;
        precise float comp = 0.0;
        for (uint k = 0; k < p.len; ++k) {
            precise float yv = src[base + k * p.inner] - comp;
            precise float t = sum + yv;
            comp = (t - sum) - yv;
            sum = t;
        }
        r = (p.op == MEAN) ? sum / float(p.len) : sum;
    }
    dst[p.out_off + i] = r;
}
