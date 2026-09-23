#version 450
#extension GL_GOOGLE_include_directive : require
// One timestep of an integrate-and-fire neuron, charge through reset, in a
// single dispatch.
//
//   h = decay * (v - rest) + rest + in_scale * x       charge
//   s = h >= threshold                                 fire
//   v' = soft ? h - threshold * s                      reset
//            : h * (1 - s) + rest * s
//
// `decay = 1 - 1/tau` and `in_scale = 1/tau` or 1, so IF (decay 1,
// in_scale 1) and both of LIF's input conventions are the same code.
// A soft reset has `rest = 0` and keeps the overshoot; a hard one clears to
// `rest`.
//
// Why it is fused: composed from ordinary ops this is 12 dispatches, each
// reading and writing the whole tensor. Here it is one, reading v and x
// once and writing s and v' once — the arithmetic was never the cost, and
// the measured step goes 0.841 ms to 0.068 ms at (1024, 4096).
//
// `h` is written out when a gradient will be wanted, because the backward
// needs the membrane *before* the reset and recomputing it there would mean
// keeping x and the old v alive instead. Under inference it is skipped, so
// the step writes two tensors rather than three.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer V { float v[]; };
layout(std430, binding = 1) readonly buffer X { float x[]; };
layout(std430, binding = 2) writeonly buffer S { float s[]; };
layout(std430, binding = 3) writeonly buffer VN { float v_next[]; };
layout(std430, binding = 4) writeonly buffer H { float h_out[]; };
layout(push_constant) uniform P {
    uint n; float decay; float in_scale; float rest;
    float threshold; uint soft; uint v_off; uint x_off;
    uint want_h;
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;

    float prev = v[p.v_off + i];
    float h = p.decay * (prev - p.rest) + p.rest + p.in_scale * x[p.x_off + i];
    float spike = h >= p.threshold ? 1.0 : 0.0;

    if (p.want_h != 0u) h_out[i] = h;
    s[i] = spike;
    v_next[i] = p.soft != 0u ? h - p.threshold * spike
                             : h * (1.0 - spike) + p.rest * spike;
}
