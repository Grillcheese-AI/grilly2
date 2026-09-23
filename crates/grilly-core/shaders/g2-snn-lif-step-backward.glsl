#version 450
#extension GL_GOOGLE_include_directive : require
// The backward of one integrate-and-fire timestep (g2-snn-lif-step).
//
// The spike is a step function, so its derivative is replaced by a
// surrogate bump `g` evaluated at `h - threshold` — that substitution is
// the whole reason a spiking network trains at all. `kind` picks it:
//
//   0  ATan         g = a / (2 (1 + (pi a u / 2)^2))
//   1  Sigmoid      g = a sig(a u) (1 - sig(a u))
//   2  FastSigmoid  g = a / (2 (1 + a |u|)^2)
//
// Both outputs feed back, so both gradients arrive:
//
//   dh = dL/ds * ds/dh + dL/dv' * dv'/dh
//   ds/dh  = g(h - threshold)
//   dv'/dh = soft ? 1 - threshold * ds/dh
//                 : (1 - s) + (rest - h) * ds/dh
//
// and with `detach_reset` the reset's own path is cut, leaving dv'/dh at 1
// or (1 - s). That shortens the route back through time, which is often
// what makes a deep spiking net trainable.
//
// Then dL/dx = dh * in_scale and dL/dv = dh * decay, since the charge is
// linear in both.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer H { float h[]; };
layout(std430, binding = 1) readonly buffer S { float s[]; };
layout(std430, binding = 2) readonly buffer GS { float g_spike[]; };
layout(std430, binding = 3) readonly buffer GV { float g_vnext[]; };
layout(std430, binding = 4) writeonly buffer GX { float g_x[]; };
layout(std430, binding = 5) writeonly buffer GVP { float g_vprev[]; };
layout(push_constant) uniform P {
    uint n; float decay; float in_scale; float rest;
    float threshold; uint soft; uint detach_reset; uint kind;
    float alpha; uint has_gs; uint has_gv; uint pad;
} p;

float surrogate(float u) {
    if (p.kind == 0u) {                       // ATan
        float t = (3.14159265358979 * p.alpha / 2.0) * u;
        return p.alpha / (2.0 * (1.0 + t * t));
    }
    if (p.kind == 1u) {                       // Sigmoid
        float sig = 1.0 / (1.0 + exp(-p.alpha * u));
        return p.alpha * sig * (1.0 - sig);
    }
    float d = 1.0 + p.alpha * abs(u);         // FastSigmoid
    return p.alpha / (2.0 * d * d);
}

void main() {
    uint i = flat_index();
    if (i >= p.n) return;

    float hi = h[i], si = s[i];
    float ds_dh = surrogate(hi - p.threshold);

    float dv_dh;
    if (p.detach_reset != 0u) {
        dv_dh = p.soft != 0u ? 1.0 : 1.0 - si;
    } else if (p.soft != 0u) {
        dv_dh = 1.0 - p.threshold * ds_dh;
    } else {
        dv_dh = (1.0 - si) + (p.rest - hi) * ds_dh;
    }

    float dh = 0.0;
    if (p.has_gs != 0u) dh += g_spike[i] * ds_dh;
    if (p.has_gv != 0u) dh += g_vnext[i] * dv_dh;

    g_x[i] = dh * p.in_scale;
    g_vprev[i] = dh * p.decay;
}
