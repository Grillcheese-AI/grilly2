#version 450
#extension GL_GOOGLE_include_directive : require
// Advance a KV cache layer's cursor by `t` positions, on the device.
//
// The cursor is what g2-kv-write writes at and g2-attn-decode reads its
// length from, so advancing it has to be a command like any other: a host
// increment would not happen on a replay, and every replayed step would
// write the same slot and attend over the same positions.
//
// Saturating at max_len matches g2-kv-write, which declines to write into a
// full cache. One invocation.

layout(local_size_x = 1) in;

layout(std430, binding = 0) buffer L { uint lens[]; };
layout(push_constant) uniform P { uint layer; uint t; uint max_len; } p;

void main() {
    uint n = lens[p.layer] + p.t;
    lens[p.layer] = n > p.max_len ? p.max_len : n;
}
