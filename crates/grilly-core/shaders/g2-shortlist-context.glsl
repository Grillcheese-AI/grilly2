#version 450
// Shortlist head, dispatch 1 of 8 (grilly_infer::shortlist): the context
// pass, `scores[c] = x . centroids[c]` for every cluster centroid, eight
// centroids a workgroup, reduced as a tree in shared memory.
//
// Workgroup 0 also does the step's setup, since the next dispatch needs it:
// it packs the state's sign bits (`x > 0`, as g2-shortlist-pack) and clears
// the histogram of Hamming distances.

layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer Cn { float cent[]; };
layout(std430, binding = 2) writeonly buffer S { float scores[]; };
layout(std430, binding = 3) writeonly buffer Q { uint q[]; };
layout(std430, binding = 4) writeonly buffer H { uint hist[]; };
layout(push_constant) uniform P { uint clusters; uint dim; uint x_off; } p;

const uint ROWS = 8u;
shared float part[ROWS * 256u];

void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    if (g == 0u) {
        uint words = p.dim / 32u;
        if (t < words) {
            uint bits = 0u;
            for (uint b = 0u; b < 32u; ++b) {
                if (x[p.x_off + t * 32u + b] > 0.0) bits |= 1u << b;
            }
            q[t] = bits;
        }
        for (uint i = t; i <= p.dim; i += 256u) hist[i] = 0u;
    }

    uint r0 = g * ROWS;
    float acc[ROWS];
    for (uint k = 0u; k < ROWS; ++k) acc[k] = 0.0;
    for (uint i = t; i < p.dim; i += 256u) {
        float xv = x[p.x_off + i];
        for (uint k = 0u; k < ROWS; ++k) {
            if (r0 + k < p.clusters) acc[k] += xv * cent[(r0 + k) * p.dim + i];
        }
    }
    for (uint k = 0u; k < ROWS; ++k) part[k * 256u + t] = acc[k];
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) {
            for (uint k = 0u; k < ROWS; ++k) part[k * 256u + t] += part[k * 256u + t + s];
        }
        barrier();
    }
    if (t < ROWS && r0 + t < p.clusters) scores[r0 + t] = part[t * 256u];
}
