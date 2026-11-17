#version 450
// Shortlist head, dispatch 7 of 8 (grilly_infer::shortlist): the union's
// rows scored, straight from the head's weight (no gather), 64 rows a
// workgroup. A row is in the union when it is in the context list (first
// position before the cut, or a successor), or among the k nearest by
// Hamming distance (below T, or at T up to the last tied id taken).
//
// Whether a row is in is the same for the whole workgroup, so the loop's
// branch and its barriers are uniform. Each workgroup leaves its best row
// (the first on a tie, since rows go in ascending order) and how many rows
// it scored; dispatch 8 picks among them.

layout(local_size_x = 256) in;

const uint ROWS = 64u;
const uint NONE = 0xFFFFFFFFu;

layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer Wt { float w[]; };
layout(std430, binding = 2) readonly buffer D { uint dist[]; };
layout(std430, binding = 3) readonly buffer Fp { uint firstpos[]; };
layout(std430, binding = 4) readonly buffer Pl { uint plan[]; };
layout(std430, binding = 5) writeonly buffer Pt { uint part[]; };
layout(push_constant) uniform P { uint rows; uint dim; uint x_off; uint w_off; } p;

shared uint s_in[ROWS];
shared float red[256];

void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint T = plan[0], tie = plan[1], cut = plan[3];
    uint row0 = g * ROWS;
    if (t < ROWS) {
        uint r = row0 + t;
        bool in_union = false;
        if (r < p.rows) {
            uint d = dist[r];
            in_union = firstpos[r] < cut || d < T || (d == T && r <= tie);
        }
        s_in[t] = in_union ? 1u : 0u;
    }
    barrier();

    float best = 0.0;
    uint best_row = NONE, scored = 0u;
    for (uint k = 0u; k < ROWS; ++k) {
        if (s_in[k] == 0u) continue;            // uniform
        uint r = row0 + k;
        float acc = 0.0;
        for (uint i = t; i < p.dim; i += 256u) acc += x[p.x_off + i] * w[p.w_off + r * p.dim + i];
        red[t] = acc;
        barrier();
        for (uint s = 128u; s > 0u; s >>= 1) {
            if (t < s) red[t] += red[t + s];
            barrier();
        }
        float v = red[0];
        if (best_row == NONE || v > best) { best = v; best_row = r; }
        scored += 1u;
        barrier();                              // red is rewritten next row
    }
    if (t == 0u) {
        part[3u * g] = floatBitsToUint(best);
        part[3u * g + 1u] = best_row;
        part[3u * g + 2u] = scored;
    }
}
