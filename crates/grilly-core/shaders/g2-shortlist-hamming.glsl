#version 450
// Shortlist head, dispatch 2 of 8 (grilly_infer::shortlist): every row's
// Hamming distance to the state's sign bits, and the histogram of those
// distances, which is what lets dispatch 3 find the k-th nearest without
// sorting anything.
//
// Eight lanes share a row and each reads one uvec4 of its code, so a
// workgroup's 32 rows of a pass are one contiguous 4 KB read (an earlier
// one-row-per-thread kernel walked each row alone and ran at a fraction of
// the card's bandwidth). Distances are counted into a shared histogram and
// only its non-empty bins go to the global one, so the global atomics are a
// few per bin per workgroup rather than one per row.
//
// Also resets every row's first position in the context list (dispatch 4
// takes minimums into it).

layout(local_size_x = 256) in;

const uint MAX_DIM = 4096u;
const uint LANES = 8u;

layout(std430, binding = 0) readonly buffer Q { uint q[]; };
layout(std430, binding = 1) readonly buffer Cd { uvec4 codes[]; };
layout(std430, binding = 2) writeonly buffer D { uint dist[]; };
layout(std430, binding = 3) writeonly buffer Fp { uint firstpos[]; };
layout(std430, binding = 4) buffer H { uint hist[]; };
layout(push_constant) uniform P { uint rows; uint words; uint tile; } p;

shared uint sq[MAX_DIM / 32u];
shared uint lh[MAX_DIM + 1u];
shared uint pc[256];

void main() {
    uint g = gl_WorkGroupID.x, t = gl_LocalInvocationID.x;
    uint bins = p.words * 32u + 1u;
    for (uint i = t; i < p.words; i += 256u) sq[i] = q[i];
    for (uint i = t; i < bins; i += 256u) lh[i] = 0u;
    barrier();

    uint lane = t % LANES, slot = t / LANES;
    uint quads = p.words / 4u;
    uint row0 = g * p.tile;
    for (uint pass = 0u; pass < p.tile; pass += 256u / LANES) {
        uint row = row0 + pass + slot;
        uint c = 0u;
        if (row < p.rows) {
            for (uint j = lane; j < quads; j += LANES) {
                uvec4 v = codes[row * quads + j];
                uint w = 4u * j;
                c += uint(bitCount(v.x ^ sq[w])) + uint(bitCount(v.y ^ sq[w + 1u]))
                   + uint(bitCount(v.z ^ sq[w + 2u])) + uint(bitCount(v.w ^ sq[w + 3u]));
            }
        }
        pc[t] = c;
        barrier();
        if (lane == 0u && row < p.rows) {
            uint d = 0u;
            for (uint l = 0u; l < LANES; ++l) d += pc[t + l];
            dist[row] = d;
            firstpos[row] = 0xFFFFFFFFu;
            atomicAdd(lh[d], 1u);
        }
        barrier();
    }
    for (uint i = t; i < bins; i += 256u) {
        if (lh[i] != 0u) atomicAdd(hist[i], lh[i]);
    }
}
