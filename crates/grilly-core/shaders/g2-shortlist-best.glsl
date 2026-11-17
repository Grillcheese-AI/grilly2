#version 450
// Shortlist head, dispatch 8 of 8 (grilly_infer::shortlist), one workgroup:
// the best of the scoring workgroups' bests, the lower row on a tie (so the
// first index of the maximum, as `torch.argmax`), and the union's size.
//
// Out (uint): [0] the token, [1] rows scored, [2] short flag, [3] the
// context cut.

layout(local_size_x = 1024) in;

const uint NONE = 0xFFFFFFFFu;

layout(std430, binding = 0) readonly buffer Pt { uint part[]; };
layout(std430, binding = 1) readonly buffer Pl { uint plan[]; };
layout(std430, binding = 2) writeonly buffer O { uint outv[]; };
layout(push_constant) uniform P { uint groups; } p;

shared float s_v[1024];
shared uint s_r[1024];
shared uint s_n[1024];

/// Is (va, ra) the better pick than (vb, rb)?
bool better(float va, uint ra, float vb, uint rb) {
    if (ra == NONE) return false;
    if (rb == NONE) return true;
    return va > vb || (va == vb && ra < rb);
}

void main() {
    uint t = gl_LocalInvocationID.x;
    float v = 0.0;
    uint r = NONE, n = 0u;
    for (uint g = t; g < p.groups; g += 1024u) {
        float gv = uintBitsToFloat(part[3u * g]);
        uint gr = part[3u * g + 1u];
        n += part[3u * g + 2u];
        if (better(gv, gr, v, r)) { v = gv; r = gr; }
    }
    s_v[t] = v; s_r[t] = r; s_n[t] = n;
    barrier();
    for (uint s = 512u; s > 0u; s >>= 1) {
        if (t < s) {
            if (better(s_v[t + s], s_r[t + s], s_v[t], s_r[t])) { s_v[t] = s_v[t + s]; s_r[t] = s_r[t + s]; }
            s_n[t] += s_n[t + s];
        }
        barrier();
    }
    if (t == 0u) {
        outv[0] = s_r[0];
        outv[1] = s_n[0];
        outv[2] = plan[4];
        outv[3] = plan[3];
    }
}
