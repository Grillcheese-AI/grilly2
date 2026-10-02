#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// Check an int64 index tensor against [lo, size) on the device, recording
// the first offending value rather than telling the host about it now.
//
// Reading the indices back to validate them is what kept every model with
// an embedding out of a captured graph: a replay re-issues commands and
// cannot re-run the host, so an op that asks the host a question is
// refused. torch does not ask either — CUDA asserts on the device and the
// error surfaces at the next synchronize — and this is the same bargain.
//
// It is also cheaper. The host check was two reductions and a readback,
// and a readback ends the command batch; this is one dispatch and no
// submit.
//
// `lo` is 0, or -size where negative indices count from the end. The first
// thread to find a bad value claims the record with a compare-and-swap, so
// what the host reports is one real offending index rather than whichever
// thread finished last.
//
// `mode` picks the predicate. Mode 0 is the range check above; mode 1 is
// "is any element zero", which is an integer divisor. torch raises for an
// integer zero divisor and a GPU cannot, so that check used to be a
// download — and a download is a host read, so `x % 512` on an int64
// tensor could not be captured, which is every ring-buffer cursor.

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) readonly buffer I { uint idx[]; };
layout(std430, binding = 1) buffer F { int fault[]; };   // kind, value, size, dim
layout(push_constant) uniform P {
    uint n; uint idx_off; uint size; uint allow_negative; uint kind; uint dim; uint mode;
} p;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;

    uint pos = p.idx_off + i;
    int64_t v = int64_t(uint64_t(idx[2u * pos]) | (uint64_t(idx[2u * pos + 1u]) << 32));
    if (p.mode == 1u) {
        if (v != 0l) return;                 // a zero divisor is the fault
    } else {
        int64_t lo = p.allow_negative != 0u ? -int64_t(p.size) : 0l;
        if (v >= lo && v < int64_t(p.size)) return;
    }

    if (atomicCompSwap(fault[0], 0, int(p.kind)) == 0) {
        // saturate for reporting: a value this far out is already the bug
        int64_t shown = clamp(v, int64_t(-2147483647), int64_t(2147483647));
        fault[1] = int(shown);
        fault[2] = int(p.size);
        fault[3] = int(p.dim);
    }
}
