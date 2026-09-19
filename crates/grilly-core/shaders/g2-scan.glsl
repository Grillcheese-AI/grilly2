#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
// Inclusive scans along the middle axis of a contiguous (outer, len, inner)
// view, one invocation per line, forward or reversed:
//   CUMSUM    float32, Kahan-compensated (torch's CPU cumsum accumulates in
//             double), or int64 exactly;
//   LOGCUMSUMEXP  float32, y_i = m_i + log(S_i) with m_i the running max
//             and S_i = sum_j exp(x_j - m_i) Kahan-compensated (rescaled when
//             the max moves), so the error does not grow with the length.
//             (Chaining y_i = logaddexp(y_{i-1}, x_i) did: 170 ULP at 8192.)

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer I { uint src[]; };
layout(std430, binding = 1) buffer O { uint dst[]; };
layout(push_constant) uniform P {
    uint outer; uint len; uint inner; uint op; uint reverse; uint is_int; uint in_off; uint out_off;
} p;

// Op codes; keep in sync with grilly_core::generic::Scan.
const uint CUMSUM = 0, LOGCUMSUMEXP = 1;

void main() {
    uint i = flat_index();
    if (i >= p.outer * p.inner) return;
    uint o = i / p.inner, j = i % p.inner;
    uint line = o * p.len * p.inner + j;
    if (p.is_int == 1u) {
        int64_t acc = 0l;
        for (uint t = 0u; t < p.len; ++t) {
            uint k = line + (p.reverse == 1u ? p.len - 1u - t : t) * p.inner;
            uint a = 2u * (p.in_off + k), b = 2u * (p.out_off + k);
            acc += int64_t(uint64_t(src[a]) | (uint64_t(src[a + 1u]) << 32));
            dst[b] = uint(uint64_t(acc) & 0xffffffffUL);
            dst[b + 1u] = uint(uint64_t(acc) >> 32);
        }
        return;
    }
    if (p.op == CUMSUM) {
        precise float acc = 0.0, comp = 0.0;
        for (uint t = 0u; t < p.len; ++t) {
            uint k = line + (p.reverse == 1u ? p.len - 1u - t : t) * p.inner;
            precise float y = uintBitsToFloat(src[p.in_off + k]) - comp;
            precise float s = acc + y;
            comp = isinf(s) ? 0.0 : (s - acc) - y;  // see g2-reduce: keep inf, not NaN
            acc = s;
            dst[p.out_off + k] = floatBitsToUint(acc);
        }
        return;
    }
    const float NEG_INF = uintBitsToFloat(0xff800000u);
    float m = NEG_INF;
    precise float sum = 0.0, comp = 0.0;
    bool nan_seen = false;
    for (uint t = 0u; t < p.len; ++t) {
        uint k = line + (p.reverse == 1u ? p.len - 1u - t : t) * p.inner;
        float x = uintBitsToFloat(src[p.in_off + k]);
        nan_seen = nan_seen || isnan(x);
        float y;
        if (nan_seen) {
            y = NAN;
        } else if (isinf(m) && m > 0.0 || isinf(x) && x > 0.0) {
            m = uintBitsToFloat(0x7f800000u);  // +inf absorbs everything after it
            y = m;
        } else {
            if (x > m) {  // the max moves: rescale what is summed so far
                float r = exp_acc(m - x);  // exp(-inf) = 0 on the first element
                sum *= r;
                comp *= r;
                m = x;
            }
            if (isinf(m)) {
                y = NEG_INF;  // everything so far is -inf
            } else {
                precise float term = exp_acc(x - m) - comp;
                precise float s = sum + term;
                comp = (s - sum) - term;
                sum = s;
                y = m + log(sum);
            }
        }
        dst[p.out_off + k] = floatBitsToUint(y);
    }
}
