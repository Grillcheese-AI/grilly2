#version 450
#extension GL_GOOGLE_include_directive : require
// y[i] = f(x[i]; a, b). x and y may be the same buffer (in-place ops).

layout(local_size_x = 256) in;
#include "g2-common.glsl"

layout(std430, binding = 0) buffer X { float x[]; };
layout(std430, binding = 1) buffer Y { float y[]; };
layout(push_constant) uniform P { uint n; uint op; float a; float b; uint x_off; uint y_off; } p;

// Op codes; keep in sync with grilly_core::generic::Unary.
const uint NEG = 0, ABS = 1, SIGN = 2, EXP = 3, LOG = 4, SQRT = 5, RSQRT = 6,
           TANH = 7, SIGMOID = 8, ADD_S = 9, MUL_S = 10, RSUB_S = 11, RDIV_S = 12,
           POW_S = 13, CLAMP = 14, CLAMP_MASK = 15, GELU = 16, GELU_GRAD = 17,
           IS_FINITE = 18, IS_INF = 19, IS_NAN = 20, SIN = 21, COS = 22, FLOOR = 23, CEIL = 24,
           TRUNC = 25, ROUND = 26;

void main() {
    uint i = flat_index();
    if (i >= p.n) return;
    float v = x[p.x_off + i];
    float r;
    switch (p.op) {
        case NEG: r = -v; break;
        case ABS: r = abs(v); break;
        case SIGN: r = sign(v); break;
        case EXP: r = exp_acc(v); break;
        case LOG: r = log(v); break;
        case SQRT: r = sqrt(v); break;
        case RSQRT: r = 1.0 / sqrt(v); break;
        case TANH: r = tanh_acc(v); break;
        case SIGMOID: {
            float e = exp_acc(-abs(v));
            r = v >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
            break;
        }
        case ADD_S: r = v + p.a; break;
        case MUL_S: r = v * p.a; break;
        case RSUB_S: r = p.a - v; break;
        case RDIV_S: r = p.a / v; break;
        case POW_S: r = power(v, p.a); break;
        case CLAMP: r = min(max(v, p.a), p.b); break;
        case CLAMP_MASK: r = (v >= p.a && v <= p.b) ? 1.0 : 0.0; break;
        case GELU: r = 0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752)); break;
        case GELU_GRAD: {
            float cdf = 0.5 * (1.0 + erf_approx(v * 0.70710678118654752));
            float pdf = exp_acc(-0.5 * v * v) * 0.39894228040143268;
            r = cdf + v * pdf;
            break;
        }
        case SIN: r = sin_cos_acc(v, false); break;
        case COS: r = sin_cos_acc(v, true); break;
        case IS_FINITE: r = (isnan(v) || isinf(v)) ? 0.0 : 1.0; break;
        case IS_INF: r = isinf(v) ? 1.0 : 0.0; break;
        case IS_NAN: r = isnan(v) ? 1.0 : 0.0; break;
        case FLOOR: r = floor(v); break;
        case CEIL: r = ceil(v); break;
        case TRUNC: r = trunc(v); break;
        case ROUND: r = roundEven(v); break;  // torch rounds half to even
        default: r = NAN;
    }
    y[p.y_off + i] = r;
}
