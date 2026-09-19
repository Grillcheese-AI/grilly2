// Shared by grilly2's generic kernels (#include-d, not compiled alone).
//
// Every elementwise kernel runs one invocation per output element over a
// 2-D grid (x groups, y rows of groups), so element counts beyond
// 65535 * 256 still fit the dispatch limits. Buffers are bound whole;
// tensors are located by element offsets in the push constants, so views
// never need aligned binding offsets.

#define MAX_DIMS 6

uint flat_index() {
    return gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x
         + gl_GlobalInvocationID.x;
}

#define NAN uintBitsToFloat(0x7fc00000u)

// The GPU's native exp is ~14 ULP off (measured); this is Cephes' expf:
// Cody-Waite range reduction x = k ln2 + r, then a degree-6 polynomial,
// about 1 ULP.
float exp_acc(float v) {
    if (v > 88.72283935546875) return uintBitsToFloat(0x7f800000u);
    if (v < -103.97207641601562) return 0.0;
    float k = floor(v * 1.44269504088896341 + 0.5);
    float r = v - k * 0.693359375;
    r = r + k * 2.12194440e-4;
    float z = r * r;
    float y = ((((( 1.9875691500E-4 * r + 1.3981999507E-3) * r + 8.3334519073E-3) * r
               + 4.1665795894E-2) * r + 1.6666665459E-1) * r + 5.0000001201E-1) * z + r + 1.0;
    return ldexp(y, int(k));
}

// Cephes' tanhf: an odd polynomial below |x| = 0.625, where 1 - exp(-2|x|)
// would cancel; the exp form above it; +-1 beyond float precision.
float tanh_acc(float v) {
    float a = abs(v);
    if (a < 0.625) {
        float z = v * v;
        return ((((-5.70498872745E-3 * z + 2.06390887954E-2) * z - 5.37397155531E-2) * z
                 + 1.33314422036E-1) * z - 3.33332819422E-1) * z * v + v;
    }
    if (a > 9.0) return sign(v);
    float r = 1.0 - 2.0 / (exp_acc(2.0 * a) + 1.0);
    return v < 0.0 ? -r : r;
}

// erf by Abramowitz & Stegun 7.1.26: absolute error < 1.5e-7.
float erf_approx(float v) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(v));
    float poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
               + t * (-1.453152027 + t * 1.061405429))));
    float r = 1.0 - poly * exp_acc(-v * v);
    return v < 0.0 ? -r : r;
}

float power(float v, float e) {
    if (e == 0.0) return 1.0;
    if (e == 0.5) return sqrt(v);              // as torch does
    if (e == floor(e) && abs(e) <= 64.0) {
        // Square-and-multiply: a few roundings instead of exp2(e * log2(v)).
        uint k = uint(abs(e));
        float base = v, acc = 1.0;
        while (k > 0u) {
            if ((k & 1u) != 0u) acc *= base;
            base *= base;
            k >>= 1u;
        }
        return e < 0.0 ? 1.0 / acc : acc;
    }
    if (v < 0.0) {
        // Negative bases need integer exponents, as in torch (else NaN).
        if (e != floor(e)) return NAN;
        float m = pow(-v, e);
        return mod(e, 2.0) == 0.0 ? m : -m;
    }
    return pow(v, e);
}

// Cephes sinf/cosf: reduce |x| by pi/4 in three parts (Cody-Waite), then a
// minimax polynomial on [-pi/4, pi/4]; about 1 ULP for |x| <= 8192. Larger
// arguments would need Payne-Hanek reduction.
float sin_cos_acc(float v, bool want_cos) {
    float a = abs(v);
    float sgn = (!want_cos && v < 0.0) ? -1.0 : 1.0;
    int j = int(a * 1.27323954473516);
    float y = float(j);
    if ((j & 1) != 0) { j += 1; y += 1.0; }
    j &= 7;
    if (j > 3) { sgn = -sgn; j -= 4; }
    if (want_cos && j > 1) sgn = -sgn;
    float x = ((a - y * 0.78515625) - y * 2.4187564849853515625e-4) - y * 3.77489497744594108e-8;
    float z = x * x;
    bool use_cos_poly = want_cos ? (j == 0 || j == 3) : (j == 1 || j == 2);
    float r;
    if (use_cos_poly) {
        r = ((2.443315711809948E-005 * z - 1.388731625493765E-003) * z + 4.166664568298827E-002) * z * z
            - 0.5 * z + 1.0;
    } else {
        r = ((-1.9515295891E-4 * z + 8.3321608736E-3) * z - 1.6666654611E-1) * z * x + x;
    }
    return sgn * r;
}

// Compensated sums, shared by the reductions. A value is carried as
// (hi, lo), true ~ hi + lo; once hi is infinite the error term is dropped
// (it would be inf - inf = NaN), so inf + x stays inf.

// (hi, lo) += v, Kahan style
void kahan_add(inout float hi, inout float lo, float v) {
    precise float y = v + lo;
    precise float t = hi + y;
    lo = isinf(t) ? 0.0 : y - (t - hi);
    hi = t;
}

// (a_hi, a_lo) += (b_hi, b_lo), two-sum on the high parts
void pair_add(inout float a_hi, inout float a_lo, float b_hi, float b_lo) {
    precise float t = a_hi + b_hi;
    precise float bp = t - a_hi;
    precise float err = (a_hi - (t - bp)) + (b_hi - bp);
    a_lo = isinf(t) ? 0.0 : a_lo + b_lo + err;
    a_hi = t;
}

// ── elementwise ops, shared by g2-unary, g2-binary and fused kernels ─────
// One definition per op, so a fused kernel (grilly_core::fusion) computes
// exactly what the unfused kernels do. Results are `precise`: no
// contraction or reassociation across ops, whatever the call site.

// Unary op codes; keep in sync with grilly_core::generic::Unary.
const uint UN_NEG = 0, UN_ABS = 1, UN_SIGN = 2, UN_EXP = 3, UN_LOG = 4, UN_SQRT = 5, UN_RSQRT = 6,
           UN_TANH = 7, UN_SIGMOID = 8, UN_ADD_S = 9, UN_MUL_S = 10, UN_RSUB_S = 11, UN_RDIV_S = 12,
           UN_POW_S = 13, UN_CLAMP = 14, UN_CLAMP_MASK = 15, UN_GELU = 16, UN_GELU_GRAD = 17,
           UN_IS_FINITE = 18, UN_IS_INF = 19, UN_IS_NAN = 20, UN_SIN = 21, UN_COS = 22, UN_FLOOR = 23, UN_CEIL = 24,
           UN_TRUNC = 25, UN_ROUND = 26;

// f(v; a, b)
float g2_unary(uint op, float v, float a, float b) {
    precise float r;
    switch (op) {
        case UN_NEG: r = -v; break;
        case UN_ABS: r = abs(v); break;
        case UN_SIGN: r = sign(v); break;
        case UN_EXP: r = exp_acc(v); break;
        case UN_LOG: r = log(v); break;
        case UN_SQRT: r = sqrt(v); break;
        case UN_RSQRT: r = 1.0 / sqrt(v); break;
        case UN_TANH: r = tanh_acc(v); break;
        case UN_SIGMOID: {
            float e = exp_acc(-abs(v));
            r = v >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
            break;
        }
        case UN_ADD_S: r = v + a; break;
        case UN_MUL_S: r = v * a; break;
        case UN_RSUB_S: r = a - v; break;
        case UN_RDIV_S: r = a / v; break;
        case UN_POW_S: r = power(v, a); break;
        case UN_CLAMP: r = min(max(v, a), b); break;
        case UN_CLAMP_MASK: r = (v >= a && v <= b) ? 1.0 : 0.0; break;
        case UN_GELU: r = 0.5 * v * (1.0 + erf_approx(v * 0.70710678118654752)); break;
        case UN_GELU_GRAD: {
            float cdf = 0.5 * (1.0 + erf_approx(v * 0.70710678118654752));
            float pdf = exp_acc(-0.5 * v * v) * 0.39894228040143268;
            r = cdf + v * pdf;
            break;
        }
        case UN_SIN: r = sin_cos_acc(v, false); break;
        case UN_COS: r = sin_cos_acc(v, true); break;
        case UN_IS_FINITE: r = (isnan(v) || isinf(v)) ? 0.0 : 1.0; break;
        case UN_IS_INF: r = isinf(v) ? 1.0 : 0.0; break;
        case UN_IS_NAN: r = isnan(v) ? 1.0 : 0.0; break;
        case UN_FLOOR: r = floor(v); break;
        case UN_CEIL: r = ceil(v); break;
        case UN_TRUNC: r = trunc(v); break;
        case UN_ROUND: r = roundEven(v); break;  // torch rounds half to even
        default: r = NAN;
    }
    return r;
}

// Binary op codes; keep in sync with grilly_core::generic::Binary.
// Comparisons give 1.0 / 0.0.
const uint BIN_ADD = 0, BIN_SUB = 1, BIN_MUL = 2, BIN_DIV = 3, BIN_POW = 4, BIN_MAXIMUM = 5, BIN_MINIMUM = 6,
           BIN_EQ = 7, BIN_NE = 8, BIN_GT = 9, BIN_GE = 10, BIN_LT = 11, BIN_LE = 12;

// op(x, y)
float g2_binary(uint op, float x, float y) {
    precise float r;
    switch (op) {
        case BIN_ADD: r = x + y; break;
        case BIN_SUB: r = x - y; break;
        case BIN_MUL: r = x * y; break;
        case BIN_DIV: r = x / y; break;
        case BIN_POW: r = power(x, y); break;
        case BIN_MAXIMUM: r = (isnan(x) || isnan(y)) ? NAN : max(x, y); break;
        case BIN_MINIMUM: r = (isnan(x) || isnan(y)) ? NAN : min(x, y); break;
        case BIN_EQ: r = x == y ? 1.0 : 0.0; break;
        case BIN_NE: r = x != y ? 1.0 : 0.0; break;
        case BIN_GT: r = x > y ? 1.0 : 0.0; break;
        case BIN_GE: r = x >= y ? 1.0 : 0.0; break;
        case BIN_LT: r = x < y ? 1.0 : 0.0; break;
        case BIN_LE: r = x <= y ? 1.0 : 0.0; break;
        default: r = NAN;
    }
    return r;
}
