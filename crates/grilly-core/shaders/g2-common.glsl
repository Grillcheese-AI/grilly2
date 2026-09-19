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
