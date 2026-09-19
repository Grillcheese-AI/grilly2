//! `div_rn` in g2-common.glsl against IEEE division (Rust's `/` on f32).
//! Vulkan lets OpFDiv be 2.5 ulp off; the int8 quantizer needs torch's
//! correctly rounded `absmax / 127.5` and `1.0 / scale` to reproduce
//! torchao's quantized values exactly.
//!
//! Tolerance: bit-exact, on 2^24 random pairs with normal operands and
//! quotients, and on the quantizer's two divisions over random values.

use grilly_core::{Arg, Device, Kernel, Tensor};

const COMMON: &str = include_str!("../shaders/g2-common.glsl");

fn gpu_div(dev: &Device, a: &[f32], b: &[f32]) -> Vec<f32> {
    let source = format!(
        "#version 450\nlayout(local_size_x = 256) in;\n{COMMON}\n\
         layout(std430, binding = 0) readonly buffer A {{ float a[]; }};\n\
         layout(std430, binding = 1) readonly buffer B {{ float b[]; }};\n\
         layout(std430, binding = 2) writeonly buffer Q {{ float q[]; }};\n\
         layout(push_constant) uniform P {{ uint n; }} p;\n\
         void main() {{ uint i = flat_index(); if (i < p.n) q[i] = div_rn(a[i], b[i]); }}\n"
    );
    let kernel = Kernel { shader: dev.generated_kernel(&source).unwrap(), bindings: 3, push_words: 1, local_size: [256, 1, 1] };
    let (ta, tb) = (Tensor::from_slice(dev, a, &[a.len()]).unwrap(), Tensor::from_slice(dev, b, &[b.len()]).unwrap());
    let q = Tensor::empty(dev, &[a.len()]).unwrap();
    kernel.launch_flat(dev, &[Arg::Tensor(&ta), Arg::Tensor(&tb), Arg::Tensor(&q)], a.len(), &[a.len() as u32]).unwrap();
    q.to_vec().unwrap()
}

/// xorshift64*, deterministic
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// a normal float with a random mantissa and sign, exponent in [lo, hi]
    fn float(&mut self, lo: i32, hi: i32) -> f32 {
        let r = self.next();
        let exp = (lo + (r % (hi - lo + 1) as u64) as i32 + 127) as u32;
        f32::from_bits(((r >> 32) as u32 & 0x8000_0000) | (exp << 23) | ((r >> 8) as u32 & 0x7F_FFFF))
    }
}

fn mismatches(a: &[f32], b: &[f32], got: &[f32]) -> Vec<(f32, f32, f32, f32)> {
    a.iter().zip(b).zip(got).filter(|((x, y), g)| (**x / **y).to_bits() != g.to_bits()).map(|((x, y), g)| (*x, *y, *g, x / y)).take(5).collect()
}

#[test]
fn div_rn_is_correctly_rounded() {
    let dev = Device::default_device().expect("open GPU");
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let n = 1 << 24;
    // random operands whose quotient stays normal
    let (a, b): (Vec<f32>, Vec<f32>) = (0..n).map(|_| (rng.float(-60, 60), rng.float(-60, 60))).unzip();
    let bad = mismatches(&a, &b, &gpu_div(&dev, &a, &b));
    assert!(bad.is_empty(), "div_rn differs from IEEE division, e.g. (a, b, gpu, ieee): {bad:?}");

    // the quantizer's divisions: absmax / 127.5, then 1 / scale
    let absmax: Vec<f32> = (0..n).map(|_| rng.float(-20, 20).abs()).collect();
    let b = vec![127.5f32; n];
    let scales = gpu_div(&dev, &absmax, &b);
    assert!(mismatches(&absmax, &b, &scales).is_empty(), "absmax / 127.5");
    let ones = vec![1.0f32; n];
    let bad = mismatches(&ones, &scales, &gpu_div(&dev, &ones, &scales));
    assert!(bad.is_empty(), "1 / scale: {bad:?}");
}
