//! int8 weight-only quantization from Rust (no Python in the path).
//!
//! Tolerances:
//! - quantized values, scales and dequantized weights: bit-exact against
//!   torchao's formula evaluated on the host in float32, with IEEE division
//!   and round-half-to-even, as torch computes it (torchao 0.18,
//!   Int8Tensor.from_hp and quantize_affine);
//! - linear: against a float64 reference of `scale * sum(x * q) + bias`,
//!   within an error bound derived from how the kernel sums. Each lane adds
//!   up to 16 * ceil(vecs / 64) products in order, a 6-level tree combines
//!   64 lanes, then one multiply and one add follow:
//!   |err| <= (terms + 6 + 2) * 2^-24 * (scale * sum|x * q| + |bias|).
//!   The worst measured ratio to that bound is printed;
//! - gradients: grad_input = grad @ dequantize(W) runs through the generic
//!   matmul, whose summation order is its own, so it is held to the serial
//!   worst case (out_features + 2) * 2^-24 * sum|grad * w| against float64;
//!   grad_bias sums rows, (rows + 2) * 2^-24 * sum|grad|.

use grilly_core::{Device, Tensor};
use grilly_infer::Int8Weight;

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

/// deterministic values in [lo, hi]
fn data(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

/// torchao's per-row symmetric int8: (q, scale), on the host in float32.
fn reference(w: &[f32], k: usize) -> (Vec<i8>, Vec<f32>) {
    let (mut q, mut scales) = (Vec::new(), Vec::new());
    for row in w.chunks(k) {
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = (amax / 127.5).max(f32::EPSILON);
        let inv = 1.0 / scale;
        q.extend(row.iter().map(|v| (v * inv).round_ties_even().clamp(-128.0, 127.0) as i8));
        scales.push(scale);
    }
    (q, scales)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn quantized_values_are_torchaos_exactly() {
    let dev = dev();
    for (n, k) in [(1, 1), (3, 5), (5, 17), (4, 16), (7, 33), (64, 896), (9, 4864)] {
        let mut w = data(n * k, (n * 1000 + k) as u64, -3.0, 3.0);
        w[..k].fill(0.0); // an all-zero row: scale is eps, q is 0
        if n > 2 {
            // a row whose extremes land exactly on +-127.5 and ties at .5
            let row = &mut w[2 * k..3 * k];
            row[0] = 2.55;
            if k > 1 {
                row[1] = -2.55;
            }
            if k > 2 {
                row[2] = 0.01; // 0.01 / (2.55 / 127.5) = 0.5: rounds to even (0)
            }
        }
        let t = Tensor::from_slice(&dev, &w, &[n, k]).unwrap();
        let qw = Int8Weight::quantize(&t).unwrap();
        let (q, scale) = reference(&w, k);
        assert_eq!(qw.shape(), [n, k]);
        assert_eq!(qw.qdata().unwrap(), q, "({n}, {k}) q");
        assert_eq!(bits(&qw.scale().to_vec().unwrap()), bits(&scale), "({n}, {k}) scale");
        let deq: Vec<f32> = q.iter().enumerate().map(|(i, &v)| v as f32 * scale[i / k]).collect();
        assert_eq!(bits(&qw.dequantize().unwrap().to_vec().unwrap()), bits(&deq), "({n}, {k}) dequantize");
    }
}

#[test]
fn linear_matches_a_float64_reference_within_its_bound() {
    let dev = dev();
    let mut worst = 0.0f64;
    for (m, n, k) in [(1, 1, 1), (1, 5, 17), (3, 4, 16), (8, 7, 33), (9, 6, 100), (20, 13, 896), (1, 64, 4864)] {
        let w = data(n * k, (m + n + k) as u64, -1.0, 1.0);
        let x = data(m * k, (m * n * k) as u64 + 7, -2.0, 2.0);
        let b = data(n, 99, -0.5, 0.5);
        let qw = Int8Weight::quantize(&Tensor::from_slice(&dev, &w, &[n, k]).unwrap()).unwrap();
        let (q, scale) = reference(&w, k);
        let terms = 16 * k.div_ceil(16).div_ceil(64);
        let gamma = (terms + 8) as f64 * 2f64.powi(-24);
        let xt = Tensor::from_slice(&dev, &x, &[m, k]).unwrap();
        for bias in [None, Some(&b)] {
            let bt = bias.map(|b| Tensor::from_slice(&dev, b, &[n]).unwrap());
            let y = qw.linear(&xt, bt.as_ref()).unwrap().to_vec().unwrap();
            for r in 0..m {
                for c in 0..n {
                    let (mut dot, mut mag) = (0.0f64, 0.0f64);
                    for i in 0..k {
                        let p = x[r * k + i] as f64 * q[c * k + i] as f64;
                        dot += p;
                        mag += p.abs();
                    }
                    let bv = bias.map_or(0.0, |b| b[c] as f64);
                    let want = dot * scale[c] as f64 + bv;
                    let bound = gamma * (mag * scale[c] as f64 + bv.abs()) + f64::MIN_POSITIVE;
                    let err = (y[r * n + c] as f64 - want).abs();
                    worst = worst.max(err / bound);
                    assert!(err <= bound, "({m}, {n}, {k}) y[{r}][{c}] = {} vs {want}: error {err:e} > bound {bound:e}", y[r * n + c]);
                }
            }
        }
    }
    println!("linear: worst error / bound = {worst:.3}");
}

#[test]
fn gradients_flow_to_input_and_bias() {
    let dev = dev();
    let (m, n, k) = (5, 7, 37);
    let w = data(n * k, 11, -1.0, 1.0);
    let qw = Int8Weight::quantize(&Tensor::from_slice(&dev, &w, &[n, k]).unwrap()).unwrap();
    let wd = qw.dequantize().unwrap().to_vec().unwrap();
    let (xv, bv, gv) = (data(m * k, 12, -1.0, 1.0), data(n, 13, -1.0, 1.0), data(m * n, 14, -1.0, 1.0));
    let x = Tensor::from_slice(&dev, &xv, &[m, k]).unwrap();
    let b = Tensor::from_slice(&dev, &bv, &[n]).unwrap();
    x.set_requires_grad(true).unwrap();
    b.set_requires_grad(true).unwrap();
    let y = qw.linear(&x, Some(&b)).unwrap();
    y.backward(Some(&Tensor::from_slice(&dev, &gv, &[m, n]).unwrap()), false).unwrap();
    let (gx, gb) = (x.grad().unwrap().to_vec().unwrap(), b.grad().unwrap().to_vec().unwrap());
    let u = 2f64.powi(-24);
    for r in 0..m {
        for i in 0..k {
            let (mut want, mut mag) = (0.0f64, 0.0f64);
            for c in 0..n {
                let p = gv[r * n + c] as f64 * wd[c * k + i] as f64;
                want += p;
                mag += p.abs();
            }
            let err = (gx[r * k + i] as f64 - want).abs();
            assert!(err <= (n + 2) as f64 * u * mag + f64::MIN_POSITIVE, "grad_x[{r}][{i}]: {} vs {want}", gx[r * k + i]);
        }
    }
    for c in 0..n {
        let want: f64 = (0..m).map(|r| gv[r * n + c] as f64).sum();
        let mag: f64 = (0..m).map(|r| (gv[r * n + c] as f64).abs()).sum();
        assert!((gb[c] as f64 - want).abs() <= (m + 2) as f64 * u * mag, "grad_b[{c}]");
    }
}

#[test]
fn linear_leading_dims_and_refusals() {
    let dev = dev();
    let (n, k) = (6, 20);
    let w = data(n * k, 5, -1.0, 1.0);
    let qw = Int8Weight::quantize(&Tensor::from_slice(&dev, &w, &[n, k]).unwrap()).unwrap();
    let x = data(2 * 3 * k, 6, -1.0, 1.0);
    let flat = qw.linear(&Tensor::from_slice(&dev, &x, &[6, k]).unwrap(), None).unwrap();
    let batched = qw.linear(&Tensor::from_slice(&dev, &x, &[2, 3, k]).unwrap(), None).unwrap();
    assert_eq!(batched.shape(), [2, 3, n]);
    assert_eq!(bits(&batched.to_vec().unwrap()), bits(&flat.to_vec().unwrap()));
    assert_eq!(qw.linear(&Tensor::from_slice(&dev, &[], &[0, k]).unwrap(), None).unwrap().shape(), [0, n]);

    assert!(qw.linear(&Tensor::from_slice(&dev, &x[..2 * 7], &[2, 7]).unwrap(), None).is_err());
    assert!(Int8Weight::quantize(&Tensor::from_slice(&dev, &x[..k], &[k]).unwrap()).is_err());
    assert!(Int8Weight::quantize(&Tensor::from_slice(&dev, &[], &[3, 0]).unwrap()).is_err());
}
