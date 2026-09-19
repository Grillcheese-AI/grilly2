//! int4 weight-only quantization, from Rust.
//!
//! Tolerances:
//! - the levels, scales and zero points are **bit-exact** against the same
//!   formula evaluated here: the divisions are correctly rounded on the GPU
//!   (`div_rn`), as they are on the host (CLAUDE.md §9 case (a));
//! - `dequantize` is bit-exact for the same reason: `q * scale + min` is one
//!   fused multiply-add on both sides;
//! - `linear` is measured against a **float64** evaluation of
//!   `x @ dequantize(w)^T + bias` under a bound derived from the kernel's
//!   summation: each lane takes whole groups, sums `group` products and the
//!   plain sum of x within a group, then a 64-lane tree, so the chain is
//!   about `k / 64 + 6` long. The worst ratio to the bound is printed.
//!
//! The scheme is torchao's `Int4WeightOnlyConfig` (group-wise, asymmetric,
//! float zero-point domain). torchao is not installed here, so — unlike the
//! int8 path, whose transcription was checked against torchao 0.18 itself —
//! this is a transcription of the published formula only. What is asserted
//! is that grilly computes that formula exactly.

use grilly_core::{Device, Tensor};
use grilly_infer::Int4Weight;

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn data(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + ((s >> 40) as f32 / (1u64 << 24) as f32) * (hi - lo)
        })
        .collect()
}

/// torchao's int4 group-wise asymmetric quantization, in float32.
fn reference(w: &[f32], k: usize, group: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let rows = w.len() / k;
    let groups = k.div_ceil(group);
    let (mut q, mut scale, mut zero) = (vec![0u8; rows * k], vec![0.0; rows * groups], vec![0.0; rows * groups]);
    for r in 0..rows {
        for g in 0..groups {
            let (start, end) = (g * group, ((g + 1) * group).min(k));
            let row = &w[r * k + start..r * k + end];
            let lo = row.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s = ((hi - lo) / 15.0).max(f32::EPSILON);
            scale[r * groups + g] = s;
            zero[r * groups + g] = lo + 8.0 * s;
            for (i, &v) in row.iter().enumerate() {
                // round half to even, as torch's round does
                let level = ((v - lo) / s).round_ties_even().clamp(0.0, 15.0);
                q[r * k + start + i] = level as u8;
            }
        }
    }
    (q, scale, zero)
}

/// The packed levels, unpacked: two per byte, earlier value in the low nibble.
fn levels(weight: &Int4Weight) -> Vec<u8> {
    let [rows, k] = weight.shape();
    let bytes = weight.qdata().unwrap();
    let row_bytes = k.div_ceil(2);
    (0..rows * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let byte = bytes[r * row_bytes + c / 2];
            if c % 2 == 0 { byte & 0xF } else { byte >> 4 }
        })
        .collect()
}

#[test]
fn quantized_values_follow_torchaos_formula_exactly() {
    let dev = dev();
    for (n, k, group) in [(1, 32, 32), (3, 128, 128), (5, 130, 128), (2, 256, 64), (7, 896, 128), (4, 4864, 128)] {
        let w = data(n * k, (n + k + group) as u64, -1.5, 1.0);
        let qw = Int4Weight::quantize(&Tensor::from_slice(&dev, &w, &[n, k]).unwrap(), group).unwrap();
        assert_eq!(qw.shape(), [n, k]);
        assert_eq!(qw.group_size(), group);
        let (q, scale, zero) = reference(&w, k, group);
        assert_eq!(levels(&qw), q, "({n}, {k}, group {group}) levels");
        assert_eq!(qw.scale().to_vec().unwrap(), scale, "({n}, {k}, group {group}) scales");
        assert_eq!(qw.zero().to_vec().unwrap(), zero, "({n}, {k}, group {group}) zero points");

        // dequantize is q * scale + min, the same multiply-add on both sides
        let got = qw.dequantize().unwrap().to_vec().unwrap();
        let groups = k.div_ceil(group);
        for r in 0..n {
            for i in 0..k {
                let (s, z) = (scale[r * groups + i / group], zero[r * groups + i / group]);
                let want = (q[r * k + i] as f32).mul_add(s, z - 8.0 * s);
                assert_eq!(got[r * k + i].to_bits(), want.to_bits(), "({n}, {k}) dequantize [{r}][{i}]");
            }
        }
    }
}

#[test]
fn int4_is_half_the_bytes_of_int8() {
    let dev = dev();
    let (n, k) = (896, 4864);
    let w = data(n * k, 3, -1.0, 1.0);
    let t = Tensor::from_slice(&dev, &w, &[n, k]).unwrap();
    let four = Int4Weight::quantize(&t, 128).unwrap();
    let eight = grilly_infer::Int8Weight::quantize(&t).unwrap();
    // int4: k/2 bytes per row plus 8 bytes per group; int8: k plus 4
    assert_eq!(four.nbytes(), n * (k.div_ceil(32) * 16 + 8 * k.div_ceil(128)));
    println!("int4 {} bytes, int8 {} bytes: {:.2}x", four.nbytes(), eight.nbytes(), eight.nbytes() as f64 / four.nbytes() as f64);
    assert!((four.nbytes() as f64) < 0.6 * eight.nbytes() as f64);
}

#[test]
fn linear_matches_a_float64_reference_within_its_bound() {
    let dev = dev();
    let mut worst = 0.0f64;
    for (m, n, k, group) in [(1, 1, 32, 32), (1, 5, 128, 128), (3, 4, 130, 128), (8, 7, 256, 64), (2, 64, 896, 128), (9, 6, 4864, 128)] {
        let w = data(n * k, (m + n + k) as u64, -1.0, 1.0);
        let x = data(m * k, (m * n * k) as u64 + 7, -2.0, 2.0);
        let b = data(n, 99, -0.5, 0.5);
        let qw = Int4Weight::quantize(&Tensor::from_slice(&dev, &w, &[n, k]).unwrap(), group).unwrap();
        let (q, scale, zero) = reference(&w, k, group);
        let groups = k.div_ceil(group);
        // each lane chains its groups' products, then a 64-lane tree
        let terms = k.div_ceil(64) + 6;
        let gamma = (terms + 8) as f64 * 2f64.powi(-24);
        let xt = Tensor::from_slice(&dev, &x, &[m, k]).unwrap();
        for bias in [None, Some(&b)] {
            let bt = bias.map(|b| Tensor::from_slice(&dev, b, &[n]).unwrap());
            let y = qw.linear(&xt, bt.as_ref()).unwrap().to_vec().unwrap();
            for r in 0..m {
                for c in 0..n {
                    let (mut dot, mut mag) = (0.0f64, 0.0f64);
                    for i in 0..k {
                        let (s, z) = (scale[c * groups + i / group] as f64, zero[c * groups + i / group] as f64);
                        let wv = q[c * k + i] as f64 * s + (z - 8.0 * s);
                        let p = x[r * k + i] as f64 * wv;
                        dot += p;
                        mag += p.abs();
                    }
                    let bv = bias.map_or(0.0, |b| b[c] as f64);
                    let want = dot + bv;
                    let bound = gamma * (mag + bv.abs()) + f64::MIN_POSITIVE;
                    let err = (y[r * n + c] as f64 - want).abs();
                    worst = worst.max(err / bound);
                    assert!(err <= bound, "({m}, {n}, {k}, group {group}) y[{r}][{c}]: error {err:e} > bound {bound:e}");
                }
            }
        }
    }
    println!("int4 linear: worst error / bound = {worst:.3}");
}

#[test]
fn refusals() {
    let dev = dev();
    let t = Tensor::from_slice(&dev, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert!(Int4Weight::quantize(&t, 0).is_err(), "group_size 0");
    assert!(Int4Weight::quantize(&t, 12).is_err(), "group_size not a multiple of 32");
    let w = Int4Weight::quantize(&t, 32).unwrap();
    let x = Tensor::from_slice(&dev, &[1.0, 2.0, 3.0], &[1, 3]).unwrap();
    assert!(w.linear(&x, None).is_err(), "k mismatch");
}
