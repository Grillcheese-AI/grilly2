//! Sparse block-code binding against a reference (crate::block_code).
//!
//! Two tolerances, for two different claims:
//!
//! - **one-hot codes: bit-exact.** Each output sum has exactly one non-zero
//!   term, 1.0, so the result is an exact one-hot whatever order the zeros
//!   are added in. This is what v1's `blockcode_bind` produced by finding
//!   the hot positions, so the port is `==` against that rule, not near it.
//!   CLAUDE.md §8.2: the inputs are (n, k, l) batches of ONE-HOT vectors,
//!   and dense random ones are a different test, below.
//! - **continuous codes: a float64 reference.** `|got - ref| <= 4 * l * eps
//!   * max|a| * max|b|`, l products summed in float32, and the worst
//!   measured ratio to that bound is printed. Case (b) of CLAUDE.md §9.

use grilly_core::{Device, Tensor};
use grilly_vsa::{bind, unbind};

const K: usize = 80;
const L: usize = 128;

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` one-hot codes, and the hot position of every block.
fn one_hot(n: usize, seed: u64) -> (Vec<f32>, Vec<usize>) {
    let mut next = rng(seed);
    let mut data = vec![0.0f32; n * K * L];
    let mut hot = Vec::with_capacity(n * K);
    for v in 0..n {
        for b in 0..K {
            let j = (next() * L as f64) as usize % L;
            data[(v * K + b) * L + j] = 1.0;
            hot.push(j);
        }
    }
    (data, hot)
}

fn dense(n: usize, seed: u64) -> Vec<f32> {
    let mut next = rng(seed);
    (0..n * K * L).map(|_| (next() * 2.0 - 1.0) as f32).collect()
}

/// The same convolution in float64, straight from the definition.
fn reference(a: &[f32], b: &[f32], n: usize, unbind: bool) -> Vec<f64> {
    let mut out = vec![0.0f64; n * K * L];
    for v in 0..n {
        for blk in 0..K {
            let base = (v * K + blk) * L;
            for i in 0..L {
                let mut s = 0.0f64;
                for j in 0..L {
                    s += if unbind {
                        a[base + (i + j) % L] as f64 * b[base + j] as f64
                    } else {
                        a[base + j] as f64 * b[base + (i + L - j) % L] as f64
                    };
                }
                out[base + i] = s;
            }
        }
    }
    out
}

#[test]
fn binding_one_hot_codes_is_exact_and_lands_where_v1_put_it() {
    let dev = dev();
    let n = 8;
    let (a, ha) = one_hot(n, 1);
    let (b, hb) = one_hot(n, 2);
    let ta = Tensor::from_slice(&dev, &a, &[n, K, L]).unwrap();
    let tb = Tensor::from_slice(&dev, &b, &[n, K, L]).unwrap();
    let got = bind(&ta, &tb).unwrap().to_vec().unwrap();

    // v1's rule: one-hot at (hot_a + hot_b) % l, exactly 1.0 and 0.0
    let mut want = vec![0.0f32; n * K * L];
    for (blk, (&ia, &ib)) in ha.iter().zip(&hb).enumerate() {
        want[blk * L + (ia + ib) % L] = 1.0;
    }
    assert_eq!(got, want, "one-hot binding is not exact");
}

#[test]
fn unbinding_recovers_a_one_hot_code_exactly() {
    let dev = dev();
    let n = 16;
    let (x, _) = one_hot(n, 3);
    let (key, _) = one_hot(n, 4);
    let tx = Tensor::from_slice(&dev, &x, &[n, K, L]).unwrap();
    let tk = Tensor::from_slice(&dev, &key, &[n, K, L]).unwrap();
    let composite = bind(&tx, &tk).unwrap();
    assert_eq!(unbind(&composite, &tk).unwrap().to_vec().unwrap(), x, "bind then unbind lost the code");
}

#[test]
fn continuous_codes_match_a_float64_convolution() {
    let dev = dev();
    let n = 4;
    for (unb, seed) in [(false, 5u64), (true, 7)] {
        let (a, b) = (dense(n, seed), dense(n, seed + 100));
        let ta = Tensor::from_slice(&dev, &a, &[n, K, L]).unwrap();
        let tb = Tensor::from_slice(&dev, &b, &[n, K, L]).unwrap();
        let got = if unb { unbind(&ta, &tb) } else { bind(&ta, &tb) }.unwrap().to_vec().unwrap();
        let want = reference(&a, &b, n, unb);

        let (amax, bmax) = (1.0f64, 1.0f64);  // dense() is bounded by 1
        let bound = 4.0 * L as f64 * f32::EPSILON as f64 * amax * bmax;
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs() / bound)
            .fold(0.0f64, f64::max);
        println!("{}: worst error / bound = {worst:.3}", if unb { "unbind" } else { "bind" });
        assert!(worst <= 1.0, "worst error is {worst:.3} of the bound");
    }
}

#[test]
fn one_code_binds_against_a_whole_codebook() {
    let dev = dev();
    let n = 32;
    let (cb, hc) = one_hot(n, 11);
    let (key, hk) = one_hot(1, 12);
    let tcb = Tensor::from_slice(&dev, &cb, &[n, K, L]).unwrap();
    let tkey = Tensor::from_slice(&dev, &key, &[K, L]).unwrap();
    let got = bind(&tcb, &tkey).unwrap();
    assert_eq!(got.shape(), &[n, K, L], "the batch shape is the wide operand's");

    let mut want = vec![0.0f32; n * K * L];
    for v in 0..n {
        for blk in 0..K {
            want[(v * K + blk) * L + (hc[v * K + blk] + hk[blk]) % L] = 1.0;
        }
    }
    assert_eq!(got.to_vec().unwrap(), want, "broadcasting a key over a codebook");
}

#[test]
fn mismatched_block_shapes_are_refused() {
    let dev = dev();
    let a = Tensor::from_slice(&dev, &vec![0.0f32; 4 * 8], &[4, 8]).unwrap();
    let b = Tensor::from_slice(&dev, &vec![0.0f32; 4 * 16], &[4, 16]).unwrap();
    let e = bind(&a, &b).unwrap_err();
    assert!(format!("{e}").contains("block shapes differ"), "{e}");
}
