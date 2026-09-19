//! The decoder step's fused kernels, from Rust.
//!
//! Tolerance: bit-exact where the kernel does the same arithmetic as the ops
//! it replaces (CLAUDE.md §9 case (a)).
//! - rotary: `x * cos + rotate_half(x) * sin` is two multiplies and an add,
//!   in that order, `precise` in the shader, so it equals the same float32
//!   arithmetic on the host exactly;
//! - a fused int8 linear equals the separate ones exactly: the scales are
//!   per row, and each output row's dot product is unchanged;
//! - keys rotated into the cache equal an append of the rotated keys.
//!
//! swiglu's silu is compared against float64 here (its sigmoid is a shader
//! function); tests/python/test_fused_ops.py checks it against grilly's own
//! silu bit for bit.

use grilly_core::{Device, Tensor};
use grilly_infer::{rope, swiglu, Heads, Int8Weight, KvCache, Rotary};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

/// x * cos + rotate_half(x) * sin, in float32, in the kernel's order.
#[allow(clippy::too_many_arguments)]
fn rope_reference(src: &[f32], b: usize, t: usize, row: usize, col: usize, heads: usize, dim: usize, cos: &[f32], sin: &[f32], pos0: usize) -> Vec<f32> {
    let mut out = vec![0.0; b * heads * t * dim];
    for bi in 0..b {
        for h in 0..heads {
            for ti in 0..t {
                for d in 0..dim {
                    let base = (bi * t + ti) * row + col + h * dim;
                    let x = src[base + d];
                    let r = if d < dim / 2 { -src[base + d + dim / 2] } else { src[base + d - dim / 2] };
                    let table = (pos0 + ti) * dim + d;
                    out[((bi * heads + h) * t + ti) * dim + d] = x * cos[table] + r * sin[table];
                }
            }
        }
    }
    out
}

fn tables(positions: usize, dim: usize, dev: &Device) -> (Vec<f32>, Vec<f32>, Tensor, Tensor) {
    let angles: Vec<f32> = (0..positions * dim).map(|i| (i as f32) * 0.013).collect();
    let (c, s): (Vec<f32>, Vec<f32>) = angles.iter().map(|a| (a.cos(), a.sin())).unzip();
    let ct = Tensor::from_slice(dev, &c, &[positions, dim]).unwrap();
    let st = Tensor::from_slice(dev, &s, &[positions, dim]).unwrap();
    (c, s, ct, st)
}

#[test]
fn rotary_is_the_same_arithmetic_bit_for_bit() {
    let dev = dev();
    for (b, t, heads, dim, col, extra, pos0) in
        [(1, 1, 14, 64, 0, 128 + 128, 0), (2, 5, 4, 32, 0, 0, 3), (1, 1, 2, 64, 896, 128, 17), (3, 2, 2, 8, 16, 8, 1)]
    {
        let row = col + heads * dim + extra;
        let src = data(b * t * row, (b + t + row) as u64);
        let (c, s, ct, st) = tables(64, dim, &dev);
        let x = Tensor::from_slice(&dev, &src, &[b, t, row]).unwrap();
        let got = rope(&Heads { src: &x, col, heads, dim }, &Rotary { cos: &ct, sin: &st, pos0 }).unwrap();
        assert_eq!(got.shape(), [b, heads, t, dim]);
        let want = rope_reference(&src, b, t, row, col, heads, dim, &c, &s, pos0);
        let got: Vec<u32> = got.to_vec().unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(), "({b}, {t}, {heads}, {dim}, col {col})");
    }
}

#[test]
fn keys_rotated_into_the_cache_equal_an_append_of_them() {
    let dev = dev();
    let (b, t, kv, d, heads) = (2, 3, 2, 16, 4);
    let row = heads * d + 2 * kv * d;
    let (k_col, v_col) = (heads * d, heads * d + kv * d);
    let src = data(b * t * row, 9);
    let x = Tensor::from_slice(&dev, &src, &[b, t, row]).unwrap();
    let (_, _, ct, st) = tables(32, d, &dev);
    let rot = Rotary { cos: &ct, sin: &st, pos0: 4 };

    let mut fused = KvCache::new(&dev, 1, b, kv, d, 16).unwrap();
    fused
        .append_rotary(0, &Heads { src: &x, col: k_col, heads: kv, dim: d }, &Heads { src: &x, col: v_col, heads: kv, dim: d }, &rot)
        .unwrap();

    // the same, in steps: rotate the keys, gather the values, append both
    let keys = rope(&Heads { src: &x, col: k_col, heads: kv, dim: d }, &rot).unwrap();
    let mut values = vec![0.0f32; b * kv * t * d];
    for bi in 0..b {
        for h in 0..kv {
            for ti in 0..t {
                for di in 0..d {
                    values[((bi * kv + h) * t + ti) * d + di] = src[(bi * t + ti) * row + v_col + h * d + di];
                }
            }
        }
    }
    let mut plain = KvCache::new(&dev, 1, b, kv, d, 16).unwrap();
    plain.append(0, &keys, &Tensor::from_slice(&dev, &values, &[b, kv, t, d]).unwrap()).unwrap();

    assert_eq!(fused.len(0), t);
    let q = Tensor::from_slice(&dev, &data(b * heads * d, 11), &[b, heads, 1, d]).unwrap();
    let (a, c) = (fused.attend(0, &q, None).unwrap().to_vec().unwrap(), plain.attend(0, &q, None).unwrap().to_vec().unwrap());
    assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<u32>>(), c.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
}

#[test]
fn a_fused_int8_linear_equals_the_separate_ones() {
    let dev = dev();
    let (k, m) = (128, 3);
    let parts: Vec<Int8Weight> = [7usize, 2, 2]
        .iter()
        .enumerate()
        .map(|(i, &n)| Int8Weight::quantize(&Tensor::from_slice(&dev, &data(n * k, i as u64 + 1), &[n, k]).unwrap()).unwrap())
        .collect();
    let fused = Int8Weight::cat(&parts.iter().collect::<Vec<_>>()).unwrap();
    assert_eq!(fused.shape(), [11, k]);
    let x = Tensor::from_slice(&dev, &data(m * k, 99), &[m, k]).unwrap();
    let got = fused.linear(&x, None).unwrap().to_vec().unwrap();
    let mut col = 0;
    for p in &parts {
        let want = p.linear(&x, None).unwrap().to_vec().unwrap();
        for r in 0..m {
            for c in 0..p.shape()[0] {
                assert_eq!(got[r * 11 + col + c].to_bits(), want[r * p.shape()[0] + c].to_bits(), "row {r} column {}", col + c);
            }
        }
        col += p.shape()[0];
    }
    assert_eq!(fused.qdata().unwrap().len(), 11 * k);
}

#[test]
fn swiglu_matches_float64() {
    let dev = dev();
    let (rows, width) = (4, 96);
    let g = data(rows * 2 * width, 3);
    let src = Tensor::from_slice(&dev, &g, &[rows, 2 * width]).unwrap();
    let got = swiglu(&src, 0, &src, width, width).unwrap();
    assert_eq!(got.shape(), [rows, width]);
    let got = got.to_vec().unwrap();
    for r in 0..rows {
        for j in 0..width {
            let (a, b) = (g[r * 2 * width + j] as f64, g[r * 2 * width + width + j] as f64);
            let want = a / (1.0 + (-a).exp()) * b;
            assert!((got[r * width + j] as f64 - want).abs() <= 1e-6 * (1.0 + want.abs()), "[{r}][{j}]");
        }
    }
}
