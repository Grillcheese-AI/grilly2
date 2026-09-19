//! The GPU KV cache from Rust (no Python in the path): appends, then
//! decode-step attention against a float64 reference over the filled
//! positions, with grouped-query attention.
//!
//! Tolerance: |got - ref| <= 1e-5 * |ref| + 1e-6 * max|v|. That is case (b)
//! in CLAUDE.md §9, a float32 kernel against float64. Each score is a
//! float32 dot product of head_dim terms, the softmax uses a compensated
//! sum, and the output is a compensated weighted sum, so the error is a
//! few ulp of the terms involved. The worst ratio to the bound is printed.

use grilly_core::{Device, Tensor};
use grilly_infer::KvCache;

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

/// (batch, heads, 1, dim) attention over the first `len` positions of the
/// host copies `k`, `v` (batch, kv_heads, len, dim), in float64.
#[allow(clippy::too_many_arguments)]
fn reference(q: &[f32], k: &[f32], v: &[f32], batch: usize, heads: usize, kv_heads: usize, len: usize, dim: usize) -> Vec<f64> {
    let scale = 1.0 / (dim as f64).sqrt();
    let mut out = vec![0.0; batch * heads * dim];
    for b in 0..batch {
        for h in 0..heads {
            let g = h / (heads / kv_heads);
            let qr = &q[(b * heads + h) * dim..][..dim];
            let s: Vec<f64> = (0..len)
                .map(|j| {
                    let kr = &k[((b * kv_heads + g) * len + j) * dim..][..dim];
                    qr.iter().zip(kr).map(|(a, c)| *a as f64 * *c as f64).sum::<f64>() * scale
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..dim {
                let o: f64 = (0..len).map(|j| e[j] * v[((b * kv_heads + g) * len + j) * dim + d] as f64).sum();
                out[(b * heads + h) * dim + d] = o / z;
            }
        }
    }
    out
}

#[test]
fn appends_then_decode_attention_match_float64() {
    let dev = dev();
    let mut worst = 0.0f64;
    for (batch, heads, kv_heads, dim) in [(2, 6, 2, 64), (1, 14, 2, 64), (1, 4, 4, 128), (2, 3, 1, 80), (1, 2, 1, 1)] {
        let (layers, max_len) = (2, 40);
        let mut cache = KvCache::new(&dev, layers, batch, kv_heads, dim, max_len).unwrap();
        // host copies, per layer, as (batch, kv_heads, len, dim) grown step by step
        let mut hk: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); batch * kv_heads]; layers];
        let mut hv = hk.clone();
        let mut seed = 1u64;
        for step in [5usize, 1, 1, 3, 1] {
            for layer in 0..layers {
                let n = batch * kv_heads * step * dim;
                let (kn, vn) = (data(n, seed), data(n, seed + 1));
                seed += 2;
                let shape = [batch, kv_heads, step, dim];
                cache
                    .append(layer, &Tensor::from_slice(&dev, &kn, &shape).unwrap(), &Tensor::from_slice(&dev, &vn, &shape).unwrap())
                    .unwrap();
                for bh in 0..batch * kv_heads {
                    hk[layer][bh].extend_from_slice(&kn[bh * step * dim..][..step * dim]);
                    hv[layer][bh].extend_from_slice(&vn[bh * step * dim..][..step * dim]);
                }
                let len = cache.len(layer);
                assert_eq!(len, hk[layer][0].len() / dim);
                let q = data(batch * heads * dim, seed);
                seed += 1;
                let got = cache.attend(layer, &Tensor::from_slice(&dev, &q, &[batch, heads, 1, dim]).unwrap(), None).unwrap();
                assert_eq!(got.shape(), [batch, heads, 1, dim]);
                let got = got.to_vec().unwrap();
                let (fk, fv): (Vec<f32>, Vec<f32>) = (hk[layer].concat(), hv[layer].concat());
                let want = reference(&q, &fk, &fv, batch, heads, kv_heads, len, dim);
                let vmax = fv.iter().fold(0.0f64, |m, x| m.max(x.abs() as f64));
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    let bound = 1e-5 * w.abs() + 1e-6 * vmax;
                    let err = (*g as f64 - w).abs();
                    worst = worst.max(err / bound);
                    assert!(err <= bound, "({batch}, {heads}, {kv_heads}, {dim}) layer {layer} len {len} [{i}]: {g} vs {w}");
                }
            }
        }
        assert_eq!(cache.nbytes(), 2 * layers * batch * kv_heads * max_len * dim * 4);
    }
    println!("kv cache attention: worst error / bound = {worst:.3}");
}

#[test]
fn refusals() {
    let dev = dev();
    let mut cache = KvCache::new(&dev, 1, 1, 2, 8, 4).unwrap();
    let q = Tensor::zeros(&dev, &[1, 4, 1, 8]).unwrap();
    assert!(cache.attend(0, &q, None).is_err(), "an empty layer");
    let kv = |t| Tensor::zeros(&dev, &[1, 2, t, 8]).unwrap();
    cache.append(0, &kv(3), &kv(3)).unwrap();
    assert!(cache.append(0, &kv(2), &kv(2)).is_err(), "past max_len");
    assert!(cache.append(0, &Tensor::zeros(&dev, &[1, 3, 1, 8]).unwrap(), &kv(1)).is_err(), "wrong kv_heads");
    assert!(cache.append(1, &kv(1), &kv(1)).is_err(), "no such layer");
    assert!(cache.attend(0, &Tensor::zeros(&dev, &[1, 3, 1, 8]).unwrap(), None).is_err(), "heads not a multiple");
    assert!(KvCache::new(&dev, 1, 1, 1, 257, 4).is_err(), "head_dim over the kernel's limit");
    cache.reset();
    assert!(cache.is_empty());
    cache.append(0, &kv(4), &kv(4)).unwrap();
    assert_eq!(cache.len(0), 4);
}
