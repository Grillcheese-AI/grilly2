//! GPU sampling from Rust (no Python in the path). The comparison with Hugging
//! Face's processors is tests/python/test_sampling.py.
//!
//! Tolerance: exact. With equal scores every kept token's weight is
//! exp(0) = 1 exactly, so the inverse CDF picks kept token number floor(u * n)
//! in token order, with no rounding to absorb.

use grilly_core::{DType, Device, Tensor};
use grilly_infer::sampling::{multinomial, next_tokens_with, probabilities};
use grilly_infer::Sampling;

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn ints(t: &Tensor) -> Vec<i64> {
    assert_eq!(t.dtype(), DType::I64);
    t.to_bytes().unwrap().chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn sample(top_k: usize, top_p: f32) -> Sampling {
    Sampling { do_sample: true, top_k, top_p, ..Sampling::default() }
}

#[test]
fn the_draw_is_the_inverse_cdf_in_token_order() {
    let dev = dev();
    let n = 40;
    let mut row = vec![0.0f32; n];
    for i in (0..n).step_by(4) {
        row[i] = f32::NEG_INFINITY; // never kept: tokens 0, 4, 8, ...
    }
    let kept: Vec<i64> = (0..n as i64).filter(|i| i % 4 != 0).collect();
    let uniforms = [0.0, 0.2, 0.5, 0.74, 0.999_999];
    let logits = Tensor::from_slice(&dev, &row.repeat(uniforms.len()), &[uniforms.len(), n]).unwrap();
    let got = ints(&next_tokens_with(&logits, None, &sample(0, 1.0), &uniforms).unwrap());
    let want: Vec<i64> = uniforms.iter().map(|u| kept[(u * kept.len() as f32) as usize]).collect();
    assert_eq!(got, want);

    // top_k keeps every token tied with the k-th value: all 30 here
    assert_eq!(ints(&next_tokens_with(&logits, None, &sample(5, 1.0), &uniforms).unwrap()), want);
}

#[test]
fn rows_with_nothing_to_draw_give_minus_one_quickly() {
    // This case once ran a 2^32-step loop on one GPU thread (see g2-sample.glsl).
    let dev = dev();
    let dead = Tensor::from_slice(&dev, &[f32::NEG_INFINITY; 2 * 50], &[2, 50]).unwrap();
    assert_eq!(ints(&next_tokens_with(&dead, None, &sample(0, 1.0), &[0.3, 0.9]).unwrap()), [-1, -1]);
    assert_eq!(ints(&next_tokens_with(&dead, None, &sample(10, 0.5), &[0.3, 0.9]).unwrap()), [-1, -1]);
    let zeros = Tensor::zeros(&dev, &[3, 7]).unwrap();
    assert_eq!(ints(&multinomial(&zeros, &mut Default::default()).unwrap()), [-1, -1, -1]);
    // greedy is torch.argmax: the first index
    assert_eq!(ints(&next_tokens_with(&dead, None, &Sampling::default(), &[]).unwrap()), [0, 0]);
}

#[test]
fn probabilities_sum_to_one_over_the_kept_tokens() {
    let dev = dev();
    let row: Vec<f32> = (0..1000).map(|i| ((i * 37) % 101) as f32 / 10.0).collect();
    let logits = Tensor::from_slice(&dev, &row, &[1, 1000]).unwrap();
    for (k, p) in [(0, 1.0), (50, 1.0), (0, 0.9), (20, 0.8)] {
        let probs = probabilities(&logits, None, &sample(k, p)).unwrap().to_vec().unwrap();
        let total: f64 = probs.iter().map(|&v| v as f64).sum();
        assert!((total - 1.0).abs() < 1e-5, "top_k {k} top_p {p}: sum {total}");
        if k > 0 && p == 1.0 {
            assert!(probs.iter().filter(|&&v| v > 0.0).count() >= k.min(1000));
        }
    }
}
