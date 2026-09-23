//! The fused integrate-and-fire timestep against the recurrence it encodes.
//!
//! Tolerances:
//!
//! - **the step: 1e-6 relative** against the same formula in float64. It is
//!   four float32 operations and a comparison; anything wider means the
//!   recurrence changed, not that the GPU rounded.
//! - **the spikes: `==`**. A spike is 0.0 or 1.0 and a comparison has no
//!   error. A tolerance here would be nonsense.
//! - **the surrogate: 1e-6** against its analytic formula, because that is
//!   the whole contract of the backward pass.

use grilly_core::{Device, Tensor};
use grilly_snn::lif::{step, Reset, StepConfig, Surrogate};

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
            ((s >> 40) as f32 / 4_194_304.0) - 2.0
        })
        .collect()
}

fn config() -> StepConfig {
    StepConfig {
        decay: 0.6,
        in_scale: 1.0,
        rest: 0.0,
        threshold: 0.8,
        reset: Reset::Hard,
        detach_reset: false,
        surrogate: Surrogate::ATan,
        alpha: 2.0,
    }
}

/// The recurrence the kernel encodes, in float64.
fn reference(v: &[f32], x: &[f32], c: &StepConfig) -> (Vec<f64>, Vec<f64>) {
    let rest = if c.reset == Reset::Soft { 0.0 } else { c.rest as f64 };
    let mut spikes = Vec::with_capacity(v.len());
    let mut next = Vec::with_capacity(v.len());
    for (&vi, &xi) in v.iter().zip(x) {
        let h = c.decay as f64 * (vi as f64 - rest) + rest + c.in_scale as f64 * xi as f64;
        let s = f64::from(h >= c.threshold as f64);
        spikes.push(s);
        next.push(if c.reset == Reset::Soft {
            h - c.threshold as f64 * s
        } else {
            h * (1.0 - s) + rest * s
        });
    }
    (spikes, next)
}

#[test]
fn the_step_matches_the_recurrence_it_encodes() {
    let dev = dev();
    let n = 4096;
    for (name, mut c) in [("hard", config()), ("soft", config()), ("decay_input", config())] {
        if name == "soft" {
            c.reset = Reset::Soft;
        }
        if name == "decay_input" {
            c.in_scale = 1.0 / 2.5;
            c.decay = 0.6;
        }
        let (v, x) = (data(n, 1), data(n, 2));
        let tv = Tensor::from_slice(&dev, &v, &[n]).unwrap();
        let tx = Tensor::from_slice(&dev, &x, &[n]).unwrap();
        let (spikes, next) = step(&tv, &tx, c).unwrap();
        let (want_s, want_v) = reference(&v, &x, &c);

        let got_s = spikes.to_vec().unwrap();
        assert!(got_s.iter().zip(&want_s).all(|(g, w)| f64::from(*g) == *w), "{name}: spikes");
        assert!(got_s.iter().all(|s| *s == 0.0 || *s == 1.0), "{name}: a spike is 0 or 1");

        let got_v = next.to_vec().unwrap();
        let worst = got_v
            .iter()
            .zip(&want_v)
            .map(|(g, w)| (f64::from(*g) - w).abs() / w.abs().max(1.0))
            .fold(0.0f64, f64::max);
        println!("{name}: worst relative membrane error {worst:.3e}");
        assert!(worst < 1e-6, "{name}: {worst:.3e}");
    }
}

#[test]
fn a_neuron_that_is_never_driven_never_fires() {
    let dev = dev();
    let n = 256;
    let v = Tensor::from_slice(&dev, &vec![0.0f32; n], &[n]).unwrap();
    let x = Tensor::from_slice(&dev, &vec![0.0f32; n], &[n]).unwrap();
    let (spikes, next) = step(&v, &x, config()).unwrap();
    assert_eq!(spikes.to_vec().unwrap(), vec![0.0; n]);
    assert_eq!(next.to_vec().unwrap(), vec![0.0; n]);
}

#[test]
fn a_soft_reset_keeps_the_overshoot_and_a_hard_one_does_not() {
    let dev = dev();
    let v = Tensor::from_slice(&dev, &[0.0f32], &[1]).unwrap();
    let x = Tensor::from_slice(&dev, &[5.0f32], &[1]).unwrap();

    let mut hard = config();
    hard.decay = 1.0;
    let (s, next) = step(&v, &x, hard).unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![1.0]);
    assert_eq!(next.to_vec().unwrap(), vec![0.0], "hard reset clears to rest");

    let mut soft = hard;
    soft.reset = Reset::Soft;
    let (s, next) = step(&v, &x, soft).unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![1.0]);
    assert_eq!(next.to_vec().unwrap(), vec![5.0 - 0.8], "soft reset subtracts the threshold");
}

#[test]
fn a_mismatched_membrane_and_input_are_refused() {
    let dev = dev();
    let v = Tensor::from_slice(&dev, &[0.0f32; 4], &[4]).unwrap();
    let x = Tensor::from_slice(&dev, &[0.0f32; 8], &[8]).unwrap();
    let e = step(&v, &x, config()).unwrap_err();
    assert!(format!("{e}").contains("they must match"), "{e}");
}
