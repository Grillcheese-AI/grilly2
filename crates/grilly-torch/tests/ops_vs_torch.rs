//! Each op against torch on CPU. Tolerances:
//!
//! | op | atol | rtol | why |
//! |---|---|---|---|
//! | relu | 0 | 0 | exact in float32 |
//! | add | 0 | 0 | one IEEE add per element, same as torch |
//!
//! linear is a length-K reduction, so its tolerance scales with K instead:
//! `|grilly - torch| <= sqrt(K) * eps_f32 * sum_k |x_k| |w_k|` (+ |bias|).
//! Measured against a float64 reference, grilly's sequential per-thread
//! accumulation errs 4-7x more than torch's GEMM at K = 1024..4096, but at
//! most ~2.6e-7 of sum|x||w| -- about a tenth of this bound.

mod common;

use common::{assert_close, assert_within, data, torch};
use grilly_core::{Device, Tensor};
use grilly_torch::{add, kernels, linear, relu};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(&dev(), data, shape).unwrap()
}

#[test]
fn every_kernel_is_present() {
    for k in kernels::ALL {
        assert!(dev().has_kernel(k.shader).unwrap(), "missing kernel {}", k.shader);
    }
}

#[test]
fn relu_matches_torch() {
    for (seed, shape) in [(1, vec![1000]), (2, vec![3, 7, 11]), (3, vec![0]), (4, vec![70_000])] {
        let n = shape.iter().product();
        let x = data(n, seed);
        let got = relu(&t(&x, &shape)).unwrap();
        assert_eq!(got.shape(), &shape[..]);
        let want = torch("torch.relu(x)", &[("x", &x, &shape)]);
        assert_close(&got.to_vec().unwrap(), &want, 0.0, 0.0, &format!("relu {shape:?}"));
    }
}

#[test]
fn add_matches_torch() {
    for (seed, shape) in [(5, vec![1000]), (6, vec![4, 9, 13]), (7, vec![0, 3])] {
        let n = shape.iter().product();
        let (a, b) = (data(n, seed), data(n, seed + 100));
        let got = add(&t(&a, &shape), &t(&b, &shape)).unwrap();
        let want = torch("torch.add(a, b)", &[("a", &a, &shape), ("b", &b, &shape)]);
        assert_close(&got.to_vec().unwrap(), &want, 0.0, 0.0, &format!("add {shape:?}"));
    }
}

#[test]
fn add_rejects_mismatched_shapes() {
    let err = add(&t(&[0.0; 6], &[2, 3]), &t(&[0.0; 6], &[3, 2])).unwrap_err();
    assert!(err.to_string().contains("[2, 3]"), "{err}");
}

#[test]
fn linear_matches_torch() {
    // (input shape, out_features, bias)
    let cases: &[(&[usize], usize, bool)] = &[
        (&[37, 70], 23, true),     // no dimension a multiple of the 16x16 tile
        (&[2, 5, 33], 17, false),  // 3-D input, no bias
        (&[64], 10, true),         // 1-D input
        (&[256, 1024], 512, true), // long reduction
        (&[64, 4096], 64, false),  // longer reduction
        (&[5, 0], 4, true),        // empty reduction: bias broadcast
        (&[0, 8], 3, true),        // no rows
    ];
    for (i, &(in_shape, out_f, has_bias)) in cases.iter().enumerate() {
        let in_f = *in_shape.last().unwrap();
        let x = data(in_shape.iter().product(), 10 + i as u32);
        let w = data(out_f * in_f, 20 + i as u32);
        let b = data(out_f, 30 + i as u32);
        let w_shape = [out_f, in_f];

        let bias_t = t(&b, &[out_f]);
        let got = linear(&t(&x, in_shape), &t(&w, &w_shape), has_bias.then_some(&bias_t)).unwrap();

        let mut inputs = vec![("x", &x[..], in_shape), ("w", &w[..], &w_shape[..])];
        let expr = if has_bias {
            inputs.push(("b", &b[..], std::slice::from_ref(&out_f)));
            "F.linear(x, w, b)"
        } else {
            "F.linear(x, w)"
        };
        let want = torch(expr, &inputs);
        let mag_expr = if has_bias { "F.linear(x.abs(), w.abs(), b.abs())" } else { "F.linear(x.abs(), w.abs())" };
        let scale = (in_f as f32).sqrt() * f32::EPSILON;
        let bound: Vec<f32> = torch(mag_expr, &inputs).iter().map(|m| m * scale).collect();

        let mut want_shape = in_shape.to_vec();
        *want_shape.last_mut().unwrap() = out_f;
        assert_eq!(got.shape(), &want_shape[..], "case {i}");
        let worst = assert_within(&got.to_vec().unwrap(), &want, &bound, &format!("linear case {i} {in_shape:?}->{out_f}"));
        println!("linear {in_shape:?} -> {out_f} bias={has_bias}: worst |diff| / bound = {worst:.3}");
    }
}

#[test]
fn linear_rejects_bad_shapes() {
    let x = t(&[0.0; 6], &[2, 3]);
    let err = linear(&x, &t(&[0.0; 8], &[2, 4]), None).unwrap_err();
    assert!(err.to_string().contains("3 features"), "{err}");
    let err = linear(&x, &t(&[0.0; 6], &[2, 3]), Some(&t(&[0.0; 3], &[3]))).unwrap_err();
    assert!(err.to_string().contains("bias"), "{err}");
}

#[test]
fn ops_chain_without_host_round_trips() {
    // linear -> relu -> add, all recorded before one readback.
    let (x, w, b) = (data(8 * 16, 40), data(12 * 16, 41), data(12, 42));
    let xt = t(&x, &[8, 16]);
    let h = linear(&xt, &t(&w, &[12, 16]), Some(&t(&b, &[12]))).unwrap();
    let got = add(&relu(&h).unwrap(), &h).unwrap();
    let want = torch(
        "torch.relu(F.linear(x, w, b)) + F.linear(x, w, b)",
        &[("x", &x, &[8, 16]), ("w", &w, &[12, 16]), ("b", &b, &[12])],
    );
    // relu(h) + h stays within 2x the linear bound (K = 16).
    let mag = torch("2 * F.linear(x.abs(), w.abs(), b.abs())", &[("x", &x, &[8, 16]), ("w", &w, &[12, 16]), ("b", &b, &[12])]);
    let bound: Vec<f32> = mag.iter().map(|m| m * 4.0 * f32::EPSILON).collect();
    assert_within(&got.to_vec().unwrap(), &want, &bound, "chain");
}
