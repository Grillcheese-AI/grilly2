//! grilly_torch::fused: a fused region's gradients against the same ops
//! recorded eagerly (grilly_torch::math).
//!
//! Where every node is used once and nothing broadcasts, the backward graph
//! applies eager's formulas term for term in one kernel, so gradients are
//! bit for bit eager's. With reuse (a node feeding several ops) or
//! broadcasting, eager sums contributions and reduces broadcast dims in a
//! different order than the fused kernel does: those match within rtol 1e-5.

use grilly_core::fusion::Graph;
use grilly_core::generic::{Binary, Unary};
use grilly_core::{Device, Tensor};
use grilly_torch::{fused::fused, math};

fn device() -> Device {
    Device::default_device().expect("open GPU")
}

/// values in [lo, hi], deterministic
fn data(n: usize, seed: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(777);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            lo + (x as f32 / u32::MAX as f32) * (hi - lo)
        })
        .collect()
}

fn leaf(dev: &Device, v: &[f32], shape: &[usize]) -> Tensor {
    let t = Tensor::from_slice(dev, v, shape).unwrap();
    t.set_requires_grad(true).unwrap();
    t
}

fn grads(leaves: &[&Tensor]) -> Vec<Vec<f32>> {
    leaves.iter().map(|l| l.grad().expect("a gradient").to_vec().unwrap()).collect()
}

fn same_bits(a: &[f32], b: &[f32], what: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()), "{what}: element {i}: fused {x:e} vs eager {y:e}");
    }
}

fn close(a: &[f32], b: &[f32], rtol: f32, what: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= rtol * y.abs().max(1.0), "{what}: element {i}: fused {x:e} vs eager {y:e}");
    }
}

#[test]
fn every_unary_rule_matches_eager_bit_for_bit() {
    let dev = device();
    let n = 2000;
    let (xv, gv) = (data(n, 1, 0.3, 2.5), data(n, 2, -1.0, 1.0));
    let ops = [
        (Unary::Neg, 0.0, 0.0), (Unary::Abs, 0.0, 0.0), (Unary::Exp, 0.0, 0.0), (Unary::Log, 0.0, 0.0),
        (Unary::Sqrt, 0.0, 0.0), (Unary::Rsqrt, 0.0, 0.0), (Unary::Tanh, 0.0, 0.0), (Unary::Sigmoid, 0.0, 0.0),
        (Unary::AddScalar, 1.5, 0.0), (Unary::MulScalar, -0.7, 0.0), (Unary::RsubScalar, 2.0, 0.0),
        (Unary::RdivScalar, 3.0, 0.0), (Unary::PowScalar, 2.5, 0.0), (Unary::PowScalar, 3.0, 0.0),
        (Unary::PowScalar, 0.0, 0.0), (Unary::Clamp, 0.8, 1.9), (Unary::Sin, 0.0, 0.0), (Unary::Cos, 0.0, 0.0),
        (Unary::Gelu, 0.0, 0.0), (Unary::Floor, 0.0, 0.0),
    ];
    for (op, a, b) in ops {
        let g = Tensor::from_slice(&dev, &gv, &[n]).unwrap();
        let x = leaf(&dev, &xv, &[n]);
        math::unary(&x, op, a, b).unwrap().backward(Some(&g), false).unwrap();
        let eager = grads(&[&x]);

        let xf = leaf(&dev, &xv, &[n]);
        let mut gr = Graph::new();
        let xi = gr.input();
        let (pa, pb) = (gr.param(), gr.param());
        let y = gr.unary(op, xi, Some(pa), Some(pb));
        gr.output(y);
        fused(&gr, &[&xf], &[a, b], &[n]).unwrap().backward(Some(&g), false).unwrap();
        same_bits(&grads(&[&xf])[0], &eager[0], &format!("{op:?}"));
    }
}

#[test]
fn every_binary_rule_matches_eager_bit_for_bit() {
    let dev = device();
    let n = 2000;
    let (xv, yv, gv) = (data(n, 3, 0.3, 2.5), data(n, 4, 0.3, 2.5), data(n, 5, -1.0, 1.0));
    let mut yv_ties = yv.clone();
    for i in (0..n).step_by(7) {
        yv_ties[i] = xv[i]; // maximum / minimum split ties
    }
    for op in [Binary::Add, Binary::Sub, Binary::Mul, Binary::Div, Binary::Pow, Binary::Maximum, Binary::Minimum] {
        let g = Tensor::from_slice(&dev, &gv, &[n]).unwrap();
        let yvals = if matches!(op, Binary::Maximum | Binary::Minimum) { &yv_ties } else { &yv };
        let (x, y) = (leaf(&dev, &xv, &[n]), leaf(&dev, yvals, &[n]));
        math::binary(&x, &y, op).unwrap().backward(Some(&g), false).unwrap();
        let eager = grads(&[&x, &y]);

        let (xf, yf) = (leaf(&dev, &xv, &[n]), leaf(&dev, yvals, &[n]));
        let mut gr = Graph::new();
        let (xi, yi) = (gr.input(), gr.input());
        let z = gr.binary(op, xi, yi);
        gr.output(z);
        fused(&gr, &[&xf, &yf], &[], &[n]).unwrap().backward(Some(&g), false).unwrap();
        let got = grads(&[&xf, &yf]);
        same_bits(&got[0], &eager[0], &format!("{op:?} dx"));
        same_bits(&got[1], &eager[1], &format!("{op:?} dy"));
    }
}

#[test]
fn chains_with_reuse_and_broadcasting_match_eager() {
    // f = tanh(x * w + b) * x + exp(-x^2), x (4, 3, 5), w (3, 1), b (5,):
    // x is used three times, w and b broadcast
    let dev = device();
    let shape = [4, 3, 5];
    let (xv, wv, bv, gv) = (data(60, 6, -2.0, 2.0), data(3, 7, -1.0, 1.0), data(5, 8, -1.0, 1.0), data(60, 9, -1.0, 1.0));
    let g = Tensor::from_slice(&dev, &gv, &shape).unwrap();

    let (x, w, b) = (leaf(&dev, &xv, &shape), leaf(&dev, &wv, &[3, 1]), leaf(&dev, &bv, &[5]));
    let t = math::unary(&math::binary(&math::binary(&x, &w, Binary::Mul).unwrap(), &b, Binary::Add).unwrap(), Unary::Tanh, 0.0, 0.0).unwrap();
    let sq = math::unary(&math::unary(&x, Unary::PowScalar, 2.0, 0.0).unwrap(), Unary::Neg, 0.0, 0.0).unwrap();
    let f = math::binary(&math::binary(&t, &x, Binary::Mul).unwrap(), &math::unary(&sq, Unary::Exp, 0.0, 0.0).unwrap(), Binary::Add).unwrap();
    let eager_f = f.to_vec().unwrap();
    f.backward(Some(&g), false).unwrap();
    let eager = grads(&[&x, &w, &b]);

    let (xf, wf, bf) = (leaf(&dev, &xv, &shape), leaf(&dev, &wv, &[3, 1]), leaf(&dev, &bv, &[5]));
    let mut gr = Graph::new();
    let (xi, wi, bi) = (gr.input(), gr.input(), gr.input());
    let two = gr.param();
    let xw = gr.binary(Binary::Mul, xi, wi);
    let pre = gr.binary(Binary::Add, xw, bi);
    let t = gr.unary(Unary::Tanh, pre, None, None);
    let sq = gr.unary(Unary::PowScalar, xi, Some(two), None);
    let sq = gr.unary(Unary::Neg, sq, None, None);
    let e = gr.unary(Unary::Exp, sq, None, None);
    let tx = gr.binary(Binary::Mul, t, xi);
    let f = gr.binary(Binary::Add, tx, e);
    gr.output(f);
    let out = fused(&gr, &[&xf, &wf, &bf], &[2.0], &shape).unwrap();
    same_bits(&out.to_vec().unwrap(), &eager_f, "forward");
    out.backward(Some(&g), false).unwrap();
    let got = grads(&[&xf, &wf, &bf]);
    for (i, name) in ["dx", "dw", "db"].iter().enumerate() {
        close(&got[i], &eager[i], 1e-5, name);
    }
}

#[test]
fn inputs_without_a_path_get_zero_gradients_and_frozen_inputs_none() {
    let dev = device();
    let (x, y) = (leaf(&dev, &[1.0, 2.0], &[2]), leaf(&dev, &[3.0, 4.0], &[2]));
    let frozen = Tensor::from_slice(&dev, &[5.0, 6.0], &[2]).unwrap();
    let mut gr = Graph::new();
    let (xi, _yi, fi) = (gr.input(), gr.input(), gr.input());
    let s = gr.unary(Unary::Floor, xi, None, None); // no gradient through floor
    let z = gr.binary(Binary::Add, s, fi);
    gr.output(z);
    let out = fused(&gr, &[&x, &y, &frozen], &[], &[2]).unwrap();
    assert_eq!(out.to_vec().unwrap(), [6.0, 8.0]);
    out.backward(Some(&Tensor::from_slice(&dev, &[1.0, 1.0], &[2]).unwrap()), false).unwrap();
    assert_eq!(x.grad().unwrap().to_vec().unwrap(), [0.0, 0.0]);
    assert_eq!(y.grad().unwrap().to_vec().unwrap(), [0.0, 0.0]);
    assert!(frozen.grad().is_none());
}
