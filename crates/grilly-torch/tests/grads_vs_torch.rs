//! Gradients against torch.autograd on CPU. Tolerances:
//!
//! - relu: exact (`grad * (x > 0)`, the same selection torch makes).
//! - linear: each gradient is a reduction, bounded like the forward pass by
//!   `sqrt(K) * eps_f32 * (the same reduction over absolute values)`, with
//!   K = out_features for grad_input and K = rows for grad_weight/grad_bias.

mod common;

use common::{assert_close, assert_within, data, torch_many};
use grilly_core::{Device, Tensor};
use grilly_torch::{add, linear, relu};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(&dev(), data, shape).unwrap()
}

fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor {
    let x = t(data, shape);
    x.set_requires_grad(requires_grad).unwrap();
    x
}

#[test]
fn relu_grad_is_exact() {
    for (seed, shape) in [(1, vec![1000]), (2, vec![3, 7, 11])] {
        let n: usize = shape.iter().product();
        let (x, g) = (data(n, seed), data(n, seed + 50));
        let xt = leaf(&x, &shape, true);
        relu(&xt).unwrap().backward(Some(&t(&g, &shape)), false).unwrap();
        let want = torch_many("grads(torch.relu, (x,), g)", &[("x", &x, &shape), ("g", &g, &shape)]);
        assert_close(&xt.grad().unwrap().to_vec().unwrap(), &want[0], 0.0, 0.0, "relu grad");
    }
}

#[test]
fn linear_grads_within_bound() {
    // (input shape, out_features, bias, which of x/w/b require grad)
    let cases: &[(&[usize], usize, bool, [bool; 3])] = &[
        (&[37, 70], 23, true, [true, true, true]),
        (&[2, 5, 33], 17, false, [true, true, false]),
        (&[64], 10, true, [true, true, true]),
        (&[256, 1024], 512, true, [true, true, true]),
        (&[40, 30], 20, true, [false, true, true]), // frozen input
        (&[40, 30], 20, true, [true, false, false]), // frozen layer
        (&[5, 0], 4, true, [true, true, true]),  // empty reduction
        (&[0, 8], 3, true, [true, true, true]),  // no rows: zero weight/bias grads
    ];
    for (i, &(in_shape, out_f, has_bias, rg)) in cases.iter().enumerate() {
        let in_f = *in_shape.last().unwrap();
        let rows: usize = in_shape[..in_shape.len() - 1].iter().product();
        let mut out_shape = in_shape.to_vec();
        *out_shape.last_mut().unwrap() = out_f;
        let (x, w, b) = (data(rows * in_f, 100 + i as u32), data(out_f * in_f, 200 + i as u32), data(out_f, 300 + i as u32));
        let g = data(rows * out_f, 400 + i as u32);

        let (xt, wt, bt) = (leaf(&x, in_shape, rg[0]), leaf(&w, &[out_f, in_f], rg[1]), leaf(&b, &[out_f], rg[2]));
        let y = linear(&xt, &wt, has_bias.then_some(&bt)).unwrap();
        y.backward(Some(&t(&g, &out_shape)), false).unwrap();

        let w_shape = [out_f, in_f];
        let inputs = [("x", &x[..], in_shape), ("w", &w[..], &w_shape[..]), ("b", &b[..], std::slice::from_ref(&out_f)), ("g", &g[..], &out_shape[..])];
        let want = torch_many(
            if has_bias { "grads(F.linear, (x, w, b), g)" } else { "grads(F.linear, (x, w), g)" },
            &inputs,
        );
        let mag = torch_many(
            &format!(
                "(g.abs().reshape({rows}, {out_f}) @ w.abs(), g.abs().reshape({rows}, {out_f}).T @ x.abs().reshape({rows}, {in_f}), g.abs().reshape({rows}, {out_f}).sum(0))"
            ),
            &inputs,
        );
        let eps = f32::EPSILON;
        let bound = |m: &[f32], k: usize| -> Vec<f32> { m.iter().map(|v| v * (k as f32).sqrt() * eps).collect() };

        let what = |name: &str| format!("case {i} {in_shape:?}->{out_f} {name}");
        let check = |got: &Tensor, idx: usize, k: usize, name: &str| {
            let worst = assert_within(&got.to_vec().unwrap(), &want[idx], &bound(&mag[idx], k), &what(name));
            println!("{}: worst |diff| / bound = {worst:.3}", what(name));
        };
        if rg[0] { check(&xt.grad().expect("x.grad"), 0, out_f, "grad_input") } else { assert!(xt.grad().is_none()) }
        if rg[1] { check(&wt.grad().expect("w.grad"), 1, rows, "grad_weight") } else { assert!(wt.grad().is_none()) }
        if has_bias && rg[2] { check(&bt.grad().expect("b.grad"), 2, rows, "grad_bias") }
        if !(has_bias && rg[2]) { assert!(bt.grad().is_none()) }
    }
}

#[test]
fn chain_grads_match_torch() {
    // y = relu(x W1^T + b1) + (x W1^T + b1); loss grad g
    let (x, w, b, g) = (data(8 * 16, 1), data(12 * 16, 2), data(12, 3), data(8 * 12, 4));
    let (xt, wt, bt) = (leaf(&x, &[8, 16], true), leaf(&w, &[12, 16], true), leaf(&b, &[12], true));
    let h = linear(&xt, &wt, Some(&bt)).unwrap();
    add(&relu(&h).unwrap(), &h).unwrap().backward(Some(&t(&g, &[8, 12])), false).unwrap();

    let inputs = [("x", &x[..], &[8, 16][..]), ("w", &w[..], &[12, 16][..]), ("b", &b[..], &[12][..]), ("g", &g[..], &[8, 12][..])];
    let want = torch_many("grads(lambda x, w, b: torch.relu(F.linear(x, w, b)) + F.linear(x, w, b), (x, w, b), g)", &inputs);
    // Two paths each within the linear bound; 2x covers their sum.
    let mag = torch_many("(2 * g.abs() @ w.abs(), 2 * g.abs().T @ x.abs(), 2 * g.abs().sum(0))", &inputs);
    for (idx, (got, k)) in [(&xt, 12usize), (&wt, 8), (&bt, 8)].into_iter().enumerate() {
        let bound: Vec<f32> = mag[idx].iter().map(|m| m * (k as f32).sqrt() * f32::EPSILON).collect();
        assert_within(&got.grad().unwrap().to_vec().unwrap(), &want[idx], &bound, &format!("chain grad {idx}"));
    }
}

#[test]
fn second_backward_needs_retain_graph() {
    let x = leaf(&[-1.0, 2.0], &[2], true);
    let g = t(&[1.0, 1.0], &[2]);

    let y = relu(&x).unwrap();
    y.backward(Some(&g), false).unwrap();
    let e = y.backward(Some(&g), false).unwrap_err();
    assert!(e.to_string().contains("backward through the graph a second time"), "{e}");

    x.set_grad(None).unwrap();
    let y = relu(&x).unwrap();
    y.backward(Some(&g), true).unwrap();
    y.backward(Some(&g), false).unwrap();
    assert_eq!(x.grad().unwrap().to_vec().unwrap(), [0.0, 2.0]);
}

#[test]
fn modifying_a_saved_tensor_is_caught_at_backward() {
    use grilly_core::autograd::no_grad;
    let g = t(&[1.0, 1.0, 1.0], &[1, 3]);

    // linear saves its input; change the input after the forward pass
    let x = leaf(&[1.0, -1.0], &[1, 2], true);
    let w = leaf(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], &[3, 2], true);
    let y = linear(&x, &w, None).unwrap();
    {
        let _g = no_grad();
        x.fill_(0.0).unwrap();
    }
    let e = y.backward(Some(&g), false).unwrap_err();
    assert!(e.to_string().contains("modified by an inplace operation"), "{e}");

    // relu saves its output; a detached alias shares the version counter
    let x = leaf(&[1.0, -1.0, 2.0], &[1, 3], true);
    let y = relu(&x).unwrap();
    y.detach().zero_().unwrap();
    let e = y.backward(Some(&g), false).unwrap_err();
    assert!(e.to_string().contains("modified by an inplace operation"), "{e}");
}
