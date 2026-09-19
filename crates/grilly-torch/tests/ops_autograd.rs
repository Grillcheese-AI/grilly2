//! Differentiable ops (grilly_torch::math and ::shape) against torch run in
//! float64: forward values and every input gradient, for a random upstream
//! gradient.
//!
//! Tolerance for these elementwise, reduction, shape and composite ops:
//! |grilly - truth| <= 1e-6 * max|truth| + 1e-7 over each tensor (about 8
//! float32 ULP of the tensor's scale); measured worst was 0.14 of it. The worst
//! measured ratio to that bound is printed per case.

mod common;

use common::{data, torch_many};
use grilly_core::generic::{Binary, Unary};
use grilly_core::{Device, Result, Tensor};
use grilly_torch::{math, shape};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(&dev(), data, shape).unwrap()
}

type Op = fn(&[Tensor]) -> Result<Tensor>;

struct Case {
    name: &'static str,
    shapes: &'static [&'static [usize]],
    /// Inputs made positive (for log, sqrt, pow, division).
    positive: bool,
    /// A Python lambda over the inputs, as torch would write it.
    torch: &'static str,
    grilly: Op,
}

fn check(case: &Case) -> f32 {
    let inputs: Vec<Vec<f32>> = case
        .shapes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n = s.iter().product();
            data(n, 100 + i as u32 * 7).into_iter().map(|v| if case.positive { v.abs() + 0.2 } else { v * 2.0 }).collect()
        })
        .collect();
    let leaves: Vec<Tensor> = inputs
        .iter()
        .zip(case.shapes)
        .map(|(d, s)| {
            let x = t(d, s);
            x.set_requires_grad(true).unwrap();
            x
        })
        .collect();
    let out = (case.grilly)(&leaves).unwrap_or_else(|e| panic!("{}: {e}", case.name));
    let g = data(out.numel(), 999);
    out.backward(Some(&t(&g, out.shape())), false).unwrap();

    let names: Vec<String> = (0..inputs.len()).map(|i| format!("x{i}")).collect();
    let args = names.join(", ");
    let expr = format!(
        "(lambda f: [f(*[v.double() for v in ({args},)])] + list(grads(lambda *xs: f(*[x.double() for x in xs]), ({args},), g.double())))({})",
        case.torch
    );
    let mut torch_inputs: Vec<(&str, &[f32], &[usize])> =
        names.iter().zip(&inputs).zip(case.shapes).map(|((n, d), s)| (n.as_str(), &d[..], *s)).collect();
    torch_inputs.push(("g", &g, out.shape()));
    let truth = torch_many(&expr, &torch_inputs);

    let mut got = vec![out.to_vec().unwrap()];
    got.extend(leaves.iter().map(|l| l.grad().expect("every input requires grad").to_vec().unwrap()));
    let mut worst = 0f32;
    for (i, (a, b)) in got.iter().zip(&truth).enumerate() {
        assert_eq!(a.len(), b.len(), "{}: tensor {i} length", case.name);
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let bound = 1e-6 * scale + 1e-7;
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            let r = (x - y).abs() / bound;
            assert!(r <= 1.0, "{}: {} element {j}: grilly {x}, float64 torch {y}", case.name, if i == 0 { "output".to_string() } else { format!("grad of input {}", i - 1) });
            worst = worst.max(r);
        }
    }
    worst
}

fn b(op: Binary) -> impl Fn(&[Tensor]) -> Result<Tensor> {
    move |x| math::binary(&x[0], &x[1], op)
}

macro_rules! cases {
    ($($name:literal, $shapes:expr, $pos:expr, $torch:literal, $f:expr;)*) => {
        &[$(Case { name: $name, shapes: $shapes, positive: $pos, torch: $torch, grilly: $f },)*]
    };
}

const CASES: &[Case] = cases! {
    "add broadcast", &[&[3, 1, 4], &[2, 4]], false, "lambda a, b: a + b", |x| b(Binary::Add)(x);
    "sub broadcast", &[&[2, 4], &[3, 1, 4]], false, "lambda a, b: a - b", |x| b(Binary::Sub)(x);
    "mul broadcast", &[&[5, 1], &[1, 6]], false, "lambda a, b: a * b", |x| b(Binary::Mul)(x);
    "div broadcast", &[&[4, 3], &[3]], true, "lambda a, b: a / b", |x| b(Binary::Div)(x);
    "pow tensor", &[&[6], &[6]], true, "lambda a, b: a ** b", |x| b(Binary::Pow)(x);
    "maximum", &[&[7], &[7]], false, "lambda a, b: torch.maximum(a, b)", |x| b(Binary::Maximum)(x);
    "minimum broadcast", &[&[3, 4], &[4]], false, "lambda a, b: torch.minimum(a, b)", |x| b(Binary::Minimum)(x);
    "x * 2.5", &[&[3, 5]], false, "lambda a: a * 2.5", |x| math::unary(&x[0], Unary::MulScalar, 2.5, 0.0);
    "x + 1", &[&[3, 5]], false, "lambda a: a + 1", |x| math::unary(&x[0], Unary::AddScalar, 1.0, 0.0);
    "3 - x", &[&[3, 5]], false, "lambda a: 3 - a", |x| math::unary(&x[0], Unary::RsubScalar, 3.0, 0.0);
    "2 / x", &[&[3, 5]], true, "lambda a: 2 / a", |x| math::unary(&x[0], Unary::RdivScalar, 2.0, 0.0);
    "x ** 3", &[&[3, 5]], false, "lambda a: a ** 3", |x| math::unary(&x[0], Unary::PowScalar, 3.0, 0.0);
    "x ** 0.5", &[&[3, 5]], true, "lambda a: a ** 0.5", |x| math::unary(&x[0], Unary::PowScalar, 0.5, 0.0);
    "clamp", &[&[20]], false, "lambda a: a.clamp(-0.5, 0.7)", |x| math::unary(&x[0], Unary::Clamp, -0.5, 0.7);
    "neg", &[&[8]], false, "lambda a: -a", |x| math::unary(&x[0], Unary::Neg, 0.0, 0.0);
    "abs", &[&[8]], false, "lambda a: a.abs()", |x| math::unary(&x[0], Unary::Abs, 0.0, 0.0);
    "exp", &[&[8]], false, "lambda a: a.exp()", |x| math::unary(&x[0], Unary::Exp, 0.0, 0.0);
    "log", &[&[8]], true, "lambda a: a.log()", |x| math::unary(&x[0], Unary::Log, 0.0, 0.0);
    "sqrt", &[&[8]], true, "lambda a: a.sqrt()", |x| math::unary(&x[0], Unary::Sqrt, 0.0, 0.0);
    "rsqrt", &[&[8]], true, "lambda a: a.rsqrt()", |x| math::unary(&x[0], Unary::Rsqrt, 0.0, 0.0);
    "tanh", &[&[8]], false, "lambda a: a.tanh()", |x| math::unary(&x[0], Unary::Tanh, 0.0, 0.0);
    "sigmoid", &[&[8]], false, "lambda a: a.sigmoid()", |x| math::unary(&x[0], Unary::Sigmoid, 0.0, 0.0);
    "gelu", &[&[8]], false, "lambda a: torch.nn.functional.gelu(a)", |x| math::unary(&x[0], Unary::Gelu, 0.0, 0.0);
    "silu", &[&[8]], false, "lambda a: torch.nn.functional.silu(a)", |x| math::silu(&x[0]);
    "sum dim 1", &[&[3, 4, 5]], false, "lambda a: a.sum(1)", |x| math::sum(&x[0], &[1], false);
    "sum dims keepdim", &[&[3, 4, 5]], false, "lambda a: a.sum((0, 2), keepdim=True)", |x| math::sum(&x[0], &[0, 2], true);
    "sum all", &[&[3, 4, 5]], false, "lambda a: a.sum()", |x| math::sum(&x[0], &[], false);
    "mean dim", &[&[3, 4, 5]], false, "lambda a: a.mean(-1)", |x| math::mean(&x[0], &[2], false);
    "mean all", &[&[3, 4]], false, "lambda a: a.mean()", |x| math::mean(&x[0], &[], false);
    "amax dim", &[&[3, 7]], false, "lambda a: a.amax(1)", |x| math::amax(&x[0], &[1], false, true);
    "amin keepdim", &[&[3, 7]], false, "lambda a: a.amin(0, keepdim=True)", |x| math::amax(&x[0], &[0], true, false);
    "softmax", &[&[4, 9]], false, "lambda a: torch.softmax(a, -1)", |x| math::softmax(&x[0], 1);
    "log_softmax", &[&[4, 9]], false, "lambda a: torch.log_softmax(a, 0)", |x| math::log_softmax(&x[0], 0);
    "normalize", &[&[5, 6]], false, "lambda a: torch.nn.functional.normalize(a, dim=-1)", |x| math::normalize(&x[0], 1, 1e-12);
    "layer_norm", &[&[4, 3, 8], &[8], &[8]], false, "lambda a, w, b: torch.nn.functional.layer_norm(a, (8,), w, b)",
        |x| math::layer_norm(&x[0], 1, Some(&x[1]), Some(&x[2]), 1e-5);
    "transpose", &[&[2, 3, 4]], false, "lambda a: a.transpose(0, 2)", |x| shape::transpose(&x[0], 0, 2);
    "permute", &[&[2, 3, 4]], false, "lambda a: a.permute(1, 2, 0)", |x| shape::permute(&x[0], &[1, 2, 0]);
    "expand", &[&[3, 1]], false, "lambda a: a.expand(2, 3, 5)", |x| shape::expand(&x[0], &[2, 3, 5]);
    "narrow", &[&[4, 6]], false, "lambda a: a.narrow(1, 2, 3)", |x| shape::narrow(&x[0], 1, 2, 3);
    "select", &[&[4, 6]], false, "lambda a: a.select(0, 2)", |x| shape::select(&x[0], 0, 2);
    "strided slice", &[&[5, 6]], false, "lambda a: a[1:5:2, ::3]", |x| shape::strided_read(&x[0], 6, &[12, 3], &[2, 2], "SliceBackward0");
    "cat", &[&[2, 3], &[2, 5]], false, "lambda a, b: torch.cat([a, b], 1)", |x| shape::cat(&[&x[0], &x[1]], 1);
    "stack", &[&[2, 3], &[2, 3], &[2, 3]], false, "lambda a, b, c: torch.stack([a, b, c], 1)", |x| shape::stack(&[&x[0], &x[1], &x[2]], 1);
    "matmul 2d", &[&[4, 5], &[5, 3]], false, "lambda a, b: a @ b", |x| shape::matmul(&x[0], &x[1]);
    "matmul 3d@2d", &[&[2, 4, 5], &[5, 3]], false, "lambda a, b: a @ b", |x| shape::matmul(&x[0], &x[1]);
    "matmul batched", &[&[2, 4, 5], &[2, 5, 3]], false, "lambda a, b: a @ b", |x| shape::matmul(&x[0], &x[1]);
    "matmul vec", &[&[5], &[5]], false, "lambda a, b: a @ b", |x| shape::matmul(&x[0], &x[1]);
    "matmul mat@vec", &[&[3, 5], &[5]], false, "lambda a, b: a @ b", |x| shape::matmul(&x[0], &x[1]);
};

#[test]
fn ops_and_gradients_match_float64_torch() {
    for case in CASES {
        let worst = check(case);
        println!("{:<18} worst / bound = {worst:.3}", case.name);
    }
}

#[test]
fn maximum_splits_tied_gradients_like_torch() {
    let (a, bb) = (t(&[1.0, 2.0, 3.0], &[3]), t(&[1.0, 5.0, 0.0], &[3]));
    a.set_requires_grad(true).unwrap();
    bb.set_requires_grad(true).unwrap();
    math::binary(&a, &bb, Binary::Maximum).unwrap().backward(Some(&t(&[1.0; 3], &[3])), false).unwrap();
    assert_eq!(a.grad().unwrap().to_vec().unwrap(), [0.5, 0.0, 1.0]);
    assert_eq!(bb.grad().unwrap().to_vec().unwrap(), [0.5, 1.0, 0.0]);
}

#[test]
fn view_copies_refuse_in_place_writes() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tr = shape::transpose(&x, 0, 1).unwrap();
    assert!(tr.is_view_copy());
    assert!(matches!(tr.fill_(0.0), Err(grilly_core::Error::NotImplemented(_))));
    // write_region is a real in-place write into the original
    shape::write_region(&x, 1, &[2], &[2], &t(&[9.0], &[])).unwrap();
    assert_eq!(x.to_vec().unwrap(), [1.0, 9.0, 3.0, 9.0]);
}

#[test]
fn grad_fn_names_follow_torch() {
    let x = t(&[1.0; 6], &[2, 3]);
    x.set_requires_grad(true).unwrap();
    let names = [
        (math::binary(&x, &x, Binary::Mul).unwrap(), "MulBackward0"),
        (math::unary(&x, Unary::Exp, 0.0, 0.0).unwrap(), "ExpBackward0"),
        (math::sum(&x, &[], false).unwrap(), "SumBackward0"),
        (math::sum(&x, &[1], false).unwrap(), "SumBackward1"),
        (shape::transpose(&x, 0, 1).unwrap(), "TransposeBackward0"),
        (shape::cat(&[&x, &x], 0).unwrap(), "CatBackward0"),
    ];
    for (y, name) in names {
        assert_eq!(y.grad_fn_name(), Some(name));
    }
}
