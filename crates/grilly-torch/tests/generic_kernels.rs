//! grilly2's generic kernels (grilly_core::generic) against torch on CPU.
//!
//! Tolerances, all measured on an RX 6750 XT and recorded here:
//!
//! | ops | bound | measured grilly | torch f32 (same metric) |
//! |---|---|---|---|
//! | neg, abs, sign, +a, *a, a-x, pow(2), clamp, add, sub, mul, max, min, comparisons, copies | bit-identical to torch | | |
//! | exp, log, sqrt, rsqrt, tanh, sigmoid, a/x, integer pow, div, sin/cos (abs(x) <= 8192) | 2 ULP of the float64 truth | 0.95-1.57 | 0-1.13 |
//! | pow with a non-integer exponent (native GLSL pow) | 32 ULP | 16.6 (x^1.7), 7.4 (tensor exponents) | 0-1 |
//! | gelu, gelu' (erf approximation) | 3e-7 * max(1, abs(x)) absolute | 2.0e-7 | 2.1e-7 |
//! | sum, mean over K elements (Kahan) | (sqrt(K) + 2) * eps * sum(abs(x)) | within | |
//!
//! Non-integer pow is the one weak spot: matching torch there needs an
//! extended-precision log, left until a workload needs it.

mod common;

use common::{data, torch, torch_many};
use grilly_core::generic::{self, Binary, Reduce, Unary};
use grilly_core::{Device, Tensor};

const ULP_BOUND: f64 = 2.0;
const NON_INTEGER_POW_ULP_BOUND: f64 = 32.0;

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(&dev(), data, shape).unwrap()
}

/// Error of `got` against the float64 truth, in float32 ULPs of the truth.
fn ulps(got: &[f32], truth: &[f32]) -> f64 {
    got.iter()
        .zip(truth)
        .map(|(&g, &w)| {
            if w.is_nan() {
                return if g.is_nan() { 0.0 } else { f64::INFINITY };
            }
            if w.is_infinite() {
                return if g == w { 0.0 } else { f64::INFINITY };
            }
            let ulp = (f32::EPSILON as f64) * (w.abs() as f64).max(f32::MIN_POSITIVE as f64);
            (g as f64 - w as f64).abs() / ulp
        })
        .fold(0.0, f64::max)
}

fn exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()) || (*g == 0.0 && *w == 0.0),
            "{what}: element {i}: grilly {g} torch {w}"
        );
    }
}

/// Inputs spanning signs, magnitudes and zeros.
fn wide(n: usize, seed: u32) -> Vec<f32> {
    data(n, seed)
        .into_iter()
        .enumerate()
        .map(|(i, v)| match i % 7 {
            0 => v * 30.0,
            1 => v * 1e-3,
            2 => 0.0,
            _ => v * 4.0,
        })
        .collect()
}

#[test]
fn exact_unary_ops() {
    let x = wide(4099, 1);
    let xt = t(&x, &[4099]);
    let cases: &[(Unary, f32, f32, &str)] = &[
        (Unary::Neg, 0.0, 0.0, "-x"),
        (Unary::Abs, 0.0, 0.0, "x.abs()"),
        (Unary::Sign, 0.0, 0.0, "x.sign()"),
        (Unary::AddScalar, 0.75, 0.0, "x + 0.75"),
        (Unary::MulScalar, -2.5, 0.0, "x * -2.5"),
        (Unary::RsubScalar, 3.0, 0.0, "3.0 - x"),
        (Unary::PowScalar, 2.0, 0.0, "x.pow(2)"),
        (Unary::Clamp, -1.5, 2.0, "x.clamp(-1.5, 2.0)"),
    ];
    for &(op, a, b, expr) in cases {
        let got = generic::unary(&xt, op, a, b).unwrap().to_vec().unwrap();
        exact(&got, &torch(expr, &[("x", &x, &[4099])]), expr);
    }
}

#[test]
fn transcendental_unary_ops_within_2_ulp() {
    let x = wide(4099, 2);
    let pos: Vec<f32> = x.iter().map(|v| v.abs() + 1e-3).collect();
    let neg_int_pow: Vec<f32> = x.iter().map(|v| v.clamp(-8.0, 8.0)).collect();
    // RoPE-like angles, up to the documented 8192 limit
    let angles: Vec<f32> = data(4099, 9).iter().enumerate().map(|(i, v)| v * [1.0, 30.0, 700.0, 8000.0][i % 4]).collect();
    let cases: &[(Unary, f32, &[f32], &str)] = &[
        (Unary::Exp, 0.0, &x, "x.double().exp()"),
        (Unary::Log, 0.0, &pos, "x.double().log()"),
        (Unary::Sqrt, 0.0, &pos, "x.double().sqrt()"),
        (Unary::Rsqrt, 0.0, &pos, "x.double().rsqrt()"),
        (Unary::Tanh, 0.0, &x, "x.double().tanh()"),
        (Unary::Sigmoid, 0.0, &x, "x.double().sigmoid()"),
        (Unary::RdivScalar, 1.0, &pos, "1.0 / x.double()"),
        (Unary::PowScalar, 3.0, &neg_int_pow, "x.double().pow(3)"),
        (Unary::PowScalar, -2.0, &pos, "x.double().pow(-2)"),
        (Unary::PowScalar, 0.5, &pos, "x.double().pow(0.5)"),
        (Unary::PowScalar, 1.7, &pos, "x.double().pow(1.7)"),
        (Unary::Sin, 0.0, &angles, "x.double().sin()"),
        (Unary::Cos, 0.0, &angles, "x.double().cos()"),
    ];
    for &(op, a, input, truth_expr) in cases {
        let n = input.len();
        let got = generic::unary(&t(input, &[n]), op, a, 0.0).unwrap().to_vec().unwrap();
        let truth = torch(truth_expr, &[("x", input, &[n])]);
        let e = ulps(&got, &truth);
        println!("{op:?} a={a}: {e:.2} ulp");
        let bound = if op == Unary::PowScalar && a != a.floor() && a != 0.5 { NON_INTEGER_POW_ULP_BOUND } else { ULP_BOUND };
        assert!(e <= bound, "{op:?} a={a}: {e} ulp > {bound}");
    }
}

#[test]
fn gelu_within_absolute_bound() {
    let x = wide(4099, 3);
    let n = x.len();
    for (op, truth_expr) in [
        (Unary::Gelu, "torch.nn.functional.gelu(x.double())"),
        (Unary::GeluGrad, "grads(lambda v: torch.nn.functional.gelu(v.double()), (x,), torch.ones_like(x))[0]"),
    ] {
        let got = generic::unary(&t(&x, &[n]), op, 0.0, 0.0).unwrap().to_vec().unwrap();
        let truth = torch(truth_expr, &[("x", &x, &[n])]);
        for ((g, w), xi) in got.iter().zip(&truth).zip(&x) {
            assert!((g - w).abs() <= 3e-7 * xi.abs().max(1.0), "{op:?} at x={xi}: {g} vs {w}");
        }
    }
}

#[test]
fn unary_in_place_writes_through_and_bumps_version() {
    let x = t(&[1.0, -2.0, 3.0], &[3]);
    generic::unary_inplace(&x, Unary::MulScalar, 0.5, 0.0).unwrap();
    assert_eq!(x.to_vec().unwrap(), [0.5, -1.0, 1.5]);
    assert_eq!(x.version(), 1);
}

#[test]
fn binary_ops_broadcast_like_torch() {
    // (a shape, b shape)
    let shapes: &[(&[usize], &[usize])] = &[
        (&[5, 7], &[5, 7]),
        (&[3, 1, 5], &[4, 1]),
        (&[6], &[2, 3, 6]),
        (&[2, 3], &[]),
        (&[1, 4, 1, 2], &[3, 1, 5, 1]),
    ];
    let exact_ops = [
        (Binary::Add, "a + b"),
        (Binary::Sub, "a - b"),
        (Binary::Mul, "a * b"),
        (Binary::Maximum, "torch.maximum(a, b)"),
        (Binary::Minimum, "torch.minimum(a, b)"),
        (Binary::Gt, "(a > b).float()"),
        (Binary::Le, "(a <= b).float()"),
        (Binary::Eq, "(a == b).float()"),
    ];
    for (i, &(sa, sb)) in shapes.iter().enumerate() {
        let (a, b) = (wide(sa.iter().product(), 10 + i as u32), wide(sb.iter().product(), 20 + i as u32));
        let (at, bt) = (t(&a, sa), t(&b, sb));
        let inputs = [("a", &a[..], sa), ("b", &b[..], sb)];
        for (op, expr) in exact_ops {
            let got = generic::binary(&at, &bt, op).unwrap();
            let want = torch(expr, &inputs);
            exact(&got.to_vec().unwrap(), &want, &format!("{expr} {sa:?} {sb:?}"));
        }
        let got = generic::binary(&at, &bt, Binary::Div).unwrap().to_vec().unwrap();
        let e = ulps(&got, &torch("(a.double() / b.double()).float()", &inputs));
        assert!(e <= ULP_BOUND, "div {sa:?} {sb:?}: {e} ulp");
    }
    let e = generic::binary(&t(&[0.0; 6], &[2, 3]), &t(&[0.0; 4], &[4]), Binary::Add).unwrap_err();
    assert!(e.to_string().contains("must match the size"), "{e}");
}

#[test]
fn binary_pow_matches_float64() {
    let base: Vec<f32> = wide(1000, 4).iter().map(|v| v.abs() + 0.1).collect();
    let expo: Vec<f32> = data(1000, 5).iter().map(|v| v * 3.0).collect();
    let got = generic::binary(&t(&base, &[1000]), &t(&expo, &[1000]), Binary::Pow).unwrap().to_vec().unwrap();
    let truth = torch("a.double().pow(b.double()).float()", &[("a", &base, &[1000]), ("b", &expo, &[1000])]);
    let e = ulps(&got, &truth);
    println!("pow(tensor, tensor): {e:.2} ulp");
    assert!(e <= NON_INTEGER_POW_ULP_BOUND, "{e}");
}

#[test]
fn reductions_match_torch() {
    let shape = [3usize, 257, 5];
    let x = wide(shape.iter().product(), 6);
    let xt = t(&x, &shape);
    let inputs = [("x", &x[..], &shape[..])];
    for (dim, &k) in shape.iter().enumerate() {
        for keepdim in [false, true] {
            let kd = if keepdim { "True" } else { "False" };
            for (op, name) in [(Reduce::Sum, "sum"), (Reduce::Mean, "mean")] {
                let got = generic::reduce(&xt, dim, op, keepdim).unwrap();
                let want = torch_many(
                    &format!("(x.{name}({dim}, keepdim={kd}), x.double().abs().{name}({dim}, keepdim={kd}).float())"),
                    &inputs,
                );
                let bound: Vec<f32> = want[1].iter().map(|m| m * ((k as f32).sqrt() + 2.0) * f32::EPSILON).collect();
                let g = got.to_vec().unwrap();
                assert_eq!(g.len(), want[0].len());
                let worst = g.iter().zip(&want[0]).zip(&bound).map(|((a, b), c)| (a - b).abs() / c.max(f32::MIN_POSITIVE)).fold(0.0f32, f32::max);
                assert!(worst <= 1.0, "{name} dim {dim}: {worst} of bound");
            }
            for (op, expr) in [(Reduce::Max, "amax"), (Reduce::Min, "amin")] {
                let got = generic::reduce(&xt, dim, op, keepdim).unwrap();
                exact(&got.to_vec().unwrap(), &torch(&format!("x.{expr}({dim}, keepdim={kd})"), &inputs), expr);
                assert_eq!(got.dim(), if keepdim { 3 } else { 2 });
            }
        }
    }
    let total = generic::reduce_all(&xt, Reduce::Sum).unwrap();
    assert_eq!(total.shape(), &[] as &[usize]);
    let want = torch_many("(x.sum(), x.double().abs().sum().float())", &inputs);
    let bound = want[1][0] * ((x.len() as f32).sqrt() + 2.0) * f32::EPSILON;
    assert!((total.to_vec().unwrap()[0] - want[0][0]).abs() <= bound);
}

#[test]
fn sum_to_shape_undoes_broadcasting() {
    let g = t(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), &[2, 3, 4]);
    let want = torch("g.sum(0).sum(1, keepdim=True)", &[("g", &g.to_vec().unwrap(), &[2, 3, 4])]);
    let got = generic::sum_to_shape(&g, &[3, 1]).unwrap();
    assert_eq!(got.shape(), &[3, 1]);
    exact(&got.to_vec().unwrap(), &want, "sum_to_shape");
}

#[test]
fn strided_copy_transposes_and_writes_slices() {
    // transpose (2,3,4) -> (4,3,2)
    let x: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let src = t(&x, &[2, 3, 4]);
    let dst = Tensor::zeros(&dev(), &[4, 3, 2]).unwrap();
    generic::copy_strided(&src, 0, &[1, 4, 12], &dst, 0, &[6, 2, 1], &[4, 3, 2]).unwrap();
    exact(&dst.to_vec().unwrap(), &torch("x.permute(2, 1, 0).contiguous()", &[("x", &x, &[2, 3, 4])]), "permute");

    // write a (2, 2) block into rows 1..3, cols 2..4 of a (4, 5) zeros
    let out = Tensor::zeros(&dev(), &[4, 5]).unwrap();
    let block = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    generic::write_strided(&block, 0, &[2, 1], &out, 5 + 2, &[5, 1], &[2, 2]).unwrap();
    let want = torch("torch.nn.functional.pad(x, (2, 1, 1, 1))", &[("x", &[1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    exact(&out.to_vec().unwrap(), &want, "slice write");
    assert_eq!(out.version(), 1); // a fresh zeros tensor is at 0, as in torch; one write
}

#[test]
fn launches_beyond_one_dimension_of_workgroups() {
    // 65535 * 256 = 16,776,960; go past it so the grid needs a second row.
    let n = 16_777_216 + 1000;
    let x = Tensor::full(&dev(), &[n], 1.5).unwrap();
    let y = generic::unary(&x, Unary::MulScalar, 2.0, 0.0).unwrap();
    let total = generic::reduce_all(&y, Reduce::Max).unwrap().to_vec().unwrap()[0];
    let low = generic::reduce_all(&y, Reduce::Min).unwrap().to_vec().unwrap()[0];
    assert_eq!((total, low), (3.0, 3.0), "every element written exactly once");
}
