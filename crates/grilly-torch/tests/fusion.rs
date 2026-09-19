//! grilly_core::fusion: a fused graph computes exactly (bit for bit) what
//! the same ops compute one g2 kernel at a time, because both call the same
//! `g2_unary` / `g2_binary` with `precise` results. The one exception is a
//! NaN's sign and payload, which IEEE 754 leaves unspecified and which
//! differ between, e.g., `-x + y` and `y - x`: any NaN equals any NaN here.
//! Signed zeros and infinities must match.

use grilly_core::fusion::Graph;
use grilly_core::generic::{self, Binary, Unary};
use grilly_core::{Device, Tensor};

fn device() -> Device {
    Device::default_device().expect("open GPU")
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    // deterministic values in [-3, 3], with a few specials mixed in
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            match i % 97 {
                13 => f32::NAN,
                29 => f32::INFINITY,
                41 => -0.0,
                _ => (x as f32 / u32::MAX as f32) * 6.0 - 3.0,
            }
        })
        .collect()
}

fn same_bits(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape(), b.shape());
    let (xf, yf) = (a.to_vec().unwrap(), b.to_vec().unwrap());
    let differs = |(p, q): (&f32, &f32)| !(p.is_nan() && q.is_nan()) && p.to_bits() != q.to_bits();
    if let Some(i) = xf.iter().zip(&yf).position(differs) {
        panic!("differ at {i}: fused {:e} ({:#x}) vs eager {:e} ({:#x})", xf[i], xf[i].to_bits(), yf[i], yf[i].to_bits());
    }
}

#[test]
fn tanh_gelu_chain_matches_eager_bit_for_bit() {
    // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))): eight kernels unfused
    let dev = device();
    let x = Tensor::from_slice(&dev, &data(10_000, 1), &[100, 100]).unwrap();
    let (c, k) = (0.044715f32, (2.0 / std::f32::consts::PI).sqrt());

    let cube = generic::unary(&x, Unary::PowScalar, 3.0, 0.0).unwrap();
    let inner = generic::binary(&x, &generic::unary(&cube, Unary::MulScalar, c, 0.0).unwrap(), Binary::Add).unwrap();
    let t = generic::unary(&generic::unary(&inner, Unary::MulScalar, k, 0.0).unwrap(), Unary::Tanh, 0.0, 0.0).unwrap();
    let eager = generic::unary(
        &generic::binary(&x, &generic::unary(&t, Unary::AddScalar, 1.0, 0.0).unwrap(), Binary::Mul).unwrap(),
        Unary::MulScalar,
        0.5,
        0.0,
    )
    .unwrap();

    let mut g = Graph::new();
    let xi = g.input();
    let (three, cp, kp, one, half) = (g.param(), g.param(), g.param(), g.param(), g.param());
    let cube = g.unary(Unary::PowScalar, xi, Some(three), None);
    let scaled = g.unary(Unary::MulScalar, cube, Some(cp), None);
    let inner = g.binary(Binary::Add, xi, scaled);
    let t = g.unary(Unary::MulScalar, inner, Some(kp), None);
    let t = g.unary(Unary::Tanh, t, None, None);
    let t = g.unary(Unary::AddScalar, t, Some(one), None);
    let y = g.binary(Binary::Mul, xi, t);
    let y = g.unary(Unary::MulScalar, y, Some(half), None);
    g.output(y);
    let fused = g.run(&[&x], &[3.0, c, k, 1.0, 0.5], &[100, 100]).unwrap();
    same_bits(&fused[0], &eager);
}

#[test]
fn every_unary_and_binary_op_matches_eager() {
    let dev = device();
    let x = Tensor::from_slice(&dev, &data(3000, 2), &[3000]).unwrap();
    let y = Tensor::from_slice(&dev, &data(3000, 3), &[3000]).unwrap();
    let unaries = [
        (Unary::Neg, 0.0, 0.0), (Unary::Abs, 0.0, 0.0), (Unary::Sign, 0.0, 0.0), (Unary::Exp, 0.0, 0.0),
        (Unary::Log, 0.0, 0.0), (Unary::Sqrt, 0.0, 0.0), (Unary::Rsqrt, 0.0, 0.0), (Unary::Tanh, 0.0, 0.0),
        (Unary::Sigmoid, 0.0, 0.0), (Unary::AddScalar, 1.5, 0.0), (Unary::MulScalar, -0.3, 0.0),
        (Unary::RsubScalar, 2.0, 0.0), (Unary::RdivScalar, 1.0, 0.0), (Unary::PowScalar, 2.5, 0.0),
        (Unary::PowScalar, 3.0, 0.0), (Unary::Clamp, -1.0, 1.0), (Unary::Gelu, 0.0, 0.0),
        (Unary::Sin, 0.0, 0.0), (Unary::Cos, 0.0, 0.0), (Unary::Floor, 0.0, 0.0), (Unary::Round, 0.0, 0.0),
    ];
    for (op, a, b) in unaries {
        // op(x) + y, fused against two eager kernels
        let eager = generic::binary(&generic::unary(&x, op, a, 0.0 + b).unwrap(), &y, Binary::Add).unwrap();
        let mut g = Graph::new();
        let (xi, yi) = (g.input(), g.input());
        let (pa, pb) = (g.param(), g.param());
        let u = g.unary(op, xi, Some(pa), Some(pb));
        let s = g.binary(Binary::Add, u, yi);
        g.output(s);
        let fused = g.run(&[&x, &y], &[a, b], &[3000]).unwrap();
        println!("{op:?}");
        same_bits(&fused[0], &eager);
    }
    for op in [Binary::Sub, Binary::Mul, Binary::Div, Binary::Pow, Binary::Maximum, Binary::Minimum, Binary::Gt] {
        let eager = generic::unary(&generic::binary(&x, &y, op).unwrap(), Unary::Exp, 0.0, 0.0).unwrap();
        let mut g = Graph::new();
        let (xi, yi) = (g.input(), g.input());
        let r = g.binary(op, xi, yi);
        let e = g.unary(Unary::Exp, r, None, None);
        g.output(e);
        println!("{op:?}");
        same_bits(&g.run(&[&x, &y], &[], &[3000]).unwrap()[0], &eager);
    }
}

#[test]
fn broadcasting_and_several_outputs() {
    let dev = device();
    // x (4, 1, 5), y (3, 1), z (5): out (4, 3, 5)
    let x = Tensor::from_slice(&dev, &data(20, 4), &[4, 1, 5]).unwrap();
    let y = Tensor::from_slice(&dev, &data(3, 5), &[3, 1]).unwrap();
    let z = Tensor::from_slice(&dev, &data(5, 6), &[5]).unwrap();
    let xy = generic::binary(&x, &y, Binary::Mul).unwrap();
    let eager_sum = generic::binary(&xy, &z, Binary::Add).unwrap();
    let eager_sig = generic::unary(&eager_sum, Unary::Sigmoid, 0.0, 0.0).unwrap();

    let mut g = Graph::new();
    let (xi, yi, zi) = (g.input(), g.input(), g.input());
    let m = g.binary(Binary::Mul, xi, yi);
    let s = g.binary(Binary::Add, m, zi);
    let sig = g.unary(Unary::Sigmoid, s, None, None);
    g.output(s);
    g.output(sig);
    let fused = g.run(&[&x, &y, &z], &[], &[4, 3, 5]).unwrap();
    same_bits(&fused[0], &eager_sum);
    same_bits(&fused[1], &eager_sig);
}

#[test]
fn adam_style_update_with_runtime_params_compiles_once() {
    // p - lr * m / (sqrt(v) + eps): the values of lr and eps change per step;
    // the kernel is compiled once
    let dev = device();
    let n = 5000;
    let p = Tensor::from_slice(&dev, &data(n, 7), &[n]).unwrap();
    let m = Tensor::from_slice(&dev, &data(n, 8), &[n]).unwrap();
    let v = generic::unary(&Tensor::from_slice(&dev, &data(n, 9), &[n]).unwrap(), Unary::Abs, 0.0, 0.0).unwrap();
    let mut g = Graph::new();
    let (pi, mi, vi) = (g.input(), g.input(), g.input());
    let (lr, eps) = (g.param(), g.param());
    let d = g.unary(Unary::Sqrt, vi, None, None);
    let d = g.unary(Unary::AddScalar, d, Some(eps), None);
    let step = g.binary(Binary::Div, mi, d);
    let step = g.unary(Unary::MulScalar, step, Some(lr), None);
    let out = g.binary(Binary::Sub, pi, step);
    g.output(out);
    for (lr, eps) in [(1e-3f32, 1e-8f32), (3e-4, 1e-6), (0.1, 1e-8)] {
        let d = generic::unary(&generic::unary(&v, Unary::Sqrt, 0.0, 0.0).unwrap(), Unary::AddScalar, eps, 0.0).unwrap();
        let step = generic::unary(&generic::binary(&m, &d, Binary::Div).unwrap(), Unary::MulScalar, lr, 0.0).unwrap();
        let eager = generic::binary(&p, &step, Binary::Sub).unwrap();
        same_bits(&g.run(&[&p, &m, &v], &[lr, eps], &[n]).unwrap()[0], &eager);
    }
    let name = dev.generated_kernel(&g.source()).unwrap();
    assert!(name.starts_with("g2-fused-"));
}

#[test]
fn malformed_graphs_are_refused() {
    let dev = device();
    let x = Tensor::from_slice(&dev, &[1.0, 2.0], &[2]).unwrap();
    let mut g = Graph::new();
    let xi = g.input();
    g.output(xi);
    assert!(g.run(&[], &[], &[2]).is_err()); // wrong input count
    assert!(g.run(&[&x], &[], &[3]).is_err()); // does not broadcast
    let mut bad = Graph::new();
    bad.nodes.push(grilly_core::fusion::Node::Binary(Binary::Add, 1, 2));
    bad.output(0);
    assert!(bad.run(&[&x], &[], &[2]).is_err());
}

#[test]
#[ignore = "timing, run with --ignored --nocapture"]
fn tanh_gelu_timing() {
    let dev = device();
    let n = 16 << 20;
    let x = Tensor::from_slice(&dev, &data(n, 11), &[n]).unwrap();
    let (c, k) = (0.044715f32, (2.0 / std::f32::consts::PI).sqrt());
    let eager = || {
        let cube = generic::unary(&x, Unary::PowScalar, 3.0, 0.0).unwrap();
        let inner = generic::binary(&x, &generic::unary(&cube, Unary::MulScalar, c, 0.0).unwrap(), Binary::Add).unwrap();
        let t = generic::unary(&generic::unary(&inner, Unary::MulScalar, k, 0.0).unwrap(), Unary::Tanh, 0.0, 0.0).unwrap();
        let y = generic::binary(&x, &generic::unary(&t, Unary::AddScalar, 1.0, 0.0).unwrap(), Binary::Mul).unwrap();
        generic::unary(&y, Unary::MulScalar, 0.5, 0.0).unwrap()
    };
    let mut g = Graph::new();
    let xi = g.input();
    let (three, cp, kp, one, half) = (g.param(), g.param(), g.param(), g.param(), g.param());
    let cube = g.unary(Unary::PowScalar, xi, Some(three), None);
    let scaled = g.unary(Unary::MulScalar, cube, Some(cp), None);
    let inner = g.binary(Binary::Add, xi, scaled);
    let t = g.unary(Unary::MulScalar, inner, Some(kp), None);
    let t = g.unary(Unary::Tanh, t, None, None);
    let t = g.unary(Unary::AddScalar, t, Some(one), None);
    let y = g.binary(Binary::Mul, xi, t);
    let y = g.unary(Unary::MulScalar, y, Some(half), None);
    g.output(y);
    let fused = || g.run(&[&x], &[3.0, c, k, 1.0, 0.5], &[n]).unwrap().remove(0);
    for (name, f) in [("eager (8 kernels)", &eager as &dyn Fn() -> Tensor), ("fused (1 kernel)", &fused)] {
        f().to_bytes().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            let _ = f();
        }
        f().to_bytes().unwrap();
        println!("tanh-gelu over 16M: {name}: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3 / 6.0);
    }
}
