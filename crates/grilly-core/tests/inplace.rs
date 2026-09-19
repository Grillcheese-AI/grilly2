//! In-place ops and version counters, against torch's rules.

use grilly_core::autograd::no_grad;
use grilly_core::rng::Generator;
use grilly_core::{add, Device, Error, Tensor};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(&dev(), data, shape).unwrap()
}

#[test]
fn inplace_ops_write_and_bump_the_version() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let view = x.reshape(&[2, 2]).unwrap();
    assert_eq!(x.version(), 0);

    x.add_(&t(&[1.0; 4], &[4])).unwrap();
    assert_eq!(x.to_vec().unwrap(), [2.0, 3.0, 4.0, 5.0]);
    x.fill_(7.5).unwrap();
    assert_eq!(x.to_vec().unwrap(), [7.5; 4]);
    x.copy_(&t(&[4.0, 3.0, 2.0, 1.0], &[4])).unwrap();
    assert_eq!(view.to_vec().unwrap(), [4.0, 3.0, 2.0, 1.0]); // views share data
    x.zero_().unwrap();
    assert_eq!(x.to_vec().unwrap(), [0.0; 4]);

    assert_eq!(x.version(), 4);
    assert_eq!(view.version(), 4); // and the counter, as in torch
}

#[test]
fn random_fills_are_deterministic_and_scaled() {
    let a = t(&[0.0; 1000], &[1000]);
    let b = t(&[0.0; 1000], &[1000]);
    a.normal_(3.0, 0.5, &mut Generator::new(9)).unwrap();
    b.normal_(3.0, 0.5, &mut Generator::new(9)).unwrap();
    let (va, vb) = (a.to_vec().unwrap(), b.to_vec().unwrap());
    assert_eq!(va, vb);
    let mean = va.iter().sum::<f32>() / 1000.0;
    assert!((mean - 3.0).abs() < 0.1, "mean {mean}");

    a.uniform_(-2.0, 2.0, &mut Generator::new(1)).unwrap();
    assert!(a.to_vec().unwrap().iter().all(|v| (-2.0..2.0).contains(v)));
}

#[test]
fn torch_inplace_rules() {
    let w = t(&[1.0, 2.0], &[2]);
    w.set_requires_grad(true).unwrap();

    // a leaf that requires grad: error outside no_grad, fine inside
    let e = w.zero_().unwrap_err();
    assert!(e.to_string().contains("leaf Variable that requires grad"), "{e}");
    {
        let _g = no_grad();
        w.fill_(0.5).unwrap();
    }
    assert_eq!(w.to_vec().unwrap(), [0.5, 0.5]);

    // a tensor inside a graph: history rebasing is not implemented
    let y = add(&w, &w).unwrap();
    assert!(matches!(y.fill_(1.0), Err(Error::NotImplemented(_))));

    // an operand that requires grad
    let c = t(&[0.0, 0.0], &[2]);
    assert!(matches!(c.add_(&w), Err(Error::NotImplemented(_))));

    // mul_ follows the same rules, and works
    assert!(w.mul_(2.0).is_err());
    c.fill_(3.0).unwrap().mul_(0.5).unwrap();
    assert_eq!(c.to_vec().unwrap(), [1.5, 1.5]);
}
