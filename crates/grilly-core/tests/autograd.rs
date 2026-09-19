//! Engine semantics, against torch's documented behaviour. Uses only core
//! ops (add, reshape, clone), whose gradients are exact.

use grilly_core::autograd::{is_grad_enabled, no_grad, set_grad_enabled};
use grilly_core::{add, Device, Error, Tensor};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(&dev(), data, shape).unwrap()
}

fn leaf(data: &[f32], shape: &[usize]) -> Tensor {
    let x = t(data, shape);
    x.set_requires_grad(true).unwrap();
    x
}

fn ones(shape: &[usize]) -> Tensor {
    Tensor::full(&dev(), shape, 1.0).unwrap()
}

#[test]
fn leaves_accumulate_in_place_across_backwards() {
    let x = leaf(&[1.0, 2.0, 3.0], &[3]);
    add(&x, &x).unwrap().backward(Some(&ones(&[3])), false).unwrap();
    let g = x.grad().expect("grad after backward");
    assert_eq!(g.to_vec().unwrap(), [2.0, 2.0, 2.0]);

    add(&x, &x).unwrap().backward(Some(&ones(&[3])), false).unwrap();
    assert_eq!(x.grad().unwrap().to_vec().unwrap(), [4.0, 4.0, 4.0]);
    // Accumulation is in place, as in torch: the earlier handle sees it.
    assert_eq!(g.to_vec().unwrap(), [4.0, 4.0, 4.0]);
}

#[test]
fn gradients_from_several_paths_are_summed() {
    // y = x.reshape(2,2) + clone(x).reshape(2,2) + reshape(x)  -> dy/dx = 3
    let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let a = x.reshape(&[2, 2]).unwrap();
    let b = x.clone_data().unwrap().reshape(&[2, 2]).unwrap();
    let y = add(&add(&a, &b).unwrap(), &x.reshape(&[2, 2]).unwrap()).unwrap();
    y.backward(Some(&ones(&[2, 2])), false).unwrap();
    assert_eq!(x.grad().unwrap().to_vec().unwrap(), [3.0; 4]);
    assert_eq!(x.grad().unwrap().shape(), &[4]);
}

#[test]
fn only_inputs_that_require_grad_get_one() {
    let x = leaf(&[1.0, 2.0], &[2]);
    let c = t(&[5.0, 6.0], &[2]);
    let y = add(&x, &c).unwrap();
    assert!(y.requires_grad() && !y.is_leaf());
    y.backward(Some(&ones(&[2])), false).unwrap();
    assert!(x.grad().is_some());
    assert!(c.grad().is_none());
}

#[test]
fn no_grad_records_nothing() {
    let x = leaf(&[1.0], &[1]);
    {
        let _g = no_grad();
        assert!(!is_grad_enabled());
        let y = add(&x, &x).unwrap();
        assert!(!y.requires_grad() && y.is_leaf() && y.grad_fn_name().is_none());
        {
            let _inner = set_grad_enabled(true);
            assert!(add(&x, &x).unwrap().requires_grad());
        }
        assert!(!is_grad_enabled());
    }
    assert!(is_grad_enabled());
}

#[test]
fn detach_cuts_the_graph_but_shares_data() {
    let x = leaf(&[1.0, 2.0], &[2]);
    let y = add(&x, &x).unwrap();
    let d = y.detach();
    assert!(!d.requires_grad() && d.is_leaf());
    assert_eq!(d.to_vec().unwrap(), y.to_vec().unwrap());
}

#[test]
fn grad_fn_names_follow_torch() {
    let x = leaf(&[1.0, 2.0], &[2]);
    assert_eq!(x.grad_fn_name(), None);
    assert_eq!(add(&x, &x).unwrap().grad_fn_name(), Some("AddBackward0"));
    assert_eq!(x.reshape(&[1, 2]).unwrap().grad_fn_name(), Some("ViewBackward0"));
    assert_eq!(x.clone_data().unwrap().grad_fn_name(), Some("CloneBackward0"));
}

#[test]
fn backward_on_a_leaf_sets_its_grad() {
    let x = leaf(&[1.0, 2.0], &[2]);
    let g = t(&[0.5, -1.0], &[2]);
    x.backward(Some(&g), false).unwrap();
    assert_eq!(x.grad().unwrap().to_vec().unwrap(), [0.5, -1.0]);
}

#[test]
fn a_scalar_root_needs_no_gradient() {
    let x = leaf(&[3.0], &[]);
    add(&x, &x).unwrap().backward(None, false).unwrap();
    assert_eq!(x.grad().unwrap().to_vec().unwrap(), [2.0]);
}

#[test]
fn torch_errors() {
    let x = leaf(&[1.0, 2.0], &[2]);
    let c = t(&[1.0, 2.0], &[2]);

    let e = add(&c, &c).unwrap().backward(Some(&ones(&[2])), false).unwrap_err();
    assert!(e.to_string().contains("does not require grad"), "{e}");

    let e = add(&x, &x).unwrap().backward(None, false).unwrap_err();
    assert!(e.to_string().contains("only for scalar outputs"), "{e}");

    let e = add(&x, &x).unwrap().backward(Some(&ones(&[3])), false).unwrap_err();
    assert!(matches!(e, Error::Shape(ref m) if m.contains("Mismatch in shape")), "{e}");

    let e = add(&x, &x).unwrap().set_requires_grad(false).unwrap_err();
    assert!(e.to_string().contains("only change requires_grad flags of leaf"), "{e}");
}

#[test]
fn gradients_are_bit_identical_across_runs() {
    let run = || {
        let x = leaf(&(0..64).map(|i| i as f32 * 0.37 - 5.0).collect::<Vec<_>>(), &[64]);
        let mut y = x.clone();
        for _ in 0..10 {
            y = add(&y, &x.reshape(&[64]).unwrap()).unwrap();
        }
        y.backward(Some(&ones(&[64])), false).unwrap();
        x.grad().unwrap().to_vec().unwrap()
    };
    let a = run();
    assert_eq!(a, vec![11.0; 64]);
    assert_eq!(a, run());
}
