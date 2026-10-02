//! grilly-core behaviour on a real device. Uses `g2-relu` only as a
//! probe of the dispatch path (exact in float32, so comparisons are exact).

use grilly_core::{Arg, Device, Error, Kernel, MemClass, Tensor};

const PROBE: Kernel = Kernel {
    shader: "g2-relu",
    bindings: 2,
    push_words: 1,
    local_size: [256, 1, 1],
};

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn probe(x: &Tensor) -> Tensor {
    let out = Tensor::empty(x.device(), x.shape()).unwrap();
    let n = x.numel() as u32;
    PROBE.launch(x.device(), &[Arg::Tensor(x), Arg::Tensor(&out)], [n, 1, 1], &[n]).unwrap();
    out
}

fn signed(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 - n as f32 / 2.0).collect()
}

#[test]
fn round_trip_and_reshape() {
    let x = signed(24);
    let t = Tensor::from_slice(&dev(), &x, &[2, 3, 4]).unwrap();
    assert_eq!(t.to_vec().unwrap(), x);
    let r = t.reshape(&[6, 4]).unwrap();
    assert_eq!(r.shape(), &[6, 4]);
    assert_eq!(r.to_vec().unwrap(), x);
    assert!(matches!(t.reshape(&[5, 5]), Err(Error::Shape(_))));
    assert!(matches!(Tensor::from_slice(&dev(), &x, &[5]), Err(Error::Shape(_))));
}

#[test]
fn zero_element_tensor() {
    let t = Tensor::zeros(&dev(), &[0, 7]).unwrap();
    assert_eq!(t.numel(), 0);
    assert!(t.to_vec().unwrap().is_empty());
}

/// Launches are recorded and run at the next host read; intermediates dropped
/// while the batch is open must not be recycled under the running work.
/// 2500 launches also crosses the per-batch launch cap twice.
#[test]
fn long_chain_with_dropped_intermediates() {
    let x = signed(4096);
    let mut t = Tensor::from_slice(&dev(), &x, &[4096]).unwrap();
    for _ in 0..2500 {
        t = probe(&t); // the previous tensor is dropped here
    }
    let want: Vec<f32> = x.iter().map(|v| v.max(0.0)).collect();
    assert_eq!(t.to_vec().unwrap(), want);
}

#[test]
fn launch_validates_interface() {
    let d = dev();
    let t = Tensor::zeros(&d, &[4]).unwrap();
    let err = PROBE.launch(&d, &[Arg::Tensor(&t)], [4, 1, 1], &[4]).unwrap_err();
    assert!(err.to_string().contains("takes 2 bindings"), "{err}");
    let err = PROBE.launch(&d, &[Arg::Tensor(&t), Arg::Tensor(&t)], [4, 1, 1], &[]).unwrap_err();
    assert!(err.to_string().contains("push-constant"), "{err}");
}

#[test]
fn unknown_kernel_is_an_error_not_a_crash() {
    let d = dev();
    let bogus = Kernel { shader: "no-such-kernel", ..PROBE };
    let t = Tensor::zeros(&d, &[4]).unwrap();
    let err = bogus.launch(&d, &[Arg::Tensor(&t), Arg::Tensor(&t)], [4, 1, 1], &[4]).unwrap_err();
    assert!(matches!(err, Error::Device(ref m) if m.contains("no-such-kernel")), "{err}");
    assert!(!d.has_kernel("no-such-kernel").unwrap());
    // The device stays usable.
    assert_eq!(probe(&Tensor::from_slice(&d, &[-1.0, 2.0], &[2]).unwrap()).to_vec().unwrap(), [0.0, 2.0]);
}

#[test]
fn buffers_round_trip_at_offsets() {
    let d = dev();
    let b = d.alloc(64, MemClass::Readback).unwrap();
    b.write(16, &[1, 2, 3, 4]).unwrap();
    let mut out = [0u8; 4];
    b.read(16, &mut out).unwrap();
    assert_eq!(out, [1, 2, 3, 4]);
    assert!(matches!(d.alloc(0, MemClass::Device), Err(Error::Invalid(_))));
}

#[test]
fn device_memory_transfers_at_offsets_and_across_staging_chunks() {
    // Device-local transfers are staged by the shim 64 MiB at a time: 80 MiB
    // crosses a chunk boundary, and the offsets are not chunk- or word-aligned.
    let d = dev();
    for mem in [MemClass::Device, MemClass::Persistent] {
        let n = 80 << 20;
        let b = d.alloc(n as u64 + 64, mem).unwrap();
        let data: Vec<u8> = (0..n).map(|i| (i * 7 + i / 251) as u8).collect();
        b.write(3, &data).unwrap();
        b.write(n as u64 + 5, &[9, 8, 7]).unwrap();
        let mut back = vec![0u8; n];
        b.read(3, &mut back).unwrap();
        assert!(back == data, "{mem:?}: the 80 MiB round trip differs");
        let mut tail = [0u8; 5];
        b.read(n as u64 + 3, &mut tail).unwrap();
        assert_eq!(tail[2..], [9, 8, 7], "{mem:?}");
    }
}
