//! Capture a sequence of commands once and replay it (crate::graph).
//!
//! What is asserted:
//! - a replay **recomputes**: change the input tensor's bytes, replay, and
//!   the output must be what running the work again would give. That is the
//!   whole claim, so it is tested with values rather than with counters;
//! - a replay issues the **same dispatches**, so the saving is host time and
//!   not work skipped;
//! - a graph **outlives the tensors it was captured from** — it holds their
//!   buffers, so dropping the Rust handles must not leave it reading freed
//!   memory;
//! - a capture that reads to the host, or writes from it, is **refused**:
//!   neither can be reproduced by re-issuing commands, and silently doing so
//!   would give stale results on every replay after the first.

use grilly_core::rng::Generator;
use grilly_core::{graph, Arg, Device, Kernel, Tensor};

const RELU: Kernel = Kernel { shader: "g2-relu", bindings: 2, push_words: 1, local_size: [256, 1, 1] };

/// Serializes every test here. Capture is a property of the device, which
/// is one process-wide singleton, so a second test running in parallel has
/// its tensor creations and dispatches swept into the first one's capture.
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn alone() -> std::sync::MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

fn relu(dev: &Device, x: &Tensor, y: &Tensor, n: usize) {
    RELU.launch(dev, &[Arg::Tensor(x), Arg::Tensor(y)], [n as u32, 1, 1], &[n as u32]).unwrap();
}

/// Host write into an existing tensor: `uniform_(v, v)` is the one public
/// route, and with an empty range every sample is exactly `v`.
fn fill_from_host(t: &Tensor, value: f32) {
    let mut generator = Generator::new(0);
    t.uniform_(value, value, &mut generator).unwrap();
}

#[test]
fn a_replay_recomputes_from_whatever_the_inputs_now_hold() {
    let _alone = alone();
    let dev = dev();
    let n = 512;
    let x = Tensor::from_slice(&dev, &vec![-1.0f32; n], &[n]).unwrap();
    let mid = Tensor::empty(&dev, &[n]).unwrap();
    let y = Tensor::empty(&dev, &[n]).unwrap();

    // capture runs the work too, so this is the first "token"
    let (_, captured) = graph::capture(&dev, || {
        relu(&dev, &x, &mid, n);
        relu(&dev, &mid, &y, n);
        Ok(())
    })
    .unwrap();
    assert_eq!(captured.len(), 2, "two dispatches were recorded");
    assert_eq!(y.to_vec().unwrap(), vec![0.0f32; n], "relu of -1");

    for value in [3.0f32, -2.0, 0.5] {
        fill_from_host(&x, value);
        captured.replay().unwrap();
        assert_eq!(y.to_vec().unwrap(), vec![value.max(0.0); n], "replay with {value}");
    }
}

#[test]
fn a_replay_issues_the_same_dispatches() {
    let _alone = alone();
    let dev = dev();
    let n = 256;
    let x = Tensor::from_slice(&dev, &vec![1.0f32; n], &[n]).unwrap();
    let outs: Vec<Tensor> = (0..8).map(|_| Tensor::empty(&dev, &[n]).unwrap()).collect();

    let (_, captured) = graph::capture(&dev, || {
        for out in &outs {
            relu(&dev, &x, out, n);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(captured.len(), 8);

    dev.synchronize().unwrap();
    let before = dev.submit_stats().unwrap();
    captured.replay().unwrap();
    dev.synchronize().unwrap();
    let after = dev.submit_stats().unwrap();
    assert_eq!(after.dispatches - before.dispatches, 8, "a replay runs the same work");
}

#[test]
fn a_graph_outlives_the_tensors_it_was_captured_from() {
    let _alone = alone();
    let dev = dev();
    let n = 128;
    let y = Tensor::empty(&dev, &[n]).unwrap();
    let captured = {
        let x = Tensor::from_slice(&dev, &vec![2.0f32; n], &[n]).unwrap();
        let (_, captured) = graph::capture(&dev, || {
            relu(&dev, &x, &y, n);
            Ok(())
        })
        .unwrap();
        captured  // x is dropped here; the graph still names its buffer
    };
    captured.replay().unwrap();
    assert_eq!(y.to_vec().unwrap(), vec![2.0f32; n], "the captured input's buffer was freed");
}

#[test]
fn reading_or_writing_from_the_host_during_a_capture_is_refused() {
    let _alone = alone();
    let dev = dev();
    let n = 64;
    let x = Tensor::from_slice(&dev, &vec![1.0f32; n], &[n]).unwrap();
    let y = Tensor::empty(&dev, &[n]).unwrap();

    let read = graph::capture(&dev, || {
        relu(&dev, &x, &y, n);
        y.to_vec()
    });
    assert!(read.is_err(), "a readback during capture was allowed");
    assert!(format!("{}", read.unwrap_err()).contains("read data back"));

    let write = graph::capture(&dev, || {
        relu(&dev, &x, &y, n);
        let mut generator = Generator::new(0);  // not fill_from_host: the error must propagate
        x.uniform_(2.0, 2.0, &mut generator).map(|_| ())
    });
    assert!(write.is_err(), "a host write during capture was allowed");
    assert!(format!("{}", write.unwrap_err()).contains("wrote host data"));

    // the device is usable afterwards: a failed capture must not leave one open
    relu(&dev, &x, &y, n);
    assert_eq!(y.to_vec().unwrap()[0], 1.0);
}

#[test]
fn capturing_nothing_is_an_error_rather_than_an_empty_graph() {
    let _alone = alone();
    let dev = dev();
    let result = graph::capture(&dev, || Ok(()));
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("nothing was captured"));
}
