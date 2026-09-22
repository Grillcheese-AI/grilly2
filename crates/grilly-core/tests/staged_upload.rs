//! A small host write records a copy into the open batch instead of closing
//! it. In its own binary: it reads the device-wide submit counters, which
//! another test running in parallel would move.
//!
//! What matters, and what is asserted:
//! - **the values are right**, including a write that lands *between* two
//!   recorded dispatches. Recording the copy puts it in program order behind
//!   a barrier, so a dispatch recorded before it must see the old data and
//!   one recorded after it the new — that ordering is the whole risk of not
//!   flushing, so it is the first thing tested;
//! - **it costs no submit.** That is the point: a submit is a fence wait,
//!   and a decode step that builds one small tensor per layer used to pay
//!   one per layer;
//! - a **large** write still flushes and uploads through the shim's bounded
//!   staging, so a weight is never held twice.

use grilly_core::rng::Generator;
use grilly_core::{Arg, DType, Device, Kernel, MemClass, Tensor};

/// `y = max(x, 0)`, a v1 kernel: enough to observe what a dispatch read.
const PROBE: Kernel = Kernel { shader: "activation-relu", bindings: 2, push_words: 1, local_size: [256, 1, 1] };

fn dev() -> Device {
    Device::default_device().expect("open GPU")
}

/// Write `value` into every element from the host, in place. `uniform_(v, v)`
/// is the one public route to `Buffer::write` on an existing tensor, and with
/// an empty range every sample is exactly `v`.
fn fill_from_host(t: &Tensor, value: f32) {
    let mut generator = Generator::new(0);
    t.uniform_(value, value, &mut generator).unwrap();
}

fn relu_into(dev: &Device, x: &Tensor, y: &Tensor, n: usize) {
    PROBE.launch(dev, &[Arg::Tensor(x), Arg::Tensor(y)], [n as u32, 1, 1], &[n as u32]).unwrap();
}

#[test]
fn a_small_write_is_recorded_in_order_and_costs_no_submit() {
    let dev = dev();
    let n = 1024;
    let x = Tensor::from_slice(&dev, &vec![1.0f32; n], &[n]).unwrap();
    let before_seen = Tensor::empty(&dev, &[n]).unwrap();
    let after_seen = Tensor::empty(&dev, &[n]).unwrap();

    // one batch: dispatch, write, dispatch — with nothing forcing a flush
    dev.synchronize().unwrap();
    let start = dev.submit_stats().unwrap();
    relu_into(&dev, &x, &before_seen, n);
    fill_from_host(&x, 7.0);
    relu_into(&dev, &x, &after_seen, n);
    let recorded = dev.submit_stats().unwrap();
    assert_eq!(recorded.submits, start.submits, "the write closed the batch");
    assert_eq!(recorded.staged_uploads - start.staged_uploads, 1, "the write was not staged");

    // the reads force the batch out; program order must have been kept
    assert_eq!(before_seen.to_vec().unwrap(), vec![1.0f32; n], "the earlier dispatch saw the new data");
    assert_eq!(after_seen.to_vec().unwrap(), vec![7.0f32; n], "the later dispatch saw the old data");
    assert_eq!(x.to_vec().unwrap(), vec![7.0f32; n]);
}

#[test]
fn many_small_writes_in_one_batch_all_land() {
    let dev = dev();
    let n = 256;
    let x = Tensor::from_slice(&dev, &vec![0.0f32; n], &[n]).unwrap();
    let seen: Vec<Tensor> = (0..32).map(|_| Tensor::empty(&dev, &[n]).unwrap()).collect();

    dev.synchronize().unwrap();
    let start = dev.submit_stats().unwrap();
    for (i, out) in seen.iter().enumerate() {
        fill_from_host(&x, i as f32);
        relu_into(&dev, &x, out, n);
    }
    let recorded = dev.submit_stats().unwrap();
    assert_eq!(recorded.submits, start.submits, "32 writes and 32 dispatches should be one batch");
    assert_eq!(recorded.staged_uploads - start.staged_uploads, 32);

    for (i, out) in seen.iter().enumerate() {
        assert_eq!(out.to_vec().unwrap(), vec![i as f32; n], "write {i} did not reach its dispatch");
    }
}

#[test]
fn a_large_write_still_flushes_rather_than_staging_it() {
    let dev = dev();
    // over grilly-core's STAGED_UPLOAD_MAX (1 MiB): a weight-sized upload
    let n = 512 * 1024;
    let x = Tensor::empty_in(&dev, &[n], DType::F32, MemClass::Persistent).unwrap();
    dev.synchronize().unwrap();
    let start = dev.submit_stats().unwrap();
    fill_from_host(&x, 3.5);
    let after = dev.submit_stats().unwrap();
    assert_eq!(after.staged_uploads, start.staged_uploads, "a weight-sized upload was staged");
    assert_eq!(x.to_vec().unwrap()[..4], [3.5f32; 4]);
}

#[test]
fn a_staged_write_survives_a_batch_boundary() {
    let dev = dev();
    let n = 64;
    let x = Tensor::from_slice(&dev, &vec![0.0f32; n], &[n]).unwrap();
    let out = Tensor::empty(&dev, &[n]).unwrap();
    for round in 0..4 {
        fill_from_host(&x, round as f32 + 1.0);
        relu_into(&dev, &x, &out, n);
        // to_vec submits and waits, so the staging blocks are reused next round
        assert_eq!(out.to_vec().unwrap(), vec![round as f32 + 1.0; n], "round {round}");
    }
}
