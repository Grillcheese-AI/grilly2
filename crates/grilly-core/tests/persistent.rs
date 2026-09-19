//! `MemClass::Persistent`: device memory of exactly the requested size,
//! destroyed when released and never pooled (docs/inference.md). One test
//! function, in its own binary, so the device-wide counters it reads are not
//! moved by other tests running in parallel. `activation-relu` is the probe
//! kernel (exact in float32).

use grilly_core::{Arg, DType, Device, Kernel, MemClass, Tensor};

const PROBE: Kernel = Kernel {
    shader: "activation-relu",
    bindings: 2,
    push_words: 1,
    local_size: [256, 1, 1],
};

fn le(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn persistent_memory_is_exact_unpooled_and_outlives_its_batch() {
    let dev = Device::default_device().expect("open GPU");
    let base = dev.submit_stats().unwrap();

    // sizes that are not powers of two: a Qwen2.5-0.5B MLP weight, and two small ones
    let elems = [4864 * 896, 1000, 3];
    let held: Vec<Tensor> =
        elems.iter().map(|&n| Tensor::empty_in(&dev, &[n], DType::F32, MemClass::Persistent).unwrap()).collect();
    let s = dev.submit_stats().unwrap();
    assert_eq!(s.persistent_buffers - base.persistent_buffers, 3);
    assert_eq!(s.persistent_bytes - base.persistent_bytes, elems.iter().map(|n| n * 4).sum::<usize>() as u64);
    assert_eq!(s.buffers_outstanding, base.buffers_outstanding, "nothing taken from v1's pool");
    drop(held);
    let s = dev.submit_stats().unwrap();
    assert_eq!((s.persistent_buffers, s.persistent_bytes), (base.persistent_buffers, base.persistent_bytes));

    // released while a batch that reads it is still recording: the batch runs correctly
    let x: Vec<f32> = (0..4096).map(|i| i as f32 - 2048.0).collect();
    let w = Tensor::from_bytes_in(&dev, &le(&x), &[4096], DType::F32, MemClass::Persistent).unwrap();
    let y = Tensor::empty(&dev, &[4096]).unwrap();
    PROBE.launch(&dev, &[Arg::Tensor(&w), Arg::Tensor(&y)], [4096, 1, 1], &[4096]).unwrap();
    let recorded = dev.submit_stats().unwrap().submits;
    drop(w);
    assert_eq!(dev.submit_stats().unwrap().submits, recorded, "dropping must not submit the batch");
    assert_eq!(dev.submit_stats().unwrap().persistent_buffers, base.persistent_buffers);
    assert_eq!(y.to_vec().unwrap(), x.iter().map(|v| v.max(0.0)).collect::<Vec<_>>());
}
