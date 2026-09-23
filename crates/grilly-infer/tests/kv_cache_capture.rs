//! A decode step over a KV cache is recorded once and replayed.
//!
//! This is what the device cursor is for. The cache's fill level used to be
//! a host value the recorded commands carried — the append offset and the
//! length the decode shader reads — so a replay would write the same slot
//! and attend over the same positions, and because the cache's shapes never
//! change with the length, nothing downstream would have noticed.
//!
//! Tolerance: **bit-exact**. A replay issues the same commands over the same
//! buffers as the capture, and the reference here is the same ops run
//! eagerly on a second cache, in the same order (CLAUDE.md §9(a)). `==`.
//!
//! Its own binary, and its tests serialised, because capture is a property
//! of the device, which is one process-wide singleton: anything running in
//! parallel has its tensor creations and dispatches swept into whichever
//! capture happens to be open.

use grilly_core::{graph, Device, Tensor};
use grilly_infer::KvCache;

const B: usize = 1;
const H: usize = 2;
const D: usize = 8;
const MAX: usize = 64;

/// Serialises the tests here: see the module header.
static DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn alone() -> std::sync::MutexGuard<'static, ()> {
    DEVICE.lock().unwrap_or_else(|e| e.into_inner())
}

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn token(dev: &Device, seed: u64) -> Tensor {
    Tensor::from_slice(dev, &data(B * H * D, seed), &[B, H, 1, D]).unwrap()
}

/// One decode step: the new position in, one query against everything held.
fn step(cache: &mut KvCache, k: &Tensor, v: &Tensor, q: &Tensor) -> grilly_core::Result<Tensor> {
    cache.append(0, k, v)?;
    cache.attend(0, q, None)
}

fn seeded(dev: &Device) -> KvCache {
    let mut cache = KvCache::new(dev, 1, B, H, D, MAX).unwrap();
    for t in 0..2 {
        cache.append(0, &token(dev, 100 + t), &token(dev, 200 + t)).unwrap();
    }
    cache
}

#[test]
fn a_replayed_decode_step_advances_the_cache_and_attends_over_all_of_it() {
    let _alone = alone();
    let dev = Device::default_device().expect("open GPU");
    const STEPS: u64 = 6;

    // the reference: the same steps, eagerly, on their own cache
    let mut eager = seeded(&dev);
    let want: Vec<Vec<f32>> = (0..STEPS)
        .map(|t| {
            let out = step(&mut eager, &token(&dev, t), &token(&dev, 50 + t), &token(&dev, 80 + t)).unwrap();
            out.to_vec().unwrap()
        })
        .collect();

    // the captured one: the step's inputs are tensors the replay rereads
    let mut cache = seeded(&dev);
    let (k, v, q) = (token(&dev, 0), token(&dev, 50), token(&dev, 80));
    let (out, captured) = graph::capture(&dev, || step(&mut cache, &k, &v, &q)).unwrap();
    assert_eq!(out.to_vec().unwrap(), want[0], "the capture's own run");

    for t in 1..STEPS {
        for (dst, seed) in [(&k, t), (&v, 50 + t), (&q, 80 + t)] {
            dst.copy_(&token(&dev, seed)).unwrap();
        }
        captured.replay().unwrap();
        assert_eq!(out.to_vec().unwrap(), want[t as usize], "replay {t}");
    }

    // the cursor moved on every replay; the host mirror only moved when the
    // host drove the cache, which is exactly what KvCache::len documents
    assert_eq!(cache.device_len(0).unwrap(), 2 + STEPS as usize);
    assert_eq!(cache.len(0), 3);
}

#[test]
fn a_full_cache_saturates_under_replay_instead_of_writing_past_its_end() {
    let _alone = alone();
    let dev = Device::default_device().expect("open GPU");
    let mut cache = KvCache::new(&dev, 1, B, H, D, 4).unwrap();
    cache.append(0, &token(&dev, 1), &token(&dev, 2)).unwrap();
    let (k, v, q) = (token(&dev, 3), token(&dev, 4), token(&dev, 5));
    let (_, captured) = graph::capture(&dev, || step(&mut cache, &k, &v, &q)).unwrap();

    for _ in 0..10 {
        captured.replay().unwrap();
    }
    assert_eq!(cache.device_len(0).unwrap(), 4, "the cursor stops at max_len");
}
