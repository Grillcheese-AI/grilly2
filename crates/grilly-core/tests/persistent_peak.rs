//! The persistent high-water mark, in its own binary for the same reason
//! `persistent.rs` is: the counters are device-wide, so another test
//! allocating in parallel would move them.

use grilly_core::{DType, Device, MemClass, Tensor};

/// `persistent_peak` is the most persistent bytes alive at once, and
/// `reset_persistent_peak` starts it again from what is alive now. It is
/// what a load's memory claim is measured against
/// (`tests/python/test_load_quantized.py`), so it is tested, not trusted.
#[test]
fn the_persistent_high_water_mark_tracks_and_resets() {
    let dev = Device::default_device().expect("open GPU");
    let keep = Tensor::empty_in(&dev, &[256 * 1024], DType::F32, MemClass::Persistent).unwrap();
    dev.reset_persistent_peak().unwrap();
    let held = dev.submit_stats().unwrap();
    assert_eq!(held.persistent_peak, held.persistent_bytes, "the reset starts from what is alive");

    let big = 4 * 1024 * 1024;
    let spike = Tensor::empty_in(&dev, &[big], DType::F32, MemClass::Persistent).unwrap();
    let top = dev.submit_stats().unwrap();
    assert_eq!(top.persistent_peak, held.persistent_bytes + (big * 4) as u64);
    drop(spike);
    let after = dev.submit_stats().unwrap();
    assert_eq!(after.persistent_bytes, held.persistent_bytes, "freed");
    assert_eq!(after.persistent_peak, top.persistent_peak, "the mark stays until it is reset");

    // a smaller allocation must not lower it, and a reset must
    let small = Tensor::empty_in(&dev, &[1024], DType::F32, MemClass::Persistent).unwrap();
    assert_eq!(dev.submit_stats().unwrap().persistent_peak, top.persistent_peak);
    dev.reset_persistent_peak().unwrap();
    let s = dev.submit_stats().unwrap();
    assert_eq!(s.persistent_peak, s.persistent_bytes);
    assert!(s.persistent_peak < top.persistent_peak);
    drop((keep, small));
}
