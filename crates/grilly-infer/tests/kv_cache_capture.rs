//! A KV cache refuses to be recorded into a command graph.
//!
//! Its own binary because capture is a property of the device, which is one
//! process-wide singleton: a test running in parallel would have its tensor
//! creations swept into this one's open capture.
//!
//! Why refusing is the answer rather than a tolerance: the cache carries its
//! fill level on the host, so a capture records the append offset and the
//! decode length as they stood. The tensor shapes never change with the
//! length, so nothing downstream would notice — every replay would attend
//! over the positions the capture saw and return a stale answer at full
//! confidence.

use grilly_core::{graph, Device, Tensor};
use grilly_infer::KvCache;

#[test]
fn a_kv_cache_refuses_to_be_captured() {
    let dev = Device::default_device().expect("open GPU");
    let mut cache = KvCache::new(&dev, 1, 1, 2, 8, 64).unwrap();
    let k = Tensor::from_slice(&dev, &vec![1.0f32; 2 * 8], &[1, 2, 1, 8]).unwrap();
    let v = Tensor::from_slice(&dev, &vec![2.0f32; 2 * 8], &[1, 2, 1, 8]).unwrap();
    cache.append(0, &k, &v).unwrap();
    let q = Tensor::from_slice(&dev, &vec![1.0f32; 2 * 8], &[1, 2, 1, 8]).unwrap();

    let appended = graph::capture(&dev, || cache.append(0, &k, &v));
    let attended = graph::capture(&dev, || cache.attend(0, &q, None).map(|_| ()));
    for (what, result) in [("append", appended), ("attend", attended)] {
        let e = result.err().unwrap_or_else(|| panic!("{what} was captured"));
        assert!(format!("{e}").contains("cannot be captured"), "{what}: {e}");
    }

    // a refused capture leaves nothing open: the cache still works
    assert_eq!(cache.attend(0, &q, None).unwrap().shape(), &[1, 2, 1, 8]);
    assert_eq!(cache.len(0), 1, "a refused append must not have advanced the length");
}
