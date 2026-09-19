//! Loading safetensors files, from Rust with no Python in the path (CLAUDE.md
//! §11: the machinery must work without any adapter). Files are written
//! with the `safetensors` crate itself.
//!
//! Tolerance: bit-exact throughout. Loading copies or widens; bf16 and f16
//! widen to float32 exactly, so there is no rounding to absorb. f16 is
//! checked against hand-computed float32 values, not the `half` crate the
//! loader uses; tests/python/test_safetensors.py checks every bf16 and f16
//! bit pattern against torch.

use grilly_core::{DType, Device, Error};
use grilly_infer::{LoadError, SafeTensorsFile};
use safetensors::tensor::TensorView;
use safetensors::{serialize, Dtype};
use std::collections::HashMap;

fn device() -> Device {
    Device::default_device().expect("open GPU")
}

fn file(tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)], metadata: Option<HashMap<String, String>>) -> Vec<u8> {
    let views = tensors
        .iter()
        .map(|(name, dtype, shape, data)| (name.to_string(), TensorView::new(*dtype, shape.clone(), data).unwrap()));
    // the crate's own map type (hashbrown, without its std feature)
    serialize(views, metadata.map(|m| m.into_iter().collect())).unwrap()
}

fn le<T: Copy, const N: usize>(values: &[T], to: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| to(v)).collect()
}

fn f32_bits(t: &grilly_core::Tensor) -> Vec<u32> {
    t.to_vec().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// Integer or bool elements as stored, widened to i64.
fn ints(t: &grilly_core::Tensor) -> Vec<i64> {
    let bytes = t.to_bytes().unwrap();
    match t.dtype() {
        DType::I64 => bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
        DType::I32 => bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
        DType::Bool => bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
        other => panic!("not an integer dtype: {other:?}"),
    }
}

#[test]
fn every_supported_dtype_loads_exactly() {
    let dev = device();
    let f32s = [1.5f32, -0.0, f32::MIN_POSITIVE / 8.0, f32::INFINITY, 3.0e38, -7.25];
    // f16 bit patterns and their float32 values, computed by hand
    let f16s: [(u16, f32); 7] = [
        (0x3C00, 1.0),
        (0xC000, -2.0),
        (0x7BFF, 65504.0),
        (0x0001, 1.0 / 16_777_216.0), // smallest subnormal, 2^-24
        (0x7C00, f32::INFINITY),
        (0x8000, -0.0),
        (0x3555, 1365.0 / 4096.0), // (1 + 341/1024) / 4
    ];
    let bf16s: [u16; 5] = [0x3F80, 0xC0A0, 0x0001, 0x7F80, 0x8000]; // 1, -5, subnormal, inf, -0
    let i64s = [i64::MIN, -1, 0, i64::MAX];
    let i32s = [i32::MIN, 7, i32::MAX];
    let bools = [0u8, 1, 1, 0, 2]; // a nonzero byte is true, as torch reads it
    let bytes = file(
        &[
            ("f32", Dtype::F32, vec![2, 3], le(&f32s, f32::to_le_bytes)),
            ("f16", Dtype::F16, vec![7], le(&f16s.map(|p| p.0), u16::to_le_bytes)),
            ("bf16", Dtype::BF16, vec![5, 1], le(&bf16s, u16::to_le_bytes)),
            ("i64", Dtype::I64, vec![4], le(&i64s, i64::to_le_bytes)),
            ("i32", Dtype::I32, vec![3], le(&i32s, i32::to_le_bytes)),
            ("bool", Dtype::BOOL, vec![5], bools.to_vec()),
            ("scalar", Dtype::F32, vec![], le(&[2.5f32], f32::to_le_bytes)),
            ("empty", Dtype::F32, vec![0, 3], vec![]),
        ],
        None,
    );
    let f = SafeTensorsFile::from_bytes(bytes).unwrap();

    let t = f.tensor(&dev, "f32").unwrap();
    assert_eq!((t.shape(), t.dtype()), (&[2usize, 3][..], DType::F32));
    assert_eq!(f32_bits(&t), f32s.map(f32::to_bits));

    let t = f.tensor(&dev, "f16").unwrap();
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(f32_bits(&t), f16s.map(|p| p.1.to_bits()));

    let t = f.tensor(&dev, "bf16").unwrap();
    assert_eq!((t.shape(), t.dtype()), (&[5usize, 1][..], DType::F32));
    assert_eq!(f32_bits(&t), bf16s.map(|h| (h as u32) << 16));

    let t = f.tensor(&dev, "i64").unwrap();
    assert_eq!(t.dtype(), DType::I64);
    assert_eq!(ints(&t), i64s);

    let t = f.tensor(&dev, "i32").unwrap();
    assert_eq!(t.dtype(), DType::I32);
    assert_eq!(ints(&t), i32s.map(i64::from));

    let t = f.tensor(&dev, "bool").unwrap();
    assert_eq!(t.dtype(), DType::Bool);
    assert_eq!(ints(&t), [0, 1, 1, 0, 1]);

    let t = f.tensor(&dev, "scalar").unwrap();
    assert_eq!((t.shape(), t.to_vec().unwrap()), (&[][..], vec![2.5]));

    let t = f.tensor(&dev, "empty").unwrap();
    assert_eq!((t.shape(), t.to_vec().unwrap().len()), (&[0usize, 3][..], 0));
}

#[test]
fn keys_offsets_and_metadata_match_the_safetensors_package() {
    let meta = HashMap::from([("format".to_string(), "pt".to_string())]);
    // serialize() orders data by dtype alignment, then name: i64 first
    let bytes = file(
        &[
            ("b", Dtype::F32, vec![3], vec![0; 12]),
            ("a", Dtype::F32, vec![2], vec![0; 8]),
            ("c", Dtype::I64, vec![1], vec![0; 8]),
        ],
        Some(meta.clone()),
    );
    let f = SafeTensorsFile::from_bytes(bytes).unwrap();
    assert_eq!(f.keys(), ["a", "b", "c"]);
    assert_eq!(f.offset_keys(), ["c", "a", "b"]); // what the Python package reports for this file
    assert_eq!(f.metadata(), Some(meta));
}

#[test]
fn open_maps_a_file_and_errors_are_classified() {
    let dev = device();
    let dir = std::env::temp_dir().join(format!("grilly-infer-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.safetensors");
    std::fs::write(&path, file(&[("w", Dtype::F32, vec![2], le(&[1.0f32, 2.0], f32::to_le_bytes))], None)).unwrap();

    let f = SafeTensorsFile::open(&path).unwrap();
    assert_eq!(f.tensor(&dev, "w").unwrap().to_vec().unwrap(), [1.0, 2.0]);
    assert!(f.metadata().is_none());
    assert!(matches!(f.tensor(&dev, "missing"), Err(LoadError::Format(m)) if m == "File does not contain tensor missing"));
    drop(f);

    let io = SafeTensorsFile::open(dir.join("absent.safetensors")).err().unwrap();
    assert!(matches!(io, LoadError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound), "{io}");

    std::fs::write(dir.join("empty.safetensors"), b"").unwrap();
    assert!(matches!(SafeTensorsFile::open(dir.join("empty.safetensors")), Err(LoadError::Format(_))));

    let mut garbage = 5u64.to_le_bytes().to_vec();
    garbage.extend_from_slice(b"{bad}");
    assert!(matches!(SafeTensorsFile::from_bytes(garbage), Err(LoadError::Format(_))));

    let mut truncated = file(&[("w", Dtype::F32, vec![4], vec![0; 16])], None);
    truncated.truncate(truncated.len() - 4);
    assert!(matches!(SafeTensorsFile::from_bytes(truncated), Err(LoadError::Format(_))));

    let f64s = SafeTensorsFile::from_bytes(file(&[("d", Dtype::F64, vec![1], vec![0; 8])], None)).unwrap();
    let e = f64s.tensor(&dev, "d").err().unwrap();
    assert!(matches!(e, LoadError::Tensor(Error::NotImplemented(ref m)) if m.contains("\"d\"") && m.contains("F64")), "{e}");

    std::fs::remove_dir_all(&dir).unwrap();
}
