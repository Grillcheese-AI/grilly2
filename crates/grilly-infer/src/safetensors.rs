//! safetensors checkpoints, loaded onto the GPU (CLAUDE.md §11, step 1).
//!
//! The file is memory-mapped, and its header is parsed and validated
//! (offsets, sizes, overlaps) by Hugging Face's `safetensors` crate. Each
//! tensor is then uploaded in the form grilly stores it:
//!
//! | file dtype | grilly dtype | conversion |
//! |---|---|---|
//! | F32 | float32 | none |
//! | BF16, F16 | float32 | widened; exact for every value |
//! | I64, I32 | int64, int32 | none |
//! | BOOL | bool | one byte to grilly's one u32 per element |
//!
//! grilly has no half-precision dtypes yet. A bf16 or f16 checkpoint therefore
//! loads with its values unchanged, but reports its dtype as float32. Any other
//! dtype is refused (`Error::NotImplemented`) rather than converted with a loss.
//!
//! Loaded tensors are weights, so they go in `MemClass::Persistent` memory:
//! exactly their size, and freed back to the driver rather than pooled.

use grilly_core::{DType, Device, Error, MemClass, Tensor};
use half::f16;
use memmap2::Mmap;
use safetensors::tensor::{Dtype, Metadata, SafeTensors};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::Path;

/// Why a file or a tensor could not be loaded.
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be opened or mapped.
    Io(std::io::Error),
    /// Not a valid safetensors file, or no tensor of that name: what the
    /// safetensors package raises as `SafetensorError`.
    Format(String),
    /// grilly cannot hold the tensor's dtype, or the upload failed.
    Tensor(Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "{e}"),
            LoadError::Format(m) => write!(f, "{m}"),
            LoadError::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<Error> for LoadError {
    fn from(e: Error) -> LoadError {
        LoadError::Tensor(e)
    }
}

enum Bytes {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl Bytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Mapped(m) => m,
            Bytes::Owned(v) => v,
        }
    }
}

/// An open safetensors file: header parsed, data mapped, nothing uploaded.
pub struct SafeTensorsFile {
    bytes: Bytes,
    /// Where the data section starts: the 8-byte length plus the header.
    data_start: usize,
    metadata: Metadata,
}

impl SafeTensorsFile {
    /// Map the file at `path` and parse its header. As with any mapped file,
    /// it must not be modified while this is open.
    pub fn open(path: impl AsRef<Path>) -> Result<SafeTensorsFile, LoadError> {
        let file = File::open(path).map_err(LoadError::Io)?;
        if file.metadata().map_err(LoadError::Io)?.len() == 0 {
            // an empty file cannot be mapped on every platform; let the
            // parser reject it as it rejects any other too-short file
            return SafeTensorsFile::parse(Bytes::Owned(Vec::new()));
        }
        // SAFETY: a read-only map. Changing the file while it is mapped is
        // undefined behaviour here, as in the safetensors package, which maps
        // files the same way.
        let map = unsafe { Mmap::map(&file) }.map_err(LoadError::Io)?;
        SafeTensorsFile::parse(Bytes::Mapped(map))
    }

    /// Parse a whole file already in memory (the safetensors package's
    /// `load(bytes)`).
    pub fn from_bytes(data: Vec<u8>) -> Result<SafeTensorsFile, LoadError> {
        SafeTensorsFile::parse(Bytes::Owned(data))
    }

    fn parse(bytes: Bytes) -> Result<SafeTensorsFile, LoadError> {
        let (header_len, metadata) = SafeTensors::read_metadata(bytes.as_slice())
            .map_err(|e| LoadError::Format(format!("Error while deserializing header: {e}")))?;
        Ok(SafeTensorsFile { bytes, data_start: 8 + header_len, metadata })
    }

    /// Tensor names, sorted (the safetensors package's `keys()`).
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.metadata.tensors().into_keys().collect();
        keys.sort();
        keys
    }

    /// Tensor names in the order of their data in the file, which is the
    /// order to load them in for sequential reads (`offset_keys()`).
    pub fn offset_keys(&self) -> Vec<String> {
        self.metadata.offset_keys()
    }

    /// The header's free-form `__metadata__` strings, if it has any.
    pub fn metadata(&self) -> Option<HashMap<String, String>> {
        let m = self.metadata.metadata().as_ref()?;
        Some(m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    /// Upload one tensor to `device`, in persistent memory (see the module
    /// doc for dtypes).
    pub fn tensor(&self, device: &Device, name: &str) -> Result<Tensor, LoadError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| LoadError::Format(format!("File does not contain tensor {name}")))?;
        let (start, end) = info.data_offsets;
        let data = &self.bytes.as_slice()[self.data_start + start..self.data_start + end];
        let shape = &info.shape;
        let upload = |bytes: &[u8], dtype| Tensor::from_bytes_in(device, bytes, shape, dtype, MemClass::Persistent);
        let halves = || data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]));
        Ok(match info.dtype {
            Dtype::F32 => upload(data, DType::F32)?,
            Dtype::I64 => upload(data, DType::I64)?,
            Dtype::I32 => upload(data, DType::I32)?,
            // bf16 is the top half of a float32
            Dtype::BF16 => upload(&words(halves().map(|h| (h as u32) << 16)), DType::F32)?,
            Dtype::F16 => upload(&words(halves().map(|h| f16::from_bits(h).to_f32().to_bits())), DType::F32)?,
            Dtype::BOOL => upload(&words(data.iter().map(|&b| u32::from(b != 0))), DType::Bool)?,
            other => {
                return Err(Error::NotImplemented(format!(
                    "loading tensor {name:?} of safetensors dtype {other:?} (grilly loads F32, BF16 and F16 \
                     as float32, and I64, I32 and BOOL)"
                ))
                .into())
            }
        })
    }
}

/// 32-bit words as little-endian bytes.
fn words(values: impl ExactSizeIterator<Item = u32>) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for w in values {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}
