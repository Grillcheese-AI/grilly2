use std::sync::Arc;

use crate::device::{Buffer, Device, MemClass};
use crate::error::{Error, Result};

/// A float32 tensor in GPU memory: a shape over a contiguous row-major run of
/// elements in a shared buffer. Cloning shares the storage, as a view does.
///
/// Strided views and other dtypes are not implemented yet; every tensor is
/// contiguous float32.
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Buffer>,
    shape: Vec<usize>,
    /// Offset of the first element, in elements.
    offset: usize,
}

const ELEM: usize = std::mem::size_of::<f32>();

fn numel_of(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| {
        Error::Shape(format!("shape {shape:?} has too many elements"))
    })
}

fn as_bytes(data: &[f32]) -> &[u8] {
    // f32 has no padding or invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

impl Tensor {
    /// A tensor with undefined contents.
    pub fn empty(device: &Device, shape: &[usize]) -> Result<Tensor> {
        let bytes = numel_of(shape)?
            .checked_mul(ELEM)
            .ok_or_else(|| Error::Shape(format!("shape {shape:?} is too large")))?;
        // Zero-element tensors still get a minimal allocation.
        let storage = device.alloc(bytes.max(ELEM) as u64, MemClass::Device)?;
        Ok(Tensor { storage: Arc::new(storage), shape: shape.to_vec(), offset: 0 })
    }

    /// Copy `data` (row-major) to a new tensor of the given shape.
    pub fn from_slice(device: &Device, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let n = numel_of(shape)?;
        if n != data.len() {
            return Err(Error::Shape(format!(
                "shape {shape:?} needs {n} elements, got {}",
                data.len()
            )));
        }
        let t = Tensor::empty(device, shape)?;
        if n > 0 {
            t.storage.write(0, as_bytes(data))?;
        }
        Ok(t)
    }

    pub fn zeros(device: &Device, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_slice(device, &vec![0.0; numel_of(shape)?], shape)
    }

    /// Copy the elements back to the host (row-major). Waits for pending work.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.numel()];
        if !out.is_empty() {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * ELEM)
            };
            self.storage.read(self.byte_offset(), bytes)?;
        }
        Ok(out)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn device(&self) -> &Device {
        self.storage.device()
    }

    /// The same elements under a new shape with the same element count.
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        if numel_of(shape)? != self.numel() {
            return Err(Error::Shape(format!(
                "cannot reshape {:?} ({} elements) to {shape:?}",
                self.shape,
                self.numel()
            )));
        }
        Ok(Tensor { storage: self.storage.clone(), shape: shape.to_vec(), offset: self.offset })
    }

    pub(crate) fn storage(&self) -> &Buffer {
        &self.storage
    }

    pub(crate) fn byte_offset(&self) -> u64 {
        (self.offset * ELEM) as u64
    }

    pub(crate) fn byte_len(&self) -> u64 {
        (self.numel() * ELEM) as u64
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?}, device={:?})", self.shape, self.device())
    }
}
