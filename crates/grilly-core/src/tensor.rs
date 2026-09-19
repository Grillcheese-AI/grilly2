use std::sync::Arc;

use crate::autograd::{self, Backward, Meta};
use crate::device::{Buffer, Device, MemClass};
use crate::dtype::{require_float, DType};
use crate::error::{Error, Result};
use crate::ops;
use crate::rng::Generator;

/// A tensor in GPU memory: a shape over a contiguous row-major run of
/// elements of one dtype in a shared buffer, plus its autograd state.
///
/// Cloning a `Tensor` makes another handle to the *same* tensor (same data,
/// same `requires_grad` and `.grad`), like a second Python reference in torch.
/// [`Tensor::clone_data`] is torch's `clone()`.
///
/// Strided views are not implemented yet; every tensor is contiguous.
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Buffer>,
    shape: Vec<usize>,
    /// Offset of the first element, in elements.
    offset: usize,
    dtype: DType,
    meta: Arc<Meta>,
    /// A copy standing in for what torch returns as a view (transpose,
    /// expand, slicing). Writing through it would not reach the original, so
    /// in-place ops on it are refused rather than silently diverging.
    view_copy: bool,
}

fn numel_of(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| {
        Error::Shape(format!("shape {shape:?} has too many elements"))
    })
}

fn as_bytes(data: &[f32]) -> &[u8] {
    // f32 has no padding or invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

struct ViewBackward {
    input_shape: Vec<usize>,
}

impl Backward for ViewBackward {
    fn apply(&mut self, grads: &[Option<Tensor>], _: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let g = grads[0].as_ref().expect("single output");
        Ok(vec![Some(g.reshape(&self.input_shape)?)])
    }
}

struct CloneBackward;

impl Backward for CloneBackward {
    fn apply(&mut self, grads: &[Option<Tensor>], _: &[bool]) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![grads[0].clone()])
    }
}

impl Tensor {
    fn from_parts(storage: Arc<Buffer>, shape: Vec<usize>, offset: usize, dtype: DType) -> Tensor {
        Tensor { storage, shape, offset, dtype, meta: Arc::new(Meta::leaf()), view_copy: false }
    }

    /// A float32 tensor with undefined contents.
    pub fn empty(device: &Device, shape: &[usize]) -> Result<Tensor> {
        Tensor::empty_typed(device, shape, DType::F32)
    }

    /// A tensor of `dtype` with undefined contents.
    pub fn empty_typed(device: &Device, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let bytes = numel_of(shape)?
            .checked_mul(dtype.size())
            .ok_or_else(|| Error::Shape(format!("shape {shape:?} is too large")))?;
        // Zero-element tensors still get a minimal allocation.
        let storage = device.alloc(bytes.max(8) as u64, MemClass::Device)?;
        Ok(Tensor::from_parts(Arc::new(storage), shape.to_vec(), 0, dtype))
    }

    /// Copy float32 `data` (row-major) to a new tensor of the given shape.
    pub fn from_slice(device: &Device, data: &[f32], shape: &[usize]) -> Result<Tensor> {
        Tensor::from_bytes(device, as_bytes(data), shape, DType::F32)
    }

    /// A new tensor from its elements as stored (little-endian; bool as one
    /// u32 0/1 per element), row-major.
    pub fn from_bytes(device: &Device, data: &[u8], shape: &[usize], dtype: DType) -> Result<Tensor> {
        let n = numel_of(shape)?;
        if data.len() != n * dtype.size() {
            return Err(Error::Shape(format!(
                "shape {shape:?} of {} needs {} bytes, got {}",
                dtype.name(),
                n * dtype.size(),
                data.len()
            )));
        }
        let t = Tensor::empty_typed(device, shape, dtype)?;
        if n > 0 {
            t.storage.write(0, data)?;
        }
        Ok(t)
    }

    /// Every element `value`, as float32, filled on the device.
    pub fn full(device: &Device, shape: &[usize], value: f32) -> Result<Tensor> {
        let t = Tensor::empty(device, shape)?;
        device.record_fill(&t.storage, t.byte_offset(), t.byte_len(), value.to_bits())?;
        Ok(t)
    }

    /// Every element `value` converted to `dtype` (torch's `full(..., dtype=)`).
    pub fn full_typed(device: &Device, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor> {
        let t = Tensor::empty_typed(device, shape, dtype)?;
        t.fill_raw(value)?;
        Ok(t)
    }

    pub fn zeros(device: &Device, shape: &[usize]) -> Result<Tensor> {
        Tensor::full(device, shape, 0.0)
    }

    /// Write `value` to every element, without version or autograd checks.
    fn fill_raw(&self, value: f64) -> Result<()> {
        let pattern = self.dtype.encode(value);
        if pattern.len() == 4 {
            let word = u32::from_le_bytes(pattern.try_into().unwrap());
            return self.device().record_fill(&self.storage, self.byte_offset(), self.byte_len(), word);
        }
        // A multi-word element: broadcast one stored element with a strided copy.
        let one = Tensor::from_bytes(self.device(), &pattern, &[], self.dtype)?;
        let zero_strides = vec![0; self.dim()];
        let strides = crate::generic::contiguous_strides(&self.shape);
        crate::generic::copy_strided(&one, 0, &zero_strides, self, self.offset, &strides, &self.shape)
    }

    /// A copy of the data in new storage (torch's `clone()`); differentiable.
    pub fn clone_data(&self) -> Result<Tensor> {
        let mut out = Tensor::empty_typed(self.device(), &self.shape, self.dtype)?;
        self.device().record_copy(&self.storage, self.byte_offset(), &out.storage, 0, self.byte_len())?;
        autograd::record("CloneBackward0", &[self], &mut [&mut out], |_| Ok(CloneBackward))?;
        Ok(out)
    }

    /// Whether `other` is a handle to this same tensor (not just equal data).
    pub fn same_tensor(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.meta, &other.meta)
    }

    /// The same data, cut off from the graph (torch's `detach()`).
    pub fn detach(&self) -> Tensor {
        Tensor::from_parts(self.storage.clone(), self.shape.clone(), self.offset, self.dtype)
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The elements converted to `dtype` (torch's `.to(dtype)`): this tensor
    /// when the dtype already matches, otherwise a new one. Conversions are
    /// not differentiable (only float32 can require grad).
    pub fn cast(&self, dtype: DType) -> Result<Tensor> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }
        crate::generic::cast(self, dtype)
    }

    /// Copy float32 elements back to the host (row-major). Waits for pending work.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        require_float("to_vec", self.dtype)?;
        let bytes = self.to_bytes()?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    /// The elements as stored (see [`Tensor::from_bytes`]). Waits for pending work.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.numel() * self.dtype.size()];
        if !out.is_empty() {
            self.storage.read(self.byte_offset(), &mut out)?;
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
        let mut out = Tensor::from_parts(self.storage.clone(), shape.to_vec(), self.offset, self.dtype);
        out.view_copy = self.view_copy;
        let input_shape = self.shape.clone();
        autograd::record("ViewBackward0", &[self], &mut [&mut out], |_| Ok(ViewBackward { input_shape }))?;
        Ok(out)
    }

    // ── autograd ─────────────────────────────────────────────────────────

    pub fn requires_grad(&self) -> bool {
        autograd::requires_grad(&self.meta)
    }

    /// torch's `requires_grad_()`: only leaves can change it.
    pub fn set_requires_grad(&self, value: bool) -> Result<()> {
        if value && !self.dtype.is_floating() {
            return Err(Error::Invalid("only Tensors of floating point and complex dtype can require gradients".into()));
        }
        autograd::set_requires_grad(&self.meta, value)
    }

    pub fn is_leaf(&self) -> bool {
        autograd::is_leaf(&self.meta)
    }

    /// Name of the node that produced this tensor, as torch prints `grad_fn`.
    pub fn grad_fn_name(&self) -> Option<&'static str> {
        autograd::grad_fn_name(&self.meta)
    }

    /// The accumulated gradient (`.grad`); set on leaves by `backward`.
    pub fn grad(&self) -> Option<Tensor> {
        autograd::grad(&self.meta)
    }

    /// Replace `.grad`; `None` is what `zero_grad(set_to_none=True)` does.
    pub fn set_grad(&self, grad: Option<Tensor>) -> Result<()> {
        if let Some(g) = &grad {
            if g.shape() != self.shape() {
                return Err(Error::Shape(format!(
                    "assigned grad has data of shape {:?} but the tensor has shape {:?}",
                    g.shape(),
                    self.shape()
                )));
            }
        }
        autograd::set_grad(&self.meta, grad);
        Ok(())
    }

    /// torch's `backward(gradient=None, retain_graph=False)`.
    pub fn backward(&self, gradient: Option<&Tensor>, retain_graph: bool) -> Result<()> {
        autograd::backward(self, gradient, retain_graph)
    }

    // ── in place ─────────────────────────────────────────────────────────
    //
    // torch's rules: outside no_grad, an in-place op on a leaf that requires
    // grad is an error. In-place ops on tensors inside a graph (which torch
    // handles by rebasing their history) are not implemented yet. Every
    // in-place op bumps the storage version, which saved tensors check.

    /// How many in-place writes this tensor's storage has seen (`_version`).
    pub fn version(&self) -> u64 {
        self.storage.version()
    }

    /// Mark this tensor as a copy standing in for a torch view.
    pub fn mark_view_copy(&mut self) {
        self.view_copy = true;
    }

    /// Whether this tensor is a copy standing in for a torch view.
    pub fn is_view_copy(&self) -> bool {
        self.view_copy
    }

    /// torch's in-place rules, plus refusing writes through view copies.
    pub fn check_inplace(&self, operands: &[&Tensor]) -> Result<()> {
        if self.view_copy {
            return Err(Error::NotImplemented(
                "in-place writes through a transposed, expanded or sliced tensor (grilly returns \
                 a copy where torch returns a view; call .clone() or .contiguous() first)"
                    .into(),
            ));
        }
        if !autograd::is_grad_enabled() {
            return Ok(());
        }
        if self.requires_grad() {
            return Err(if self.is_leaf() {
                Error::Invalid("a leaf Variable that requires grad is being used in an in-place operation.".into())
            } else {
                Error::NotImplemented("in-place operations on tensors that are part of an autograd graph".into())
            });
        }
        if operands.iter().any(|t| t.requires_grad()) {
            return Err(Error::NotImplemented("in-place operations with an operand that requires grad".into()));
        }
        Ok(())
    }

    /// `self += other` (torch's `add_`).
    pub fn add_(&self, other: &Tensor) -> Result<&Tensor> {
        self.check_inplace(&[other])?;
        ops::add_into(self, other)?;
        Ok(self)
    }

    /// Every element set to `value` (torch's `fill_`).
    pub fn fill_(&self, value: f64) -> Result<&Tensor> {
        self.check_inplace(&[])?;
        self.fill_raw(value)?;
        self.bump_version();
        Ok(self)
    }

    /// torch's `zero_`.
    pub fn zero_(&self) -> Result<&Tensor> {
        self.fill_(0.0)
    }

    /// Copy `src`'s elements into this tensor (torch's `copy_`, same shape).
    pub fn copy_(&self, src: &Tensor) -> Result<&Tensor> {
        if src.shape() != self.shape() {
            return Err(Error::NotImplemented(format!(
                "copy_ between shapes {:?} and {:?} (broadcasting)",
                src.shape(),
                self.shape()
            )));
        }
        self.check_inplace(&[src])?;
        let src = src.cast(self.dtype)?; // torch converts to the destination's dtype
        self.device().record_copy(&src.storage, src.byte_offset(), &self.storage, self.byte_offset(), self.byte_len())?;
        self.bump_version();
        Ok(self)
    }

    /// Overwrite with host-computed values, in place.
    fn overwrite(&self, values: &[f32]) -> Result<&Tensor> {
        let src = Tensor::from_slice(self.device(), values, &self.shape)?;
        self.device().record_copy(&src.storage, 0, &self.storage, self.byte_offset(), self.byte_len())?;
        self.bump_version();
        Ok(self)
    }

    /// Samples from N(mean, std^2) (torch's `normal_`); deterministic per
    /// generator, not torch's values.
    pub fn normal_(&self, mean: f32, std: f32, generator: &mut Generator) -> Result<&Tensor> {
        require_float("normal_", self.dtype)?;
        self.check_inplace(&[])?;
        let values: Vec<f32> = generator.normal(self.numel()).into_iter().map(|z| mean + std * z).collect();
        self.overwrite(&values)
    }

    /// Samples uniform in [from, to) (torch's `uniform_`).
    pub fn uniform_(&self, from: f32, to: f32, generator: &mut Generator) -> Result<&Tensor> {
        require_float("uniform_", self.dtype)?;
        self.check_inplace(&[])?;
        let values: Vec<f32> = generator.uniform(self.numel()).into_iter().map(|u| from + (to - from) * u).collect();
        self.overwrite(&values)
    }

    /// `self *= value` (torch's `mul_` with a scalar).
    pub fn mul_(&self, value: f32) -> Result<&Tensor> {
        require_float("mul_", self.dtype)?;
        self.check_inplace(&[])?;
        crate::generic::unary_inplace(self, crate::generic::Unary::MulScalar, value, 0.0)?;
        Ok(self)
    }

    // ── crate internals ─────────────────────────────────────────────────

    pub(crate) fn bump_version(&self) {
        self.storage.bump_version();
    }

    pub(crate) fn meta(&self) -> &Arc<Meta> {
        &self.meta
    }

    pub(crate) fn set_meta(&mut self, meta: Arc<Meta>) {
        self.meta = meta;
    }

    /// Nothing but this handle can reach this data.
    pub(crate) fn is_sole_owner(&self) -> bool {
        Arc::strong_count(&self.storage) == 1 && Arc::strong_count(&self.meta) == 1
    }

    /// Offset of the first element within the storage, in elements; for
    /// kernels bound with [`crate::Arg::Storage`].
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn storage(&self) -> &Buffer {
        &self.storage
    }

    pub(crate) fn byte_offset(&self) -> u64 {
        (self.offset * self.dtype.size()) as u64
    }

    pub(crate) fn byte_len(&self) -> u64 {
        (self.numel() * self.dtype.size()) as u64
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?}, device={:?}", self.shape, self.device())?;
        match self.grad_fn_name() {
            Some(name) => write!(f, ", grad_fn={name})"),
            None if self.requires_grad() => write!(f, ", requires_grad=True)"),
            None => write!(f, ")"),
        }
    }
}
