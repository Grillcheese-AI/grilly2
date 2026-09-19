//! `grilly._C`: the native half of the Python package. Deliberately small —
//! a tensor handle, the ops, autograd entry points and the generator. The
//! torch-shaped surface (argument parsing, dtypes, reprs) lives in the
//! Python package above it.

// `_C` mirrors torch's native module name.
#![allow(non_snake_case)]

mod infer;
mod ops;

use std::cell::RefCell;
use std::rc::Rc;

use grilly_core::autograd::{PackedTensor, SavedTensorHooks};
use grilly_core::{autograd, rng, DType, Device, Error, MemClass, Tensor};
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

thread_local! {
    /// A Python exception raised inside a callback the core invoked (a
    /// saved-tensor hook). The core sees a plain Error; `err` re-raises the
    /// original so its type (e.g. CheckpointError) reaches the caller.
    static PENDING: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

fn from_py(e: PyErr) -> Error {
    let msg = e.to_string();
    PENDING.with(|p| *p.borrow_mut() = Some(e));
    Error::Invalid(format!("Python exception in a saved-tensor hook: {msg}"))
}

fn err(e: Error) -> PyErr {
    if let Some(original) = PENDING.with(|p| p.borrow_mut().take()) {
        return original;
    }
    match e {
        Error::NotImplemented(_) => PyNotImplementedError::new_err(e.to_string()),
        Error::Index(_) => PyIndexError::new_err(e.to_string()),
        // torch raises RuntimeError for shape, argument and device failures.
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn device() -> PyResult<Device> {
    Device::default_device().map_err(err)
}

/// A float32 tensor in GPU memory.
#[pyclass(name = "Tensor", module = "grilly._C", frozen)]
struct PyTensor {
    inner: Tensor,
}

fn wrap(inner: Tensor) -> PyTensor {
    PyTensor { inner }
}

#[pymethods]
impl PyTensor {
    /// Build from the elements as stored (little-endian; bool as u32 0/1),
    /// row-major; `dtype` is a DType code.
    #[staticmethod]
    #[pyo3(signature = (data, shape, dtype=0))]
    fn from_bytes(py: Python<'_>, data: &[u8], shape: Vec<usize>, dtype: u32) -> PyResult<PyTensor> {
        let (dev, dtype) = (device()?, DType::from_code(dtype).map_err(err)?);
        py.detach(|| Tensor::from_bytes(&dev, data, &shape, dtype)).map(wrap).map_err(err)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=0, persistent=false))]
    fn empty(py: Python<'_>, shape: Vec<usize>, dtype: u32, persistent: bool) -> PyResult<PyTensor> {
        let (dev, dtype) = (device()?, DType::from_code(dtype).map_err(err)?);
        let mem = if persistent { MemClass::Persistent } else { MemClass::Device };
        py.detach(|| Tensor::empty_in(&dev, &shape, dtype, mem)).map(wrap).map_err(err)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, value, dtype=0))]
    fn full(py: Python<'_>, shape: Vec<usize>, value: f64, dtype: u32) -> PyResult<PyTensor> {
        let (dev, dtype) = (device()?, DType::from_code(dtype).map_err(err)?);
        py.detach(|| Tensor::full_typed(&dev, &shape, value, dtype)).map(wrap).map_err(err)
    }

    /// The DType code.
    #[getter]
    fn dtype(&self) -> u32 {
        self.inner.dtype().code()
    }

    fn cast(&self, py: Python<'_>, dtype: u32) -> PyResult<PyTensor> {
        let (x, dtype) = (self.inner.clone(), DType::from_code(dtype).map_err(err)?);
        py.detach(move || x.cast(dtype)).map(wrap).map_err(err)
    }

    /// The elements as stored (little-endian; bool as u32 0/1), row-major.
    /// Waits for pending GPU work.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.inner.to_bytes()).map_err(err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        self.inner.reshape(&shape).map(wrap).map_err(err)
    }

    fn clone_data(&self, py: Python<'_>) -> PyResult<PyTensor> {
        py.detach(|| self.inner.clone_data()).map(wrap).map_err(err)
    }

    fn detach(&self) -> PyTensor {
        wrap(self.inner.detach())
    }

    /// Whether two handles are the same tensor (share autograd state).
    fn same_tensor(&self, other: &PyTensor) -> bool {
        self.inner.same_tensor(&other.inner)
    }

    // autograd
    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    fn set_requires_grad(&self, value: bool) -> PyResult<()> {
        self.inner.set_requires_grad(value).map_err(err)
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    #[getter]
    fn grad_fn_name(&self) -> Option<&'static str> {
        self.inner.grad_fn_name()
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(wrap)
    }

    fn set_grad(&self, grad: Option<&PyTensor>) -> PyResult<()> {
        self.inner.set_grad(grad.map(|g| g.inner.clone())).map_err(err)
    }

    #[pyo3(signature = (gradient=None, retain_graph=false))]
    fn backward(&self, py: Python<'_>, gradient: Option<&PyTensor>, retain_graph: bool) -> PyResult<()> {
        py.detach(|| self.inner.backward(gradient.map(|g| &g.inner), retain_graph)).map_err(err)
    }

    #[getter]
    fn version(&self) -> u64 {
        self.inner.version()
    }

    // in place
    fn add_(&self, py: Python<'_>, other: &PyTensor) -> PyResult<()> {
        py.detach(|| self.inner.add_(&other.inner).map(|_| ())).map_err(err)
    }

    fn fill_(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        py.detach(|| self.inner.fill_(value).map(|_| ())).map_err(err)
    }

    fn copy_(&self, py: Python<'_>, src: &PyTensor) -> PyResult<()> {
        py.detach(|| self.inner.copy_(&src.inner).map(|_| ())).map_err(err)
    }

    fn mul_(&self, value: f32) -> PyResult<()> {
        self.inner.mul_(value).map(|_| ()).map_err(err)
    }

    fn normal_(&self, py: Python<'_>, mean: f32, std: f32, generator: &Bound<'_, PyGenerator>) -> PyResult<()> {
        let mut g = generator.borrow_mut();
        let (inner, gen) = (&self.inner, &mut g.inner);
        py.detach(|| inner.normal_(mean, std, gen).map(|_| ())).map_err(err)
    }

    fn uniform_(&self, py: Python<'_>, low: f32, high: f32, generator: &Bound<'_, PyGenerator>) -> PyResult<()> {
        let mut g = generator.borrow_mut();
        let (inner, gen) = (&self.inner, &mut g.inner);
        py.detach(|| inner.uniform_(low, high, gen).map(|_| ())).map_err(err)
    }
}

// ── saved-tensor hooks backed by Python callables ──────────────────────────

struct PyHooks {
    pack: Py<PyAny>,
    unpack: Py<PyAny>,
}

struct PyPacked {
    obj: Py<PyAny>,
    unpack: Py<PyAny>,
}

impl SavedTensorHooks for PyHooks {
    fn pack(&self, t: &Tensor) -> grilly_core::Result<Box<dyn PackedTensor>> {
        Python::attach(|py| {
            let obj = self.pack.call1(py, (wrap(t.clone()),)).map_err(from_py)?;
            Ok(Box::new(PyPacked { obj, unpack: self.unpack.clone_ref(py) }) as Box<dyn PackedTensor>)
        })
    }
}

impl PackedTensor for PyPacked {
    fn unpack(&mut self) -> grilly_core::Result<Tensor> {
        Python::attach(|py| {
            let out = self.unpack.call1(py, (self.obj.clone_ref(py),)).map_err(from_py)?;
            let t = out.bind(py).cast::<PyTensor>().map_err(|e| from_py(PyErr::from(e)))?;
            Ok(t.get().inner.clone())
        })
    }
}

/// Install saved-tensor hooks on this thread: `pack(_C.Tensor) -> object`,
/// `unpack(object) -> _C.Tensor`.
#[pyfunction]
fn push_saved_tensor_hooks(pack: Py<PyAny>, unpack: Py<PyAny>) {
    autograd::push_saved_tensor_hooks(Rc::new(PyHooks { pack, unpack }));
}

#[pyfunction]
fn pop_saved_tensor_hooks() -> PyResult<()> {
    autograd::pop_saved_tensor_hooks().map_err(err)
}

/// A seeded random stream; see grilly_core::rng for the exact algorithm.
#[pyclass(name = "Generator", module = "grilly._C")]
struct PyGenerator {
    inner: rng::Generator,
}

#[pymethods]
impl PyGenerator {
    #[new]
    fn new(seed: u64) -> PyGenerator {
        PyGenerator { inner: rng::Generator::new(seed) }
    }

    fn manual_seed(&mut self, seed: u64) {
        self.inner.manual_seed(seed);
    }

    fn initial_seed(&self) -> u64 {
        self.inner.initial_seed()
    }

    fn get_state_pair(&self) -> (u64, u64) {
        self.inner.state()
    }

    fn set_state_pair(&mut self, state: (u64, u64)) {
        self.inner.set_state(state);
    }

    /// Standard-normal float32 tensor of the given shape.
    fn normal(&mut self, py: Python<'_>, shape: Vec<usize>) -> PyResult<PyTensor> {
        let n = shape.iter().product();
        let values = self.inner.normal(n);
        let dev = device()?;
        py.detach(|| Tensor::from_slice(&dev, &values, &shape)).map(wrap).map_err(err)
    }

    /// int64 tensor uniform in [low, high).
    fn integers(&mut self, py: Python<'_>, shape: Vec<usize>, low: i64, high: i64) -> PyResult<PyTensor> {
        let n = shape.iter().product();
        let bytes: Vec<u8> = self.inner.integers(n, low, high).iter().flat_map(|v| v.to_le_bytes()).collect();
        let dev = device()?;
        py.detach(|| Tensor::from_bytes(&dev, &bytes, &shape, DType::I64)).map(wrap).map_err(err)
    }

    /// Float32 tensor uniform in [0, 1).
    fn uniform(&mut self, py: Python<'_>, shape: Vec<usize>) -> PyResult<PyTensor> {
        let n = shape.iter().product();
        let values = self.inner.uniform(n);
        let dev = device()?;
        py.detach(|| Tensor::from_slice(&dev, &values, &shape)).map(wrap).map_err(err)
    }
}

#[pyfunction]
fn relu(py: Python<'_>, input: &PyTensor) -> PyResult<PyTensor> {
    py.detach(|| grilly_torch::relu(&input.inner)).map(wrap).map_err(err)
}

#[pyfunction]
fn add(py: Python<'_>, input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    py.detach(|| grilly_torch::add(&input.inner, &other.inner)).map(wrap).map_err(err)
}

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None))]
fn linear(py: Python<'_>, input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    py.detach(|| grilly_torch::linear(&input.inner, &weight.inner, bias.map(|b| &b.inner)))
        .map(wrap)
        .map_err(err)
}

/// Grad mode of the calling thread (`torch.is_grad_enabled`).
#[pyfunction]
fn is_grad_enabled() -> bool {
    autograd::is_grad_enabled()
}

/// Set this thread's grad mode; returns the previous mode.
#[pyfunction]
fn swap_grad_mode(enabled: bool) -> bool {
    autograd::swap_grad_mode(enabled)
}

/// Name of the GPU the default device opened.
#[pyfunction]
fn device_name() -> PyResult<String> {
    Ok(device()?.caps().name.clone())
}

/// Count of fused kernels (crate::fusion, ops::fused) compiled so far on
/// the default device: for tests confirming a `torch.compile`d graph's
/// structure, not its scalar values, is what decides whether a new kernel
/// is generated.
#[pyfunction]
fn generated_kernel_count() -> PyResult<usize> {
    Ok(device()?.generated_kernel_count())
}

#[pymodule]
fn _C(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyGenerator>()?;
    m.add("DEFAULT_SEED", rng::DEFAULT_SEED)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(swap_grad_mode, m)?)?;
    m.add_function(wrap_pyfunction!(push_saved_tensor_hooks, m)?)?;
    m.add_function(wrap_pyfunction!(pop_saved_tensor_hooks, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(generated_kernel_count, m)?)?;
    ops::register(m)?;
    infer::register(m)?;
    Ok(())
}
