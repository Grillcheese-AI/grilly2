//! `grilly._C`: the native half of the Python package. Deliberately small —
//! a tensor handle and the ops. The torch-shaped surface (argument parsing,
//! dtypes, reprs) lives in the Python package above it.

// `_C` mirrors torch's native module name.
#![allow(non_snake_case)]

use grilly_core::{Device, Error, Tensor};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn err(e: Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
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
    /// Build from little-endian float32 bytes in row-major order.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8], shape: Vec<usize>) -> PyResult<PyTensor> {
        if !data.len().is_multiple_of(4) {
            return Err(PyRuntimeError::new_err("float32 data must be a multiple of 4 bytes"));
        }
        let values: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let dev = device()?;
        py.detach(|| Tensor::from_slice(&dev, &values, &shape)).map(wrap).map_err(err)
    }

    #[staticmethod]
    fn empty(py: Python<'_>, shape: Vec<usize>) -> PyResult<PyTensor> {
        let dev = device()?;
        py.detach(|| Tensor::empty(&dev, &shape)).map(wrap).map_err(err)
    }

    /// The elements as little-endian float32 bytes, row-major. Waits for
    /// pending GPU work.
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let values = py.detach(|| self.inner.to_vec()).map_err(err)?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(PyBytes::new(py, &bytes))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        self.inner.reshape(&shape).map(wrap).map_err(err)
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

/// Name of the GPU the default device opened.
#[pyfunction]
fn device_name() -> PyResult<String> {
    Ok(device()?.caps().name.clone())
}

#[pymodule]
fn _C(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    Ok(())
}
