//! Bindings for grilly-infer. The torch-facing API, with safetensors.torch's
//! names, is python/grilly/infer/safetensors.py.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use grilly_infer::{Int8Weight, LoadError, SafeTensorsFile};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{device, err, wrap, PyTensor};

create_exception!(
    _C,
    SafetensorError,
    PyException,
    "A file is not valid safetensors, or has no tensor of the name asked for (the safetensors package's SafetensorError)."
);

fn load_err(e: LoadError, path: Option<&Path>) -> PyErr {
    let path = path.map(|p| p.display().to_string()).unwrap_or_default();
    match e {
        LoadError::Io(e) if e.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(format!("No such file or directory: {path}"))
        }
        LoadError::Io(e) => PyOSError::new_err(format!("{path}: {e}")),
        LoadError::Format(m) => SafetensorError::new_err(m),
        LoadError::Tensor(e) => err(e),
    }
}

/// An open safetensors file: header parsed, data memory-mapped.
#[pyclass(name = "SafeFile", module = "grilly._C", frozen)]
struct PySafeFile {
    inner: SafeTensorsFile,
}

#[pymethods]
impl PySafeFile {
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<PySafeFile> {
        py.detach(|| SafeTensorsFile::open(&path))
            .map(|inner| PySafeFile { inner })
            .map_err(|e| load_err(e, Some(&path)))
    }

    /// A whole file's bytes (copied).
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<PySafeFile> {
        let data = data.to_vec();
        py.detach(|| SafeTensorsFile::from_bytes(data)).map(|inner| PySafeFile { inner }).map_err(|e| load_err(e, None))
    }

    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }

    fn offset_keys(&self) -> Vec<String> {
        self.inner.offset_keys()
    }

    fn metadata(&self) -> Option<HashMap<String, String>> {
        self.inner.metadata()
    }

    /// Upload one tensor to the default device.
    fn get(&self, py: Python<'_>, name: &str) -> PyResult<PyTensor> {
        let dev = device()?;
        py.detach(|| self.inner.tensor(&dev, name)).map(wrap).map_err(|e| load_err(e, None))
    }
}

/// A linear weight quantized to int8 per output row (grilly_infer::Int8Weight).
#[pyclass(name = "Int8Weight", module = "grilly._C", frozen)]
struct PyInt8Weight {
    inner: Int8Weight,
}

#[pymethods]
impl PyInt8Weight {
    #[staticmethod]
    fn quantize(py: Python<'_>, weight: &PyTensor) -> PyResult<PyInt8Weight> {
        py.detach(|| Int8Weight::quantize(&weight.inner)).map(|inner| PyInt8Weight { inner }).map_err(err)
    }

    /// (out_features, in_features)
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Device bytes held.
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    fn scale(&self) -> PyTensor {
        wrap(self.inner.scale().clone())
    }

    /// The int8 values, row-major, as bytes.
    fn qdata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let q = py.detach(|| self.inner.qdata()).map_err(err)?;
        Ok(PyBytes::new(py, &q.iter().map(|&v| v as u8).collect::<Vec<u8>>()))
    }

    fn dequantize(&self, py: Python<'_>) -> PyResult<PyTensor> {
        py.detach(|| self.inner.dequantize()).map(wrap).map_err(err)
    }

    #[pyo3(signature = (input, bias=None))]
    fn linear(&self, py: Python<'_>, input: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
        py.detach(|| self.inner.linear(&input.inner, bias.map(|b| &b.inner))).map(wrap).map_err(err)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySafeFile>()?;
    m.add_class::<PyInt8Weight>()?;
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    Ok(())
}
