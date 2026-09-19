//! Bindings for grilly-infer. The torch-facing API, with safetensors.torch's
//! names, is python/grilly/infer/safetensors.py.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use grilly_infer::{sampling, Int8Weight, LoadError, SafeTensorsFile, Sampling};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{device, err, wrap, PyGenerator, PyTensor};

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

/// Sampling parameters from Python, as Hugging Face names them.
fn sampling_of(do_sample: bool, temperature: f32, top_k: usize, top_p: f32, repetition_penalty: f32) -> Sampling {
    Sampling { do_sample, temperature, top_k, top_p, repetition_penalty }
}

/// The next token per row (grilly_infer::sampling::next_tokens).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn next_tokens(
    py: Python<'_>,
    logits: &PyTensor,
    input_ids: Option<&PyTensor>,
    do_sample: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    generator: &Bound<'_, PyGenerator>,
) -> PyResult<PyTensor> {
    let s = sampling_of(do_sample, temperature, top_k, top_p, repetition_penalty);
    let mut g = generator.borrow_mut();
    let gen = &mut g.inner;
    py.detach(|| sampling::next_tokens(&logits.inner, input_ids.map(|t| &t.inner), &s, gen)).map(wrap).map_err(err)
}

#[pyfunction]
fn processed_scores(
    py: Python<'_>,
    logits: &PyTensor,
    input_ids: Option<&PyTensor>,
    do_sample: bool,
    temperature: f32,
    repetition_penalty: f32,
) -> PyResult<PyTensor> {
    let s = sampling_of(do_sample, temperature, 0, 1.0, repetition_penalty);
    py.detach(|| sampling::processed_scores(&logits.inner, input_ids.map(|t| &t.inner), &s)).map(wrap).map_err(err)
}

#[pyfunction]
fn sample_probabilities(
    py: Python<'_>,
    logits: &PyTensor,
    input_ids: Option<&PyTensor>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
) -> PyResult<PyTensor> {
    let s = sampling_of(true, temperature, top_k, top_p, repetition_penalty);
    py.detach(|| sampling::probabilities(&logits.inner, input_ids.map(|t| &t.inner), &s)).map(wrap).map_err(err)
}

/// torch.multinomial(weights, 1) for (batch, n) weights, as int64 (batch,).
#[pyfunction]
fn multinomial(py: Python<'_>, weights: &PyTensor, generator: &Bound<'_, PyGenerator>) -> PyResult<PyTensor> {
    let mut g = generator.borrow_mut();
    let gen = &mut g.inner;
    py.detach(|| sampling::multinomial(&weights.inner, gen)).map(wrap).map_err(err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySafeFile>()?;
    m.add_class::<PyInt8Weight>()?;
    m.add_function(wrap_pyfunction!(next_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(processed_scores, m)?)?;
    m.add_function(wrap_pyfunction!(sample_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(multinomial, m)?)?;
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    Ok(())
}
