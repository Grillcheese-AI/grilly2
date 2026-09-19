//! Bindings for grilly-infer. The torch-facing API, with safetensors.torch's
//! names, is python/grilly/infer/safetensors.py.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use grilly_infer::{ops, sampling, Heads, Int8Weight, KvCache, LoadError, Rotary, SafeTensorsFile, Sampling};
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

    /// One weight holding several weights' rows: a fused q/k/v or gate/up
    /// projection, computing each row exactly as the separate ones did.
    #[staticmethod]
    fn cat(py: Python<'_>, parts: Vec<PyRef<'_, PyInt8Weight>>) -> PyResult<PyInt8Weight> {
        let parts: Vec<&Int8Weight> = parts.iter().map(|p| &p.inner).collect();
        py.detach(|| Int8Weight::cat(&parts)).map(|inner| PyInt8Weight { inner }).map_err(err)
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

/// Keys and values of every layer, on the device (grilly_infer::KvCache).
#[pyclass(name = "KVCache", module = "grilly._C")]
struct PyKvCache {
    inner: KvCache,
}

#[pymethods]
impl PyKvCache {
    #[new]
    fn new(py: Python<'_>, layers: usize, batch: usize, kv_heads: usize, head_dim: usize, max_len: usize) -> PyResult<PyKvCache> {
        let dev = device()?;
        py.detach(|| KvCache::new(&dev, layers, batch, kv_heads, head_dim, max_len)).map(|inner| PyKvCache { inner }).map_err(err)
    }

    fn append(&mut self, py: Python<'_>, layer: usize, keys: &PyTensor, values: &PyTensor) -> PyResult<()> {
        let inner = &mut self.inner;
        py.detach(|| inner.append(layer, &keys.inner, &values.inner)).map_err(err)
    }

    #[pyo3(signature = (layer, query, scale=None))]
    fn attend(&self, py: Python<'_>, layer: usize, query: &PyTensor, scale: Option<f32>) -> PyResult<PyTensor> {
        py.detach(|| self.inner.attend(layer, &query.inner, scale)).map(wrap).map_err(err)
    }

    /// Rotate a step's keys into the cache and copy its values, taking both
    /// from projections in their own (batch, T, row) layout. Two dispatches.
    #[allow(clippy::too_many_arguments)]
    fn append_rotary(
        &mut self,
        py: Python<'_>,
        layer: usize,
        keys: &PyTensor,
        k_col: usize,
        values: &PyTensor,
        v_col: usize,
        cos: &PyTensor,
        sin: &PyTensor,
        pos0: usize,
    ) -> PyResult<()> {
        let (heads, dim) = (self.inner.kv_heads(), self.inner.head_dim());
        let inner = &mut self.inner;
        py.detach(|| {
            inner.append_rotary(
                layer,
                &Heads { src: &keys.inner, col: k_col, heads, dim },
                &Heads { src: &values.inner, col: v_col, heads, dim },
                &Rotary { cos: &cos.inner, sin: &sin.inner, pos0 },
            )
        })
        .map_err(err)
    }

    fn len(&self, layer: usize) -> usize {
        self.inner.len(layer)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn layers(&self) -> usize {
        self.inner.layers()
    }

    #[getter]
    fn max_len(&self) -> usize {
        self.inner.max_len()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }
}

/// Rotary embedding of `heads` heads of `dim` from column `col` of a
/// (batch, T, row) projection, as (batch, heads, T, dim).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rope(
    py: Python<'_>,
    src: &PyTensor,
    col: usize,
    heads: usize,
    dim: usize,
    cos: &PyTensor,
    sin: &PyTensor,
    pos0: usize,
) -> PyResult<PyTensor> {
    py.detach(|| {
        ops::rope(
            &Heads { src: &src.inner, col, heads, dim },
            &Rotary { cos: &cos.inner, sin: &sin.inner, pos0 },
        )
    })
    .map(wrap)
    .map_err(err)
}

/// silu(gate) * up, each read from its source's columns.
#[pyfunction]
fn swiglu(py: Python<'_>, gate: &PyTensor, gate_col: usize, up: &PyTensor, up_col: usize, width: usize) -> PyResult<PyTensor> {
    py.detach(|| ops::swiglu(&gate.inner, gate_col, &up.inner, up_col, width)).map(wrap).map_err(err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySafeFile>()?;
    m.add_class::<PyInt8Weight>()?;
    m.add_class::<PyKvCache>()?;
    m.add_function(wrap_pyfunction!(next_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(processed_scores, m)?)?;
    m.add_function(wrap_pyfunction!(sample_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(multinomial, m)?)?;
    m.add_function(wrap_pyfunction!(rope, m)?)?;
    m.add_function(wrap_pyfunction!(swiglu, m)?)?;
    m.add("SafetensorError", m.py().get_type::<SafetensorError>())?;
    Ok(())
}
