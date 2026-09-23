//! Bindings for grilly-vsa. The Python-facing API is python/grilly/vsa/.

use pyo3::prelude::*;

use crate::{err, wrap, PyTensor};

/// Per-block circular convolution of two block codes (bind), or its
/// inverse (unbind).
#[pyfunction]
#[pyo3(signature = (a, b, unbind = false))]
fn blockcode_bind(py: Python<'_>, a: &PyTensor, b: &PyTensor, unbind: bool) -> PyResult<PyTensor> {
    py.detach(|| if unbind { grilly_vsa::unbind(&a.inner, &b.inner) } else { grilly_vsa::bind(&a.inner, &b.inner) })
        .map(wrap)
        .map_err(err)
}

/// Hamming distance from a packed query to every row of a packed codebook.
#[pyfunction]
fn hdc_hamming(py: Python<'_>, query: &PyTensor, codebook: &PyTensor) -> PyResult<PyTensor> {
    py.detach(|| grilly_vsa::packed::hamming(&query.inner, &codebook.inner)).map(wrap).map_err(err)
}

/// Bit-wise majority vote over the rows of a packed (n, words) tensor.
#[pyfunction]
fn hdc_bundle(py: Python<'_>, vectors: &PyTensor) -> PyResult<PyTensor> {
    py.detach(|| grilly_vsa::packed::bundle(&vectors.inner)).map(wrap).map_err(err)
}

/// Cyclic rotation of a packed hypervector's bits.
#[pyfunction]
fn hdc_permute(py: Python<'_>, x: &PyTensor, dim: usize, shift: i64) -> PyResult<PyTensor> {
    py.detach(|| grilly_vsa::packed::permute(&x.inner, dim, shift)).map(wrap).map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blockcode_bind, m)?)?;
    m.add_function(wrap_pyfunction!(hdc_hamming, m)?)?;
    m.add_function(wrap_pyfunction!(hdc_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(hdc_permute, m)?)?;
    Ok(())
}
