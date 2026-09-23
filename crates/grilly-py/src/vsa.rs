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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blockcode_bind, m)?)?;
    Ok(())
}
