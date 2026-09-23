//! Bindings for grilly-snn. The Python-facing API is python/grilly/snn/.

use grilly_snn::lif::{self, Reset, StepConfig, Surrogate};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{err, wrap, PyTensor};

fn surrogate(kind: &str) -> PyResult<Surrogate> {
    match kind {
        "atan" => Ok(Surrogate::ATan),
        "sigmoid" => Ok(Surrogate::Sigmoid),
        "fast_sigmoid" => Ok(Surrogate::FastSigmoid),
        other => Err(PyValueError::new_err(format!(
            "unknown surrogate {other:?}; the fused step has atan, sigmoid and fast_sigmoid"
        ))),
    }
}

/// One integrate-and-fire timestep: charge, fire and reset in one dispatch.
/// Returns `(spikes, next membrane)`.
#[pyfunction]
#[pyo3(signature = (v, x, decay, in_scale, rest, threshold, soft, detach_reset, kind, alpha))]
#[allow(clippy::too_many_arguments)]
fn snn_lif_step(
    py: Python<'_>,
    v: &PyTensor,
    x: &PyTensor,
    decay: f32,
    in_scale: f32,
    rest: f32,
    threshold: f32,
    soft: bool,
    detach_reset: bool,
    kind: &str,
    alpha: f32,
) -> PyResult<(PyTensor, PyTensor)> {
    let config = StepConfig {
        decay,
        in_scale,
        rest,
        threshold,
        reset: if soft { Reset::Soft } else { Reset::Hard },
        detach_reset,
        surrogate: surrogate(kind)?,
        alpha,
    };
    py.detach(|| lif::step(&v.inner, &x.inner, config))
        .map(|(s, n)| (wrap(s), wrap(n)))
        .map_err(err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(snn_lif_step, m)?)?;
    Ok(())
}
