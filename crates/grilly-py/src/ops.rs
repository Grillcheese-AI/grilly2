//! Bindings for grilly-torch's math and shape ops. Ops are named by string;
//! argument checking and torch's calling conventions live in Python.

use grilly_core::generic::{Binary, Unary};
use grilly_core::Tensor;
use grilly_torch::{math, shape};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{err, wrap, PyTensor};

fn run(py: Python<'_>, f: impl FnOnce() -> grilly_core::Result<Tensor> + Send) -> PyResult<PyTensor> {
    py.detach(f).map(wrap).map_err(err)
}

fn unary_op(name: &str) -> PyResult<Unary> {
    Ok(match name {
        "neg" => Unary::Neg,
        "abs" => Unary::Abs,
        "sign" => Unary::Sign,
        "exp" => Unary::Exp,
        "log" => Unary::Log,
        "sqrt" => Unary::Sqrt,
        "rsqrt" => Unary::Rsqrt,
        "tanh" => Unary::Tanh,
        "sigmoid" => Unary::Sigmoid,
        "add_scalar" => Unary::AddScalar,
        "mul_scalar" => Unary::MulScalar,
        "rsub_scalar" => Unary::RsubScalar,
        "rdiv_scalar" => Unary::RdivScalar,
        "pow_scalar" => Unary::PowScalar,
        "clamp" => Unary::Clamp,
        "gelu" => Unary::Gelu,
        _ => return Err(PyValueError::new_err(format!("unknown unary op {name:?}"))),
    })
}

fn binary_op(name: &str) -> PyResult<Binary> {
    Ok(match name {
        "add" => Binary::Add,
        "sub" => Binary::Sub,
        "mul" => Binary::Mul,
        "div" => Binary::Div,
        "pow" => Binary::Pow,
        "maximum" => Binary::Maximum,
        "minimum" => Binary::Minimum,
        _ => return Err(PyValueError::new_err(format!("unknown binary op {name:?}"))),
    })
}

#[pyfunction]
#[pyo3(signature = (x, op, a=0.0, b=0.0))]
fn unary(py: Python<'_>, x: &PyTensor, op: &str, a: f32, b: f32) -> PyResult<PyTensor> {
    let (x, op) = (x.inner.clone(), unary_op(op)?);
    run(py, move || math::unary(&x, op, a, b))
}

#[pyfunction]
fn binary(py: Python<'_>, a: &PyTensor, b: &PyTensor, op: &str) -> PyResult<PyTensor> {
    let (a, b, op) = (a.inner.clone(), b.inner.clone(), binary_op(op)?);
    run(py, move || math::binary(&a, &b, op))
}

#[pyfunction]
fn sum(py: Python<'_>, x: &PyTensor, dims: Vec<usize>, keepdim: bool) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::sum(&x, &dims, keepdim))
}

#[pyfunction]
fn mean(py: Python<'_>, x: &PyTensor, dims: Vec<usize>, keepdim: bool) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::mean(&x, &dims, keepdim))
}

#[pyfunction]
fn amax(py: Python<'_>, x: &PyTensor, dims: Vec<usize>, keepdim: bool, is_max: bool) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::amax(&x, &dims, keepdim, is_max))
}

#[pyfunction]
fn softmax(py: Python<'_>, x: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::softmax(&x, dim))
}

#[pyfunction]
fn log_softmax(py: Python<'_>, x: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::log_softmax(&x, dim))
}

#[pyfunction]
fn normalize(py: Python<'_>, x: &PyTensor, dim: usize, eps: f32) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::normalize(&x, dim, eps))
}

#[pyfunction]
#[pyo3(signature = (x, normalized, weight, bias, eps))]
fn layer_norm(py: Python<'_>, x: &PyTensor, normalized: usize, weight: Option<&PyTensor>, bias: Option<&PyTensor>, eps: f32) -> PyResult<PyTensor> {
    let (x, w, b) = (x.inner.clone(), weight.map(|t| t.inner.clone()), bias.map(|t| t.inner.clone()));
    run(py, move || math::layer_norm(&x, normalized, w.as_ref(), b.as_ref(), eps))
}

#[pyfunction]
fn silu(py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::silu(&x))
}

#[pyfunction]
fn permute(py: Python<'_>, x: &PyTensor, dims: Vec<usize>) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || shape::permute(&x, &dims))
}

#[pyfunction]
fn transpose(py: Python<'_>, x: &PyTensor, d0: usize, d1: usize) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || shape::transpose(&x, d0, d1))
}

#[pyfunction]
fn expand(py: Python<'_>, x: &PyTensor, shape: Vec<usize>) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || shape::expand(&x, &shape))
}

/// Basic indexing: read `shape` elements at `offset + c . strides`.
#[pyfunction]
fn strided_read(py: Python<'_>, x: &PyTensor, offset: usize, strides: Vec<usize>, shape: Vec<usize>, select: bool) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    let name = if select { "SelectBackward0" } else { "SliceBackward0" };
    run(py, move || shape::strided_read(&x, offset, &strides, &shape, name))
}

#[pyfunction]
fn write_region(py: Python<'_>, dst: &PyTensor, offset: usize, strides: Vec<usize>, shape: Vec<usize>, value: &PyTensor) -> PyResult<()> {
    let (dst, value) = (dst.inner.clone(), value.inner.clone());
    py.detach(move || shape::write_region(&dst, offset, &strides, &shape, &value)).map_err(err)
}

#[pyfunction]
fn cat(py: Python<'_>, xs: Vec<PyRef<'_, PyTensor>>, dim: usize) -> PyResult<PyTensor> {
    let xs: Vec<Tensor> = xs.iter().map(|t| t.inner.clone()).collect();
    run(py, move || shape::cat(&xs.iter().collect::<Vec<_>>(), dim))
}

#[pyfunction]
fn stack(py: Python<'_>, xs: Vec<PyRef<'_, PyTensor>>, dim: usize) -> PyResult<PyTensor> {
    let xs: Vec<Tensor> = xs.iter().map(|t| t.inner.clone()).collect();
    run(py, move || shape::stack(&xs.iter().collect::<Vec<_>>(), dim))
}

#[pyfunction]
fn matmul(py: Python<'_>, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let (a, b) = (a.inner.clone(), b.inner.clone());
    run(py, move || shape::matmul(&a, &b))
}

/// torch's `contiguous()`: the tensor itself, or a real copy of a view copy
/// (so writes to the result are allowed).
#[pyfunction]
fn contiguous(py: Python<'_>, x: &PyTensor) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || if x.is_view_copy() { x.clone_data() } else { Ok(x) })
}

#[pyfunction]
fn is_view_copy(x: &PyTensor) -> bool {
    x.inner.is_view_copy()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(unary, m)?)?;
    m.add_function(wrap_pyfunction!(binary, m)?)?;
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(amax, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(silu, m)?)?;
    m.add_function(wrap_pyfunction!(permute, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(expand, m)?)?;
    m.add_function(wrap_pyfunction!(strided_read, m)?)?;
    m.add_function(wrap_pyfunction!(write_region, m)?)?;
    m.add_function(wrap_pyfunction!(cat, m)?)?;
    m.add_function(wrap_pyfunction!(stack, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(contiguous, m)?)?;
    m.add_function(wrap_pyfunction!(is_view_copy, m)?)?;
    Ok(())
}
