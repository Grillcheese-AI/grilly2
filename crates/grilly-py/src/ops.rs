//! Bindings for grilly-torch's math and shape ops. Ops are named by string;
//! argument checking and torch's calling conventions live in Python.

use grilly_core::fusion::{Graph, Node};
use grilly_core::generic::{Binary, IntBinary, Reduce, Unary};
use grilly_core::{DType, Tensor};
use grilly_torch::{attention, math, shape};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

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
        "sin" => Unary::Sin,
        "cos" => Unary::Cos,
        "floor" => Unary::Floor,
        "ceil" => Unary::Ceil,
        "trunc" => Unary::Trunc,
        "round" => Unary::Round,
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
        // 1.0 / 0.0 masks with no gradient; for library code (dropout), since
        // user-facing comparisons must return bool tensors.
        "eq" => Binary::Eq,
        "ne" => Binary::Ne,
        "gt" => Binary::Gt,
        "ge" => Binary::Ge,
        "lt" => Binary::Lt,
        "le" => Binary::Le,
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

/// Fused rms_norm (no autograd): y over rows of the last `n` elements.
#[pyfunction]
#[pyo3(signature = (x, n, weight, eps))]
fn rms_norm(py: Python<'_>, x: &PyTensor, n: usize, weight: Option<&PyTensor>, eps: f32) -> PyResult<PyTensor> {
    let (x, w) = (x.inner.clone(), weight.map(|t| t.inner.clone()));
    run(py, move || Ok(grilly_core::generic::rms_norm(&x, n, None, w.as_ref(), eps)?.0))
}

/// (rms_norm(x + residual), x + residual) in one kernel (no autograd).
#[pyfunction]
#[pyo3(signature = (x, residual, n, weight, eps))]
fn add_rms_norm(
    py: Python<'_>,
    x: &PyTensor,
    residual: &PyTensor,
    n: usize,
    weight: Option<&PyTensor>,
    eps: f32,
) -> PyResult<(PyTensor, PyTensor)> {
    let (x, r, w) = (x.inner.clone(), residual.inner.clone(), weight.map(|t| t.inner.clone()));
    let (y, h) = py
        .detach(|| grilly_core::generic::rms_norm(&x, n, Some(&r), w.as_ref(), eps))
        .map_err(err)?;
    Ok((wrap(y), wrap(h.expect("a residual gives h"))))
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

/// Scatter-add `src` along `dim` at `index` into a zero tensor of `shape`.
/// python/grilly/_ops.py builds `index_add_` on it.
#[pyfunction]
fn index_add_into_zeros(
    py: Python<'_>,
    src: &PyTensor,
    dim: usize,
    index: &PyTensor,
    shape: Vec<usize>,
) -> PyResult<PyTensor> {
    let (src, index) = (src.inner.clone(), index.inner.clone());
    run(py, move || {
        grilly_core::generic::index_add_backward(
            &src,
            dim,
            &index,
            grilly_core::generic::IndexMode::Select,
            &shape,
        )
    })
}

/// `a @ b^T` over the last two dims, without ever building `b^T`.
/// python/grilly/__init__.py routes `x @ w.t()` here.
#[pyfunction]
fn matmul_t(py: Python<'_>, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let (a, b) = (a.inner.clone(), b.inner.clone());
    run(py, move || shape::matmul_t(&a, &b))
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

fn dtype_of(code: u32) -> PyResult<DType> {
    DType::from_code(code).map_err(err)
}

/// isfinite / isinf / isnan: a bool tensor.
#[pyfunction]
fn predicate(py: Python<'_>, x: &PyTensor, op: &str) -> PyResult<PyTensor> {
    let op = match op {
        "isfinite" => Unary::IsFinite,
        "isinf" => Unary::IsInf,
        "isnan" => Unary::IsNan,
        _ => return Err(PyValueError::new_err(format!("unknown predicate {op:?}"))),
    };
    let x = x.inner.clone();
    run(py, move || math::predicate(&x, op))
}

/// Comparison of two same-dtype tensors: a bool tensor.
#[pyfunction]
fn compare(py: Python<'_>, a: &PyTensor, b: &PyTensor, op: &str) -> PyResult<PyTensor> {
    let (a, b, op) = (a.inner.clone(), b.inner.clone(), binary_op(op)?);
    run(py, move || math::compare(&a, &b, op))
}

/// Exact integer/bool arithmetic or bitwise ops, returned as dtype `out`.
#[pyfunction]
fn integer(py: Python<'_>, a: &PyTensor, b: &PyTensor, op: &str, out: u32) -> PyResult<PyTensor> {
    let op = match op {
        "add" => IntBinary::Add,
        "sub" => IntBinary::Sub,
        "mul" => IntBinary::Mul,
        "maximum" => IntBinary::Maximum,
        "minimum" => IntBinary::Minimum,
        "and" => IntBinary::And,
        "or" => IntBinary::Or,
        "xor" => IntBinary::Xor,
        "floor_divide" => IntBinary::FloorDiv,
        "remainder" => IntBinary::Remainder,
        "trunc_divide" => IntBinary::TruncDiv,
        _ => return Err(PyValueError::new_err(format!("unknown integer op {op:?}"))),
    };
    let (a, b, out) = (a.inner.clone(), b.inner.clone(), dtype_of(out)?);
    run(py, move || math::integer(&a, &b, op, out))
}

/// int64 sum / max / min over dims (all when empty).
#[pyfunction]
fn integer_reduce(py: Python<'_>, x: &PyTensor, dims: Vec<usize>, op: &str, keepdim: bool) -> PyResult<PyTensor> {
    let op = match op {
        "sum" => Reduce::Sum,
        "max" => Reduce::Max,
        "min" => Reduce::Min,
        _ => return Err(PyValueError::new_err(format!("unknown integer reduction {op:?}"))),
    };
    let x = x.inner.clone();
    run(py, move || math::integer_reduce(&x, &dims, op, keepdim))
}

#[pyfunction]
#[pyo3(name = "where_")]
fn where_py(py: Python<'_>, cond: &PyTensor, a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let (c, a, b) = (cond.inner.clone(), a.inner.clone(), b.inner.clone());
    run(py, move || math::where_(&c, &a, &b))
}

#[pyfunction]
fn gather(py: Python<'_>, x: &PyTensor, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
    let (x, i) = (x.inner.clone(), index.inner.clone());
    run(py, move || shape::gather(&x, dim, &i))
}

#[pyfunction]
fn index_select(py: Python<'_>, x: &PyTensor, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
    let (x, i) = (x.inner.clone(), index.inner.clone());
    run(py, move || shape::index_select(&x, dim, &i))
}

/// Advanced indexing along `dim` with an index tensor of any shape.
#[pyfunction]
fn index_tensor(py: Python<'_>, x: &PyTensor, dim: usize, index: &PyTensor) -> PyResult<PyTensor> {
    let (x, i) = (x.inner.clone(), index.inner.clone());
    run(py, move || shape::index_tensor(&x, dim, &i))
}

#[pyfunction]
#[pyo3(signature = (index, weight, padding_idx=None))]
fn embedding(py: Python<'_>, index: &PyTensor, weight: &PyTensor, padding_idx: Option<usize>) -> PyResult<PyTensor> {
    let (i, w) = (index.inner.clone(), weight.inner.clone());
    run(py, move || shape::embedding(&i, &w, padding_idx))
}

fn run2(py: Python<'_>, f: impl FnOnce() -> grilly_core::Result<(Tensor, Tensor)> + Send) -> PyResult<(PyTensor, PyTensor)> {
    py.detach(f).map(|(a, b)| (wrap(a), wrap(b))).map_err(err)
}

#[pyfunction]
fn argmax(py: Python<'_>, x: &PyTensor, dim: usize, keepdim: bool, max: bool) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || shape::argmax(&x, dim, keepdim, max))
}

/// (values, indices) of `x.max(dim)` or `x.min(dim)`.
#[pyfunction]
fn max_dim(py: Python<'_>, x: &PyTensor, dim: usize, keepdim: bool, max: bool) -> PyResult<(PyTensor, PyTensor)> {
    let x = x.inner.clone();
    run2(py, move || shape::max_dim(&x, dim, keepdim, max))
}

/// (values, indices) of the first `take` elements in sorted order along
/// `dim`; `name` is the autograd node's name (SortBackward0, TopkBackward0).
#[pyfunction]
fn sorted(py: Python<'_>, x: &PyTensor, dim: usize, descending: bool, take: usize, name: &str) -> PyResult<(PyTensor, PyTensor)> {
    let x = x.inner.clone();
    let name: &'static str = if name == "topk" { "TopkBackward0" } else { "SortBackward0" };
    run2(py, move || shape::sorted(&x, dim, descending, take, name))
}

#[pyfunction]
fn scatter(py: Python<'_>, x: &PyTensor, dim: usize, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> {
    let (x, i, s) = (x.inner.clone(), index.inner.clone(), src.inner.clone());
    run(py, move || shape::scatter(&x, dim, &i, &s))
}

/// In place into `x`'s storage; returns the handle the caller should hold.
/// Start the persistent high-water mark again from what is alive now.
#[pyfunction]
fn reset_persistent_peak() -> PyResult<()> {
    grilly_core::Device::default_device().and_then(|d| d.reset_persistent_peak()).map_err(err)
}

#[pyfunction]
fn scatter_(py: Python<'_>, x: &PyTensor, dim: usize, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> {
    let (x, i, s) = (x.inner.clone(), index.inner.clone(), src.inner.clone());
    run(py, move || shape::scatter_(&x, dim, &i, &s))
}

#[pyfunction]
fn index_copy_(py: Python<'_>, x: &PyTensor, dim: usize, index: &PyTensor, src: &PyTensor) -> PyResult<PyTensor> {
    let (x, i, s) = (x.inner.clone(), index.inner.clone(), src.inner.clone());
    run(py, move || shape::index_copy_(&x, dim, &i, &s))
}

/// Fill `x` in place with normals from the counter-based stream
/// `(key, stream)`. Nothing is uploaded, so it works inside a capture.
#[pyfunction]
fn philox_normal_(py: Python<'_>, x: &PyTensor, key: u64, stream: u64, std: f32) -> PyResult<()> {
    let x = x.inner.clone();
    py.detach(move || grilly_core::generic::philox_normal(&x, key, stream, std)).map_err(err)
}

#[pyfunction]
fn cumsum(py: Python<'_>, x: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::cumsum(&x, dim))
}

#[pyfunction]
fn logcumsumexp(py: Python<'_>, x: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    let x = x.inner.clone();
    run(py, move || math::logcumsumexp(&x, dim))
}

/// flex_attention's fused block-sparse path: (out, lse), with autograd
/// through `out`. `tiles` is (bits, kv_count, kv_list, q_count, q_list).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn flex_attention(
    py: Python<'_>,
    q: &PyTensor,
    k: &PyTensor,
    v: &PyTensor,
    tiles: Vec<PyRef<'_, PyTensor>>,
    mb: usize,
    mh: usize,
    scale: f32,
) -> PyResult<(PyTensor, PyTensor)> {
    let [bits, kv_count, kv_list, q_count, q_list] = <[_; 5]>::try_from(tiles)
        .map_err(|_| PyValueError::new_err("tiles is (bits, kv_count, kv_list, q_count, q_list)"))?
        .map(|t| t.inner.clone());
    let (q, k, v) = (q.inner.clone(), k.inner.clone(), v.inner.clone());
    let tiles = attention::Tiles { bits, kv_count, kv_list, q_count, q_list, mb, mh };
    run2(py, move || attention::flex_attention(&q, &k, &v, &tiles, scale))
}

/// Tile lists (bits, kv_count, kv_list, q_count, q_list) of a bool mask
/// (mb, mh, L, S), built on the device.
#[pyfunction]
fn flash_tiles(py: Python<'_>, mask: &PyTensor) -> PyResult<Vec<PyTensor>> {
    let mask = mask.inner.clone();
    py.detach(move || attention::Tiles::from_mask(&mask))
        .map(|t| [t.bits, t.kv_count, t.kv_list, t.q_count, t.q_list].into_iter().map(wrap).collect())
        .map_err(err)
}

/// Submission counters of the default device (CLAUDE.md §11), as a dict.
#[pyfunction]
fn submit_stats(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    let s = grilly_core::Device::default_device().and_then(|d| d.submit_stats()).map_err(err)?;
    let d = pyo3::types::PyDict::new(py);
    for (k, v) in [
        ("submits", s.submits),
        ("dispatches", s.dispatches),
        ("submit_ns_total", s.submit_ns_total),
        ("wait_ns_total", s.wait_ns_total),
        ("wait_ns_max", s.wait_ns_max),
        ("wait_ns_last", s.wait_ns_last),
        ("desc_pools", s.desc_pools),
        ("buffer_allocations", s.buffer_allocations),
        ("buffers_outstanding", s.buffers_outstanding),
        ("batch_reuses", s.batch_reuses),
        ("persistent_buffers", s.persistent_buffers),
        ("persistent_bytes", s.persistent_bytes),
        ("persistent_peak", s.persistent_peak),
    ] {
        d.set_item(k, v)?;
    }
    Ok(d)
}

/// A captured sequence of GPU commands, replayable without running the
/// Python that produced it (grilly_core::graph).
#[pyclass(name = "CommandGraph", module = "grilly._C", frozen)]
struct PyCommandGraph {
    inner: grilly_core::graph::CommandGraph,
}

#[pymethods]
impl PyCommandGraph {
    /// Commands recorded.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Re-issue them, in order.
    fn replay(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.replay()).map_err(err)
    }

    fn __repr__(&self) -> String {
        format!("CommandGraph({} commands)", self.inner.len())
    }
}

/// Start recording the device's commands. Every command is still issued;
/// `end_capture` returns what was recorded.
#[pyfunction]
fn begin_capture(py: Python<'_>) -> PyResult<()> {
    py.detach(|| grilly_core::Device::default_device().and_then(|d| d.begin_capture())).map_err(err)
}

/// Stop recording and return the graph.
#[pyfunction]
fn end_capture(py: Python<'_>) -> PyResult<PyCommandGraph> {
    py.detach(|| {
        let device = grilly_core::Device::default_device()?;
        let recorder = device.end_capture()?;
        Ok(grilly_core::graph::CommandGraph::new(device, recorder))
    })
    .map(|inner| PyCommandGraph { inner })
    .map_err(err)
}

/// Run everything recorded on the default device and wait (no kernel of its
/// own), as torch.cuda.synchronize.
#[pyfunction]
fn synchronize(py: Python<'_>) -> PyResult<()> {
    py.detach(|| grilly_core::Device::default_device().and_then(|d| d.synchronize())).map_err(err)
}

/// One node of a `fused()` graph, as a Python tuple: `("input", i)`,
/// `("param", i)`, `("unary", op, x, a_or_None, b_or_None)` or
/// `("binary", op, x, y)`, where `x`, `y`, `a`, `b` are earlier node
/// indices. See `crates/grilly-core/src/fusion.rs`.
fn parse_node(item: &Bound<'_, PyAny>) -> PyResult<Node> {
    let tuple = item.cast::<PyTuple>().map_err(|_| {
        PyValueError::new_err("fused(): each node must be a tuple: (\"input\", i), (\"param\", i), \
            (\"unary\", op, x, a, b) or (\"binary\", op, x, y)")
    })?;
    let kind: String = tuple.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "input" => Node::Input(tuple.get_item(1)?.extract()?),
        "param" => Node::Param(tuple.get_item(1)?.extract()?),
        "unary" => {
            let op = unary_op(&tuple.get_item(1)?.extract::<String>()?)?;
            let x: usize = tuple.get_item(2)?.extract()?;
            let a: Option<usize> = tuple.get_item(3)?.extract()?;
            let b: Option<usize> = tuple.get_item(4)?.extract()?;
            Node::Unary(op, x, a, b)
        }
        "binary" => {
            let op = binary_op(&tuple.get_item(1)?.extract::<String>()?)?;
            let x: usize = tuple.get_item(2)?.extract()?;
            let y: usize = tuple.get_item(3)?.extract()?;
            Node::Binary(op, x, y)
        }
        _ => return Err(PyValueError::new_err(format!("fused(): unknown node kind {kind:?}"))),
    })
}

/// `torch.compile`'s fusion path (docs/fusion.md): builds one
/// `grilly_core::fusion::Graph` from `nodes` and runs it as a single
/// kernel, producing one tensor per index in `outputs`. `inputs` and
/// `params` are the graph's real-tensor and scalar leaves, in the order
/// their `("input", i)` / `("param", i)` nodes reference. Not recorded for
/// autograd: grilly-py's lazy capture (python/grilly/_lazy.py) only builds
/// these graphs outside grad mode, or for tensors that don't require it.
#[pyfunction]
fn fused(
    py: Python<'_>,
    nodes: Vec<Bound<'_, PyAny>>,
    outputs: Vec<usize>,
    inputs: Vec<PyRef<'_, PyTensor>>,
    params: Vec<f32>,
    shape: Vec<usize>,
) -> PyResult<Vec<PyTensor>> {
    let nodes = nodes.iter().map(parse_node).collect::<PyResult<Vec<Node>>>()?;
    let inputs: Vec<Tensor> = inputs.iter().map(|t| t.inner.clone()).collect();
    let graph = Graph { nodes, outputs, inputs: inputs.len(), params: params.len() };
    py.detach(move || graph.run(&inputs.iter().collect::<Vec<_>>(), &params, &shape))
        .map(|ts| ts.into_iter().map(wrap).collect())
        .map_err(err)
}

#[pyfunction]
fn set_deterministic(on: bool) {
    grilly_core::generic::set_deterministic(on);
}

#[pyfunction]
fn deterministic() -> bool {
    grilly_core::generic::deterministic()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gather, m)?)?;
    m.add_function(wrap_pyfunction!(rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(add_rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(index_select, m)?)?;
    m.add_function(wrap_pyfunction!(index_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(embedding, m)?)?;
    m.add_function(wrap_pyfunction!(flex_attention, m)?)?;
    m.add_function(wrap_pyfunction!(flash_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(submit_stats, m)?)?;
    m.add_class::<PyCommandGraph>()?;
    m.add_function(wrap_pyfunction!(begin_capture, m)?)?;
    m.add_function(wrap_pyfunction!(end_capture, m)?)?;
    m.add_function(wrap_pyfunction!(reset_persistent_peak, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(scatter, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_, m)?)?;
    m.add_function(wrap_pyfunction!(index_copy_, m)?)?;
    m.add_function(wrap_pyfunction!(philox_normal_, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(logcumsumexp, m)?)?;
    m.add_function(wrap_pyfunction!(argmax, m)?)?;
    m.add_function(wrap_pyfunction!(max_dim, m)?)?;
    m.add_function(wrap_pyfunction!(sorted, m)?)?;
    m.add_function(wrap_pyfunction!(set_deterministic, m)?)?;
    m.add_function(wrap_pyfunction!(deterministic, m)?)?;
    m.add_function(wrap_pyfunction!(predicate, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_function(wrap_pyfunction!(integer, m)?)?;
    m.add_function(wrap_pyfunction!(integer_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(where_py, m)?)?;
    m.add_function(wrap_pyfunction!(unary, m)?)?;
    m.add_function(wrap_pyfunction!(binary, m)?)?;
    m.add_function(wrap_pyfunction!(fused, m)?)?;
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
    m.add_function(wrap_pyfunction!(matmul_t, m)?)?;
    m.add_function(wrap_pyfunction!(index_add_into_zeros, m)?)?;
    m.add_function(wrap_pyfunction!(contiguous, m)?)?;
    m.add_function(wrap_pyfunction!(is_view_copy, m)?)?;
    Ok(())
}
