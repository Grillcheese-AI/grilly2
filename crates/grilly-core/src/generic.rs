//! grilly2's generic kernels (shaders/g2-*.glsl): elementwise math,
//! broadcasting binary ops, reductions and strided copies. Untracked by
//! autograd: differentiable ops are built on these in the op crates.
//!
//! Tensors are passed to these kernels as whole storage buffers plus element
//! offsets, so any tensor (including one at an unaligned offset) can be an
//! operand.

use crate::error::{Error, Result};
use crate::kernel::{Arg, Kernel};
use crate::tensor::Tensor;

const MAX_DIMS: usize = 6;

pub const UNARY: Kernel = Kernel { shader: "g2-unary", bindings: 2, push_words: 6, local_size: [256, 1, 1] };
pub const BINARY: Kernel = Kernel { shader: "g2-binary", bindings: 3, push_words: 6 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const REDUCE: Kernel = Kernel { shader: "g2-reduce", bindings: 2, push_words: 6, local_size: [256, 1, 1] };
pub const COPY: Kernel = Kernel { shader: "g2-copy", bindings: 2, push_words: 4 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };

pub const ALL: &[Kernel] = &[UNARY, BINARY, REDUCE, COPY];

/// Elementwise `f(x; a, b)`. Codes match g2-unary.glsl.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unary {
    Neg = 0,
    Abs,
    Sign,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Sigmoid,
    /// `x + a`
    AddScalar,
    /// `x * a`
    MulScalar,
    /// `a - x`
    RsubScalar,
    /// `a / x`
    RdivScalar,
    /// `x ** a` (negative x only for integer a, as in torch)
    PowScalar,
    /// `min(max(x, a), b)`
    Clamp,
    /// `1` where `a <= x <= b`, else `0`
    ClampMask,
    /// exact (erf) GELU
    Gelu,
    /// derivative of exact GELU
    GeluGrad,
}

/// Elementwise `op(a, b)` with broadcasting. Codes match g2-binary.glsl.
/// Comparisons produce 1.0 / 0.0.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binary {
    Add = 0,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

/// Reductions along one axis. Codes match g2-reduce.glsl.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduce {
    Sum = 0,
    Mean,
    Max,
    Min,
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

/// Row-major strides of a contiguous tensor of `shape`.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// torch's broadcasting rule: align trailing dims; each pair must be equal
/// or contain a 1.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(Error::Shape(format!(
                    "The size of tensor a ({da}) must match the size of tensor b ({db}) at \
                     non-singleton dimension {i}"
                )))
            }
        };
    }
    Ok(out)
}

/// Strides for reading a contiguous tensor of `shape` as if it had
/// `out_shape` (0 on broadcast and missing leading dims).
fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let own = contiguous_strides(shape);
    let lead = out_shape.len() - shape.len();
    (0..out_shape.len())
        .map(|i| if i < lead || shape[i - lead] == 1 { 0 } else { own[i - lead] })
        .collect()
}

fn padded(v: &[usize], what: &str) -> Result<[u32; MAX_DIMS]> {
    if v.len() > MAX_DIMS {
        return Err(Error::NotImplemented(format!("{what} with more than {MAX_DIMS} dimensions")));
    }
    let mut out = [1u32; MAX_DIMS];
    for (o, &x) in out.iter_mut().zip(v) {
        *o = u32_of(x, what)?;
    }
    Ok(out)
}

/// `f(x; a, b)` into a new tensor.
pub fn unary(x: &Tensor, op: Unary, a: f32, b: f32) -> Result<Tensor> {
    let out = Tensor::empty(x.device(), x.shape())?;
    let push = [u32_of(x.numel(), "numel")?, op as u32, a.to_bits(), b.to_bits(), u32_of(x.offset(), "offset")?, 0];
    UNARY.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], x.numel(), &push)?;
    Ok(out)
}

/// `x = f(x; a, b)` in place; bumps `x`'s version.
pub fn unary_inplace(x: &Tensor, op: Unary, a: f32, b: f32) -> Result<()> {
    let off = u32_of(x.offset(), "offset")?;
    let push = [u32_of(x.numel(), "numel")?, op as u32, a.to_bits(), b.to_bits(), off, off];
    UNARY.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(x)], x.numel(), &push)?;
    x.bump_version();
    Ok(())
}

/// `op(a, b)` with broadcasting, into a new tensor.
pub fn binary(a: &Tensor, b: &Tensor, op: Binary) -> Result<Tensor> {
    let shape = broadcast_shape(a.shape(), b.shape())?;
    let out = Tensor::empty(a.device(), &shape)?;
    let mut push = vec![
        u32_of(out.numel(), "numel")?,
        op as u32,
        shape.len() as u32,
        u32_of(a.offset(), "offset")?,
        u32_of(b.offset(), "offset")?,
        0,
    ];
    push.extend(padded(&shape, "binary op")?);
    push.extend(padded(&broadcast_strides(a.shape(), &shape), "binary op")?);
    push.extend(padded(&broadcast_strides(b.shape(), &shape), "binary op")?);
    BINARY.launch_flat(a.device(), &[Arg::Storage(a), Arg::Storage(b), Arg::Storage(&out)], out.numel(), &push)?;
    Ok(out)
}

/// Reduce `x` along `dim`.
pub fn reduce(x: &Tensor, dim: usize, op: Reduce, keepdim: bool) -> Result<Tensor> {
    let shape = x.shape();
    if dim >= shape.len() {
        return Err(Error::Shape(format!("reduce: dim {dim} out of range for shape {shape:?}")));
    }
    let outer: usize = shape[..dim].iter().product();
    let len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    if len == 0 && matches!(op, Reduce::Max | Reduce::Min) {
        return Err(Error::Invalid(format!(
            "max(): Expected reduction dim {dim} to have non-zero size."
        )));
    }
    let mut out_shape = shape.to_vec();
    if keepdim {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }
    let out = Tensor::empty(x.device(), &out_shape)?;
    let push = [
        u32_of(outer, "outer")?,
        u32_of(len, "len")?,
        u32_of(inner, "inner")?,
        op as u32,
        u32_of(x.offset(), "offset")?,
        0,
    ];
    REDUCE.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], outer * inner, &push)?;
    Ok(out)
}

/// Reduce every element of `x` to a 0-d tensor.
pub fn reduce_all(x: &Tensor, op: Reduce) -> Result<Tensor> {
    let flat = x.detach().reshape(&[x.numel()])?;
    reduce(&flat, 0, op, false)
}

/// Sum a gradient of broadcast shape back down to `shape` (the backward of
/// broadcasting).
pub fn sum_to_shape(g: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if g.shape() == shape {
        return Ok(g.clone());
    }
    let mut t = g.clone();
    while t.dim() > shape.len() {
        t = reduce(&t, 0, Reduce::Sum, false)?;
    }
    for (d, &s) in shape.iter().enumerate() {
        if s == 1 && t.shape()[d] != 1 {
            t = reduce(&t, d, Reduce::Sum, true)?;
        }
    }
    if t.shape() != shape {
        return Err(Error::Shape(format!("cannot sum shape {:?} down to {shape:?}", g.shape())));
    }
    Ok(t)
}

/// For every coordinate `c` of `shape`:
/// `dst[dst_off + c . dst_strides] = src[src_off + c . src_strides]`
/// (offsets and strides in elements, relative to each storage). Bumps `dst`'s
/// version.
pub fn copy_strided(
    src: &Tensor,
    src_off: usize,
    src_strides: &[usize],
    dst: &Tensor,
    dst_off: usize,
    dst_strides: &[usize],
    shape: &[usize],
) -> Result<()> {
    if src_strides.len() != shape.len() || dst_strides.len() != shape.len() {
        return Err(Error::Invalid("copy_strided: strides must match the shape's rank".into()));
    }
    let n: usize = shape.iter().product();
    let mut push = vec![u32_of(n, "numel")?, shape.len() as u32, u32_of(src_off, "offset")?, u32_of(dst_off, "offset")?];
    push.extend(padded(shape, "strided copy")?);
    push.extend(padded(src_strides, "strided copy")?);
    push.extend(padded(dst_strides, "strided copy")?);
    COPY.launch_flat(src.device(), &[Arg::Storage(src), Arg::Storage(dst)], n, &push)?;
    dst.bump_version();
    Ok(())
}
