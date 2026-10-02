//! grilly2's generic kernels (shaders/g2-*.glsl): elementwise math,
//! broadcasting binary ops, reductions and strided copies. Untracked by
//! autograd: differentiable ops are built on these in the op crates.
//!
//! Tensors are passed to these kernels as whole storage buffers plus element
//! offsets, so any tensor (including one at an unaligned offset) can be an
//! operand.

use crate::dtype::{require_float, DType};
use crate::error::{Error, Result};
use crate::kernel::{Arg, Kernel};
use crate::tensor::Tensor;

const MAX_DIMS: usize = 6;

pub const UNARY: Kernel = Kernel { shader: "g2-unary", bindings: 2, push_words: 6, local_size: [256, 1, 1] };
pub const BINARY: Kernel = Kernel { shader: "g2-binary", bindings: 3, push_words: 6 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const REDUCE: Kernel = Kernel { shader: "g2-reduce", bindings: 2, push_words: 11, local_size: [256, 1, 1] };
pub const REDUCE_ROWS: Kernel = Kernel { shader: "g2-reduce-rows", bindings: 2, push_words: 10, local_size: [256, 1, 1] };
pub const COPY: Kernel = Kernel { shader: "g2-copy", bindings: 2, push_words: 4 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };

pub const CAST: Kernel = Kernel { shader: "g2-cast", bindings: 2, push_words: 5, local_size: [256, 1, 1] };
/// Push: n, dtype, start (2 words), step (2 words), dst offset.
pub const ARANGE: Kernel = Kernel { shader: "g2-arange", bindings: 1, push_words: 7, local_size: [256, 1, 1] };
/// Push: n, out offset, key (2 words), stream (2 words), std bits.
pub const PHILOX_NORMAL: Kernel =
    Kernel { shader: "g2-philox-normal", bindings: 1, push_words: 7, local_size: [256, 1, 1] };
pub const BINARY_INT: Kernel = Kernel { shader: "g2-binary-int", bindings: 3, push_words: 7 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const WHERE: Kernel = Kernel { shader: "g2-where", bindings: 4, push_words: 7 + 4 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const REDUCE_INT: Kernel = Kernel { shader: "g2-reduce-int", bindings: 2, push_words: 6, local_size: [256, 1, 1] };
/// Contiguous int64 rows, a workgroup each. Push: rows, len, chunk, chunks, op, in_off, out_off.
pub const REDUCE_INT_ROWS: Kernel =
    Kernel { shader: "g2-reduce-int-rows", bindings: 2, push_words: 7, local_size: [256, 1, 1] };

pub const INDEX: Kernel = Kernel { shader: "g2-index", bindings: 3, push_words: 9 + 2 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const INDEX_ADD: Kernel = Kernel { shader: "g2-index-add", bindings: 3, push_words: 9 + 2 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const INDEX_ADD_ATOMIC: Kernel = Kernel { shader: "g2-index-add-atomic", bindings: 3, push_words: 9 + 2 * MAX_DIMS as u32, local_size: [256, 1, 1] };

pub const ARG_REDUCE: Kernel = Kernel { shader: "g2-argreduce", bindings: 2, push_words: 7, local_size: [256, 1, 1] };
/// Contiguous rows, a workgroup each. Push: rows, len, is_max, is_int, in_off, out_off.
pub const ARG_REDUCE_ROWS: Kernel = Kernel { shader: "g2-argreduce-rows", bindings: 2, push_words: 6, local_size: [256, 1, 1] };
pub const SORT_INIT: Kernel = Kernel { shader: "g2-sort-init", bindings: 3, push_words: 5, local_size: [256, 1, 1] };
pub const SORT_STEP: Kernel = Kernel { shader: "g2-sort-step", bindings: 2, push_words: 7, local_size: [256, 1, 1] };
pub const SORT_TAKE: Kernel = Kernel { shader: "g2-sort-take", bindings: 2, push_words: 3, local_size: [256, 1, 1] };
/// Small-k top-k, a workgroup per (row, chunk). Push: rows, n, k, chunk,
/// chunks, in_off, descending, merging.
pub const TOPK: Kernel = Kernel { shader: "g2-topk", bindings: 4, push_words: 8, local_size: [64, 1, 1] };

pub const SCATTER: Kernel = Kernel { shader: "g2-scatter", bindings: 3, push_words: 8 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const SCAN: Kernel = Kernel { shader: "g2-scan", bindings: 2, push_words: 8, local_size: [256, 1, 1] };

pub const BMM: Kernel = Kernel { shader: "g2-bmm", bindings: 3, push_words: 9 + 3 * MAX_BATCH_DIMS as u32, local_size: [16, 16, 1] };

pub const LAYER_NORM: Kernel = Kernel { shader: "g2-layer-norm", bindings: 5, push_words: 7, local_size: [256, 1, 1] };
/// Bindings: x, residual, weight, y, h. Push: rows, n, eps, flags, x_off, r_off, w_off.
pub const RMS_NORM: Kernel = Kernel { shader: "g2-rms-norm", bindings: 5, push_words: 7, local_size: [256, 1, 1] };
pub const LAYER_NORM_BACKWARD: Kernel =
    Kernel { shader: "g2-layer-norm-backward", bindings: 6, push_words: 7, local_size: [256, 1, 1] };

pub const ALL: &[Kernel] = &[
    UNARY, BINARY, REDUCE, COPY, CAST, ARANGE, PHILOX_NORMAL, BINARY_INT, WHERE, REDUCE_INT, INDEX, INDEX_ADD, INDEX_ADD_ATOMIC,
    REDUCE_INT_ROWS, ARG_REDUCE, ARG_REDUCE_ROWS, RMS_NORM, SORT_INIT, SORT_STEP, SORT_TAKE, TOPK, SCATTER, SCAN, BMM, REDUCE_ROWS, LAYER_NORM,
    LAYER_NORM_BACKWARD,
];

/// Batch dimensions g2-bmm can broadcast over.
pub const MAX_BATCH_DIMS: usize = 4;

static DETERMINISTIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// torch's `use_deterministic_algorithms`: when on, ops that would sum with
/// atomics (in hardware order) use a fixed-order kernel instead.
pub fn set_deterministic(on: bool) {
    DETERMINISTIC.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn deterministic() -> bool {
    DETERMINISTIC.load(std::sync::atomic::Ordering::Relaxed)
}

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
    /// `1` where finite, else `0`
    IsFinite,
    /// `1` where +-inf, else `0`
    IsInf,
    /// `1` where NaN, else `0`
    IsNan,
    /// sine (accurate for |x| <= 8192)
    Sin,
    /// cosine (accurate for |x| <= 8192)
    Cos,
    Floor,
    Ceil,
    Trunc,
    /// to nearest, half to even (torch.round)
    Round,
}

/// int64 ops with broadcasting (g2-binary-int.glsl); comparisons and the
/// logical ops on bools produce bool.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntBinary {
    Add = 0,
    Sub,
    Mul,
    Maximum,
    Minimum,
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
    And,
    Or,
    Xor,
    /// `floor(a / b)`; b must be nonzero
    FloorDiv,
    /// `a - floor(a / b) * b` (the divisor's sign, as in torch); b nonzero
    Remainder,
    /// `a / b` rounded toward zero; b nonzero
    TruncDiv,
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

/// Inclusive scans along one axis. Codes match g2-scan.glsl.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scan {
    /// float32 (compensated) or int64 (exact)
    Cumsum = 0,
    /// float32 only
    LogCumsumExp,
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
pub(crate) fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
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
    require_float("elementwise math", x.dtype())?;
    let out = Tensor::empty(x.device(), x.shape())?;
    let push = [u32_of(x.numel(), "numel")?, op as u32, a.to_bits(), b.to_bits(), u32_of(x.offset(), "offset")?, 0];
    UNARY.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], x.numel(), &push)?;
    Ok(out)
}

/// `x = f(x; a, b)` in place; bumps `x`'s version.
pub fn unary_inplace(x: &Tensor, op: Unary, a: f32, b: f32) -> Result<()> {
    require_float("elementwise math", x.dtype())?;
    let off = u32_of(x.offset(), "offset")?;
    let push = [u32_of(x.numel(), "numel")?, op as u32, a.to_bits(), b.to_bits(), off, off];
    UNARY.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(x)], x.numel(), &push)?;
    x.bump_version();
    Ok(())
}

/// `op(a, b)` with broadcasting, into a new tensor.
pub fn binary(a: &Tensor, b: &Tensor, op: Binary) -> Result<Tensor> {
    require_float("elementwise math", a.dtype())?;
    require_float("elementwise math", b.dtype())?;
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
    require_float("reductions", x.dtype())?;
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
    if outer * inner == 0 {
        return Ok(out);
    }
    if len > 0 && inner == 1 {
        // the reduced dim is contiguous: a workgroup per row
        reduce_rows(x, x.offset(), outer, len, op, &out)?;
    } else {
        reduce_columns(x, outer, len, inner, op, &out)?;
    }
    Ok(out)
}

/// Threads a strided reduction aims for before it stops splitting the
/// reduced axis into chunks.
const COLUMN_THREADS: usize = 65_536;

/// Reduce the middle axis of `x` viewed as (outer, len, inner), a thread
/// per (output, chunk). With few outputs the axis is split into chunks
/// whose (hi, lo) partials are reduced the same way, pass after pass (each
/// shortens the axis at least 64-fold), until one chunk is left.
fn reduce_columns(x: &Tensor, outer: usize, len: usize, inner: usize, op: Reduce, out: &Tensor) -> Result<()> {
    let outputs = outer * inner;
    let (o32, i32_, count) = (u32_of(outer, "outer")?, u32_of(inner, "inner")?, u32_of(len, "len")?);
    let out_off = u32_of(out.offset(), "offset")?;
    let (mut src, mut len, mut pairs_in, mut off) = (x.clone(), len, 0, u32_of(x.offset(), "offset")?);
    loop {
        let wanted = if outputs >= COLUMN_THREADS { 1 } else { COLUMN_THREADS.div_ceil(outputs).min(len.div_ceil(64)).max(1) };
        let chunk = len.div_ceil(wanted).max(1);
        let chunks = len.div_ceil(chunk).max(1);
        let (l32, c32, k32) = (u32_of(len, "len")?, u32_of(chunks, "chunks")?, u32_of(chunk, "chunk")?);
        if chunks == 1 {
            let push = [o32, l32, i32_, op as u32, off, out_off, k32, 1, pairs_in, 0, count];
            return REDUCE.launch_flat(x.device(), &[Arg::Storage(&src), Arg::Storage(out)], outputs, &push);
        }
        let partial = Tensor::empty(x.device(), &[outputs * chunks * 2])?;
        let push = [o32, l32, i32_, op as u32, off, 0, k32, c32, pairs_in, 1, count];
        REDUCE.launch_flat(x.device(), &[Arg::Storage(&src), Arg::Storage(&partial)], outputs * chunks, &push)?;
        (src, len, pairs_in, off) = (partial, chunks, 1, 0);
    }
}

/// Workgroups a row reduction aims for before it stops splitting rows.
const TARGET_GROUPS: usize = 512;

/// Reduce `rows` contiguous rows of `len` floats starting at element
/// `offset` of `x`'s storage into `out` (one value per row). Rows too long
/// for the row count are split into chunks and finished by a second pass.
fn reduce_rows(x: &Tensor, offset: usize, rows: usize, len: usize, op: Reduce, out: &Tensor) -> Result<()> {
    let wanted = if rows >= TARGET_GROUPS { 1 } else { (TARGET_GROUPS / rows).min(len.div_ceil(2048)).max(1) };
    let chunk = len.div_ceil(wanted);
    let chunks = len.div_ceil(chunk);
    let groups = rows * chunks;
    let gx = groups.min(65_535);
    let threads = [u32_of(gx * 256, "workgroups")?, u32_of(groups.div_ceil(gx), "workgroups")?, 1];
    let launch = |src: &Tensor, push: [u32; 10], dst: &Tensor| {
        REDUCE_ROWS.launch(x.device(), &[Arg::Storage(src), Arg::Storage(dst)], threads, &push)
    };
    let (rows32, len32, count) = (u32_of(rows, "rows")?, u32_of(len, "len")?, u32_of(len, "len")?);
    let off = u32_of(offset, "offset")?;
    let out_off = u32_of(out.offset(), "offset")?;
    if chunks == 1 {
        return launch(x, [rows32, len32, len32, 1, op as u32, off, out_off, 0, 0, count], out);
    }
    let partial = Tensor::empty(x.device(), &[groups * 2])?;
    let (chunk32, chunks32) = (u32_of(chunk, "chunk")?, u32_of(chunks, "chunks")?);
    launch(x, [rows32, len32, chunk32, chunks32, op as u32, off, 0, 0, 1, count], &partial)?;
    let threads2 = [u32_of(rows.min(65_535) * 256, "workgroups")?, u32_of(rows.div_ceil(65_535), "workgroups")?, 1];
    REDUCE_ROWS.launch(
        x.device(),
        &[Arg::Storage(&partial), Arg::Storage(out)],
        threads2,
        &[rows32, chunks32, chunks32, 1, op as u32, 0, out_off, 1, 0, count],
    )
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
/// (offsets and strides in elements, relative to each storage). For filling
/// a freshly made `dst`; use [`write_strided`] to modify an existing tensor.
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
    if src.dtype() != dst.dtype() {
        return Err(Error::Invalid(format!(
            "copy_strided: {} source into {} destination",
            src.dtype().name(),
            dst.dtype().name()
        )));
    }
    // The kernel moves 32-bit words; a multi-word element becomes an extra
    // innermost dimension of its words.
    let w = src.dtype().words();
    let (mut shape, mut ss, mut ds) = (shape.to_vec(), src_strides.to_vec(), dst_strides.to_vec());
    for s in ss.iter_mut().chain(ds.iter_mut()) {
        *s *= w;
    }
    if w > 1 {
        shape.push(w);
        ss.push(1);
        ds.push(1);
    }
    let n: usize = shape.iter().product();
    let mut push = vec![u32_of(n, "numel")?, shape.len() as u32, u32_of(src_off * w, "offset")?, u32_of(dst_off * w, "offset")?];
    push.extend(padded(&shape, "strided copy")?);
    push.extend(padded(&ss, "strided copy")?);
    push.extend(padded(&ds, "strided copy")?);
    COPY.launch_flat(src.device(), &[Arg::Storage(src), Arg::Storage(dst)], n, &push)
}

/// [`copy_strided`] as an in-place write into an existing tensor: bumps
/// `dst`'s version, which saved tensors check.
pub fn write_strided(
    src: &Tensor,
    src_off: usize,
    src_strides: &[usize],
    dst: &Tensor,
    dst_off: usize,
    dst_strides: &[usize],
    shape: &[usize],
) -> Result<()> {
    copy_strided(src, src_off, src_strides, dst, dst_off, dst_strides, shape)?;
    dst.bump_version();
    Ok(())
}

/// `x` converted to `dtype` into a new tensor, with torch's rules (float to
/// int truncates toward zero; anything to bool is `!= 0`).
pub fn cast(x: &Tensor, dtype: DType) -> Result<Tensor> {
    let out = Tensor::empty_typed(x.device(), x.shape(), dtype)?;
    let push = [
        u32_of(x.numel(), "numel")?,
        x.dtype().code(),
        dtype.code(),
        u32_of(x.offset() * x.dtype().words(), "offset")?,
        0,
    ];
    CAST.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], x.numel(), &push)?;
    Ok(out)
}

/// `start + i * step` for `n` elements, computed on the device.
///
/// A counting sequence is a function of its own index, so sending one is
/// pure loss — and it is a *host write*, which a capture refuses. Every
/// `arange(n) < length` mask in a decode step went through that, so the
/// step could not be recorded. Integer starts and steps travel as two
/// words so the whole int64 range survives the push constant.
pub fn arange(device: &crate::Device, n: usize, start: i64, step: i64, dtype: DType) -> Result<Tensor> {
    let out = Tensor::empty_typed(device, &[n], dtype)?;
    if n == 0 {
        return Ok(out);
    }
    let (start_lo, start_hi, step_lo, step_hi) = match dtype {
        DType::F32 => ((start as f32).to_bits(), 0, (step as f32).to_bits(), 0),
        DType::I64 | DType::I32 => (
            start as u64 as u32,
            (start as u64 >> 32) as u32,
            step as u64 as u32,
            (step as u64 >> 32) as u32,
        ),
        _ => return Err(Error::Invalid(format!("arange: no {} sequence", dtype.name()))),
    };
    let push = [u32_of(n, "numel")?, dtype.code(), start_lo, start_hi, step_lo, step_hi, 0];
    ARANGE.launch_flat(device, &[Arg::Storage(&out)], n, &push)?;
    Ok(out)
}

/// Fill `x` with `std`-scaled standard normals from the counter-based
/// stream `(key, stream)`, on the device.
///
/// Element `i` is a function of `(key, stream, i)` and nothing else, so
/// the same three numbers give the same value on any machine and at any
/// later time. That is what lets evolution strategies regenerate a
/// population's perturbations for the update instead of storing them.
///
/// Not the same values as `Generator::normal`, and deliberately: that one
/// rounds Box-Muller from f64, this does it in f32 because f64 on RDNA2
/// runs at 1/16 rate. `randn` is unaffected.
pub fn philox_normal(x: &Tensor, key: u64, stream: u64, std: f32) -> Result<()> {
    require_float("philox_normal", x.dtype())?;
    let n = x.numel();
    if n == 0 {
        return Ok(());
    }
    let push = [
        u32_of(n, "numel")?,
        u32_of(x.offset(), "offset")?,
        key as u32,
        (key >> 32) as u32,
        stream as u32,
        (stream >> 32) as u32,
        std.to_bits(),
    ];
    PHILOX_NORMAL.launch_flat(x.device(), &[Arg::Storage(x)], n, &push)?;
    x.bump_version();
    Ok(())
}

/// Write the integer `value` to every element of `x`, on the device.
///
/// `vkCmdFillBuffer` takes a 4-byte pattern, so an int64 fill had to
/// upload one element and broadcast it with a strided copy — a **host
/// write**, which a capture refuses. That is why `torch.tensor(1)`,
/// `zeros(n, dtype=int64)` and even `x % 512` on an int64 tensor were all
/// uncapturable. `g2-arange` with a zero step is the same write with no
/// host in it.
pub fn fill_int(x: &Tensor, value: i64) -> Result<()> {
    let n = x.numel();
    if n == 0 {
        return Ok(());
    }
    let push = [
        u32_of(n, "numel")?,
        x.dtype().code(),
        value as u64 as u32,
        (value as u64 >> 32) as u32,
        0,
        0,
        u32_of(x.offset(), "offset")?,
    ];
    ARANGE.launch_flat(x.device(), &[Arg::Storage(x)], n, &push)
}

/// `start + i * step` in float32, for starts and steps that are not whole
/// numbers. Separate from [`arange`] because the integer form has to keep
/// 64 bits and this one has to keep a fraction.
pub fn arange_f32(device: &crate::Device, n: usize, start: f32, step: f32) -> Result<Tensor> {
    let out = Tensor::empty_typed(device, &[n], DType::F32)?;
    if n == 0 {
        return Ok(out);
    }
    let push = [u32_of(n, "numel")?, DType::F32.code(), start.to_bits(), 0, step.to_bits(), 0, 0];
    ARANGE.launch_flat(device, &[Arg::Storage(&out)], n, &push)?;
    Ok(out)
}

fn require(dtype: DType, want: DType, what: &str) -> Result<()> {
    if dtype == want {
        Ok(())
    } else {
        Err(Error::Invalid(format!("{what}: expected {} operands, got {}", want.name(), dtype.name())))
    }
}

/// `op(a, b)` on int64 tensors with broadcasting. The result is bool for
/// comparisons (and when `out_bool` is set, e.g. logical ops on bools cast
/// to int64), else int64.
pub fn binary_int(a: &Tensor, b: &Tensor, op: IntBinary, out_bool: bool) -> Result<Tensor> {
    require(a.dtype(), DType::I64, "integer op")?;
    require(b.dtype(), DType::I64, "integer op")?;
    let shape = broadcast_shape(a.shape(), b.shape())?;
    let comparison = matches!(op, IntBinary::Eq | IntBinary::Ne | IntBinary::Gt | IntBinary::Ge | IntBinary::Lt | IntBinary::Le);
    let to_bool = out_bool || comparison;
    let out = Tensor::empty_typed(a.device(), &shape, if to_bool { DType::Bool } else { DType::I64 })?;
    let mut push = vec![
        u32_of(out.numel(), "numel")?,
        op as u32,
        shape.len() as u32,
        u32_of(a.offset(), "offset")?,
        u32_of(b.offset(), "offset")?,
        0,
        u32::from(to_bool),
    ];
    push.extend(padded(&shape, "integer op")?);
    push.extend(padded(&broadcast_strides(a.shape(), &shape), "integer op")?);
    push.extend(padded(&broadcast_strides(b.shape(), &shape), "integer op")?);
    BINARY_INT.launch_flat(a.device(), &[Arg::Storage(a), Arg::Storage(b), Arg::Storage(&out)], out.numel(), &push)?;
    Ok(out)
}

/// `cond ? a : b` elementwise with broadcasting (torch's `where`); `a` and `b`
/// share a dtype, `cond` is bool. Untracked.
pub fn where_select(cond: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    require(cond.dtype(), DType::Bool, "where condition")?;
    if a.dtype() != b.dtype() {
        return Err(Error::Invalid(format!("where: {} and {} values", a.dtype().name(), b.dtype().name())));
    }
    let shape = broadcast_shape(&broadcast_shape(cond.shape(), a.shape())?, b.shape())?;
    let out = Tensor::empty_typed(a.device(), &shape, a.dtype())?;
    let mut push = vec![
        u32_of(out.numel(), "numel")?,
        shape.len() as u32,
        a.dtype().words() as u32,
        u32_of(cond.offset(), "offset")?,
        u32_of(a.offset(), "offset")?,
        u32_of(b.offset(), "offset")?,
        0,
    ];
    push.extend(padded(&shape, "where")?);
    push.extend(padded(&broadcast_strides(cond.shape(), &shape), "where")?);
    push.extend(padded(&broadcast_strides(a.shape(), &shape), "where")?);
    push.extend(padded(&broadcast_strides(b.shape(), &shape), "where")?);
    WHERE.launch_flat(a.device(), &[Arg::Storage(cond), Arg::Storage(a), Arg::Storage(b), Arg::Storage(&out)], out.numel(), &push)?;
    Ok(out)
}

/// int64 sum, max or min along `dim` (Mean is not defined for integers).
pub fn reduce_int(x: &Tensor, dim: usize, op: Reduce, keepdim: bool) -> Result<Tensor> {
    require(x.dtype(), DType::I64, "integer reduction")?;
    if op == Reduce::Mean {
        return Err(Error::Invalid("mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype.".into()));
    }
    let shape = x.shape();
    if dim >= shape.len() {
        return Err(Error::Shape(format!("reduce: dim {dim} out of range for shape {shape:?}")));
    }
    let (outer, len, inner): (usize, usize, usize) = (shape[..dim].iter().product(), shape[dim], shape[dim + 1..].iter().product());
    if len == 0 && op != Reduce::Sum {
        return Err(Error::Invalid(format!("max(): Expected reduction dim {dim} to have non-zero size.")));
    }
    let mut out_shape = shape.to_vec();
    if keepdim {
        out_shape[dim] = 1;
    } else {
        out_shape.remove(dim);
    }
    let out = Tensor::empty_typed(x.device(), &out_shape, DType::I64)?;
    if inner == 1 {
        // the reduced axis is the last one, so each output owns a
        // contiguous run and a workgroup can split it
        reduce_int_rows(x, x.offset(), outer, len, op, &out)?;
        return Ok(out);
    }
    let push = [u32_of(outer, "outer")?, u32_of(len, "len")?, u32_of(inner, "inner")?, op as u32, u32_of(x.offset(), "offset")?, 0];
    REDUCE_INT.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], outer * inner, &push)?;
    Ok(out)
}

/// Reduce `rows` contiguous int64 rows of `len` elements into `out`.
///
/// The integer twin of [`reduce_rows`], and it exists for the same reason
/// the float one does: one invocation per output is right when there are
/// many outputs and is a single serial thread when there is one long row.
/// `any()` over a 127996-element mask took 7.8 ms that way. Partials are
/// exact here, so there is no compensated pair to carry and the second
/// pass is this kernel again over plain int64.
fn reduce_int_rows(x: &Tensor, offset: usize, rows: usize, len: usize, op: Reduce, out: &Tensor) -> Result<()> {
    let wanted = if rows >= TARGET_GROUPS { 1 } else { (TARGET_GROUPS / rows).min(len.div_ceil(2048)).max(1) };
    let chunk = len.div_ceil(wanted).max(1);
    let chunks = len.div_ceil(chunk).max(1);
    let groups = rows * chunks;
    let gx = groups.min(65_535);
    let threads = [u32_of(gx * 256, "workgroups")?, u32_of(groups.div_ceil(gx), "workgroups")?, 1];
    let (rows32, len32) = (u32_of(rows, "rows")?, u32_of(len, "len")?);
    let (off, out_off) = (u32_of(offset, "offset")?, u32_of(out.offset(), "offset")?);
    if chunks == 1 {
        return REDUCE_INT_ROWS.launch(
            x.device(),
            &[Arg::Storage(x), Arg::Storage(out)],
            threads,
            &[rows32, len32, len32, 1, op as u32, off, out_off],
        );
    }
    let partial = Tensor::empty_typed(x.device(), &[groups], DType::I64)?;
    let (chunk32, chunks32) = (u32_of(chunk, "chunk")?, u32_of(chunks, "chunks")?);
    REDUCE_INT_ROWS.launch(
        x.device(),
        &[Arg::Storage(x), Arg::Storage(&partial)],
        threads,
        &[rows32, len32, chunk32, chunks32, op as u32, off, 0],
    )?;
    let threads2 = [u32_of(rows.min(65_535) * 256, "workgroups")?, u32_of(rows.div_ceil(65_535), "workgroups")?, 1];
    REDUCE_INT_ROWS.launch(
        x.device(),
        &[Arg::Storage(&partial), Arg::Storage(out)],
        threads2,
        &[rows32, chunks32, chunks32, 1, op as u32, 0, out_off],
    )
}

/// How an index tensor addresses the gathered dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexMode {
    /// torch.gather: the index has the output's shape.
    Gather = 0,
    /// torch.index_select: a 1-D index along `dim`.
    Select = 1,
}

/// The smallest and largest value of an int64 tensor (None when empty).
/// Downloads two numbers, so it waits for pending work.
pub fn index_bounds(index: &Tensor) -> Result<Option<(i64, i64)>> {
    require(index.dtype(), DType::I64, "index")?;
    if index.numel() == 0 {
        return Ok(None);
    }
    let flat = index.detach().reshape(&[index.numel()])?;
    let lo = reduce_int(&flat, 0, Reduce::Min, false)?;
    let hi = reduce_int(&flat, 0, Reduce::Max, false)?;
    let read = |t: &Tensor| -> Result<i64> { Ok(i64::from_le_bytes(t.to_bytes()?.try_into().unwrap())) };
    Ok(Some((read(&lo)?, read(&hi)?)))
}

fn index_shape(src_shape: &[usize], dim: usize, index: &Tensor, mode: IndexMode) -> Result<Vec<usize>> {
    require(index.dtype(), DType::I64, "index")?;
    if dim >= src_shape.len().max(1) {
        return Err(Error::Shape(format!("index: dim {dim} out of range for shape {src_shape:?}")));
    }
    match mode {
        IndexMode::Gather => {
            if index.dim() != src_shape.len() {
                return Err(Error::Shape(format!(
                    "Index tensor must have the same number of dimensions as input tensor ({} vs {})",
                    index.dim(),
                    src_shape.len()
                )));
            }
            for (d, (&i, &s)) in index.shape().iter().zip(src_shape).enumerate() {
                if d != dim && i > s {
                    return Err(Error::Shape(format!(
                        "Size does not match at dimension {d} expected index {:?} to be smaller than self {src_shape:?} apart from dimension {dim}",
                        index.shape()
                    )));
                }
            }
            Ok(index.shape().to_vec())
        }
        IndexMode::Select => {
            if index.dim() > 1 {
                return Err(Error::Shape("index_select(): Index is supposed to be a vector".into()));
            }
            let mut shape = src_shape.to_vec();
            shape[dim] = index.numel();
            Ok(shape)
        }
    }
}

/// `torch.gather` / `torch.index_select` along `dim`, any dtype. Untracked;
/// callers check index bounds (see [`index_bounds`]).
pub fn index_gather(src: &Tensor, dim: usize, index: &Tensor, mode: IndexMode) -> Result<Tensor> {
    let shape = index_shape(src.shape(), dim, index, mode)?;
    let out = Tensor::empty_typed(src.device(), &shape, src.dtype())?;
    let mut push = vec![
        u32_of(out.numel(), "numel")?,
        shape.len() as u32,
        dim as u32,
        src.dtype().words() as u32,
        mode as u32,
        u32_of(src.offset(), "offset")?,
        u32_of(index.offset(), "offset")?,
        0,
        u32_of(src.shape().get(dim).copied().unwrap_or(1), "dim size")?,
    ];
    push.extend(padded(&shape, "index")?);
    push.extend(padded(&contiguous_strides(src.shape()), "index")?);
    INDEX.launch_flat(src.device(), &[Arg::Storage(src), Arg::Storage(index), Arg::Storage(&out)], out.numel(), &push)?;
    Ok(out)
}

/// The backward of [`index_gather`] for float32: the gradient `g` summed
/// into a zero tensor of `in_shape` at the indexed positions. Fixed-order
/// when [`deterministic`] is on, atomic (hardware order) otherwise.
pub fn index_add_backward(g: &Tensor, dim: usize, index: &Tensor, mode: IndexMode, in_shape: &[usize]) -> Result<Tensor> {
    require_float("index backward", g.dtype())?;
    let det = deterministic();
    let out = Tensor::full(g.device(), in_shape, 0.0)?;
    let len = g.shape()[dim];
    let n = if det { out.numel() } else { g.numel() };
    let mut push = vec![
        u32_of(n, "numel")?,
        in_shape.len() as u32,
        dim as u32,
        mode as u32,
        u32_of(g.offset(), "offset")?,
        u32_of(index.offset(), "offset")?,
        0,
        u32_of(in_shape[dim], "dim size")?,
        u32_of(len, "len")?,
    ];
    push.extend(padded(in_shape, "index backward")?);
    push.extend(padded(g.shape(), "index backward")?);
    let kernel = if det { INDEX_ADD } else { INDEX_ADD_ATOMIC };
    kernel.launch_flat(g.device(), &[Arg::Storage(g), Arg::Storage(index), Arg::Storage(&out)], n, &push)?;
    Ok(out)
}

/// `argmax` (`max`) or `argmin` of a float32 or int64 tensor along `dim`,
/// as int64 with `dim` kept as size 1. Ties go to the first position, and
/// NaN is both the largest and the smallest value, as in torch.
/// Rows at least this long take the workgroup-per-row argmax.
const ARG_ROWS_MIN_LEN: usize = 256;

pub fn arg_reduce(x: &Tensor, dim: usize, max: bool) -> Result<Tensor> {
    let is_int = match x.dtype() {
        DType::F32 => 0,
        DType::I64 => 1,
        other => return Err(Error::Invalid(format!("arg reduction: expected float32 or int64, got {}", other.name()))),
    };
    let shape = x.shape();
    if dim >= shape.len() {
        return Err(Error::Shape(format!("arg reduction: dim {dim} out of range for shape {shape:?}")));
    }
    let (outer, len, inner): (usize, usize, usize) = (shape[..dim].iter().product(), shape[dim], shape[dim + 1..].iter().product());
    if len == 0 {
        return Err(Error::Invalid(format!(
            "{}(): Expected reduction dim {dim} to have non-zero size.",
            if max { "argmax" } else { "argmin" }
        )));
    }
    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let out = Tensor::empty_typed(x.device(), &out_shape, DType::I64)?;
    if inner == 1 && len >= ARG_ROWS_MIN_LEN {
        // A thread per output would scan a whole row alone: 11 ms for one
        // 152K-vocabulary row, against ~0.1 ms for a workgroup per row.
        let push = [u32_of(outer, "rows")?, u32_of(len, "len")?, max as u32, is_int, u32_of(x.offset(), "offset")?, 0];
        if outer > 0 {
            ARG_REDUCE_ROWS.launch(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], row_grid(outer)?, &push)?;
        }
        return Ok(out);
    }
    let push = [u32_of(outer, "outer")?, u32_of(len, "len")?, u32_of(inner, "inner")?, max as u32, is_int, u32_of(x.offset(), "offset")?, 0];
    ARG_REDUCE.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], outer * inner, &push)?;
    Ok(out)
}

/// The positions that sort each row of a float32 or int64 tensor along its
/// last dimension, first `take` of them, as int64 of shape (..., take). A bitonic
/// sort over rows padded to a power of two: log2(p)(log2(p)+1)/2 dispatches,
/// recorded in one batch. The order is total (equal values keep their
/// original order; NaN is largest, as in torch), so the result is
/// deterministic.
pub fn sort_positions(x: &Tensor, descending: bool, take: usize) -> Result<Tensor> {
    if let Some(out) = topk_positions(x, descending, take)? {
        return Ok(out);
    }
    let words = match x.dtype() {
        DType::F32 => 1u32,
        DType::I64 => 2,
        other => return Err(Error::Invalid(format!("sort: expected float32 or int64, got {}", other.name()))),
    };
    let (n, lead) = match x.shape().split_last() {
        Some((&n, lead)) => (n, lead),
        None => return Err(Error::Shape("sort: a 0-d tensor has no dimension to sort".into())),
    };
    if take > n {
        return Err(Error::Invalid(format!("sort: take {take} > {n}")));
    }
    let rows: usize = lead.iter().product();
    let mut out_shape = lead.to_vec();
    out_shape.push(take);
    let out = Tensor::empty_typed(x.device(), &out_shape, DType::I64)?;
    if rows == 0 || take == 0 {
        return Ok(out);
    }
    let padded = n.next_power_of_two();
    let total = rows.checked_mul(padded).filter(|&t| t <= u32::MAX as usize)
        .ok_or_else(|| Error::Shape(format!("sort: {rows} rows of {padded} exceed u32")))?;
    let keys = Tensor::empty_typed(x.device(), &[total], x.dtype())?;
    let idx = Tensor::empty_typed(x.device(), &[total], DType::I32)?;
    let (rows32, n32, p32) = (rows as u32, u32_of(n, "sort length")?, padded as u32);
    let dev = x.device();
    SORT_INIT.launch_flat(dev, &[Arg::Storage(x), Arg::Storage(&keys), Arg::Storage(&idx)], total, &[rows32, n32, p32, u32_of(x.offset(), "offset")?, words])?;
    let mut k = 2;
    while k <= padded {
        let mut j = k / 2;
        while j > 0 {
            let push = [rows32, n32, p32, j as u32, k as u32, descending as u32, words];
            SORT_STEP.launch_flat(dev, &[Arg::Storage(&keys), Arg::Storage(&idx)], total, &push)?;
            j /= 2;
        }
        k *= 2;
    }
    SORT_TAKE.launch_flat(dev, &[Arg::Storage(&idx), Arg::Storage(&out)], rows * take, &[rows32, take as u32, p32])?;
    Ok(out)
}

/// Largest k a single `g2-topk` list holds. Above this the bitonic sort
/// wins anyway: it is `log2(n)*(log2(n)+1)/2` dispatches whatever k is,
/// while this is `O(n*k)` comparisons.
pub const TOPK_MAX_K: usize = 64;

/// `Some(positions)` when `g2-topk` can answer, `None` to fall through.
///
/// The fast path takes float32 rows and `k <= TOPK_MAX_K`. int64 keys fall
/// through because the shared lists hold one word per entry and widening
/// them would halve the k that fits, for a case nothing asks for — torch's
/// `topk` on this backend is called on logits and on similarity scores.
///
/// One dispatch when a row fits one chunk, two otherwise. The chunk count
/// is chosen so every chunk has at least `take` elements, which is what
/// lets the merge assume each list can fill.
fn topk_positions(x: &Tensor, descending: bool, take: usize) -> Result<Option<Tensor>> {
    let Some((&n, lead)) = x.shape().split_last() else { return Ok(None) };
    let rows: usize = lead.iter().product();
    if x.dtype() != DType::F32 || take == 0 || take > TOPK_MAX_K || take > n || rows == 0 {
        return Ok(None);
    }
    let mut out_shape = lead.to_vec();
    out_shape.push(take);
    let out = Tensor::empty_typed(x.device(), &out_shape, DType::I64)?;

    // Three bounds, and the middle one is the one that bites. Pass two
    // merges `chunks * take` candidates in a SINGLE workgroup, so a fine
    // split moves the whole problem into that workgroup: at k=64 an
    // earlier version cut 16384 into 256 chunks and pass two then had the
    // original row back, 3.2 ms against the bitonic path's 1.0.
    let by_size = n.div_ceil(TOPK_CHUNK).max(1);        // ~TOPK_CHUNK elements each
    let by_merge = (TOPK_MERGE / take).max(1);          // pass two's input
    let chunks = if rows >= TOPK_GROUPS { 1 } else { by_size.min(by_merge).min(n / take).max(1) };
    let chunk = n.div_ceil(chunks);
    let chunks = n.div_ceil(chunk);
    let dev = x.device();
    let (rows32, n32, k32) = (u32_of(rows, "rows")?, u32_of(n, "sort length")?, take as u32);
    let in_off = u32_of(x.offset(), "offset")?;
    let desc = u32::from(descending);

    let launch = |groups: usize, args: &[Arg], push: [u32; 8]| -> Result<()> {
        let gx = groups.min(65_535);
        let threads = [u32_of(gx * 64, "workgroups")?, u32_of(groups.div_ceil(gx), "workgroups")?, 1];
        TOPK.launch(dev, args, threads, &push)
    };

    if chunks == 1 {
        let values = Tensor::empty_typed(dev, &out_shape, DType::F32)?;
        return launch(
            rows,
            &[Arg::Storage(x), Arg::Storage(&out), Arg::Storage(&values), Arg::Storage(&out)],
            [rows32, n32, k32, n32, 1, in_off, desc, 0],
        )
        .map(|()| Some(out));
    }

    // pass one: every chunk's own top-k, values and original positions
    let part_val = Tensor::empty_typed(dev, &[rows * chunks * take], DType::F32)?;
    let part_idx = Tensor::empty_typed(dev, &[rows * chunks * take], DType::I64)?;
    launch(
        rows * chunks,
        &[Arg::Storage(x), Arg::Storage(&part_idx), Arg::Storage(&part_val), Arg::Storage(&part_idx)],
        [rows32, n32, k32, u32_of(chunk, "chunk")?, u32_of(chunks, "chunks")?, in_off, desc, 0],
    )?;

    // pass two: merge the chunks, reading the positions pass one recorded
    let merged = u32_of(chunks * take, "merged")?;
    let values = Tensor::empty_typed(dev, &out_shape, DType::F32)?;
    launch(
        rows,
        &[Arg::Storage(&part_val), Arg::Storage(&part_idx), Arg::Storage(&values), Arg::Storage(&out)],
        [rows32, merged, k32, merged, 1, 0, desc, 1],
    )?;
    Ok(Some(out))
}

/// Rows above which a top-k already has enough workgroups to fill the
/// device and needs no chunking.
const TOPK_GROUPS: usize = 64;
/// Elements a chunk aims to hold, so a workgroup has real work to do.
///
/// Both of these were swept against the bitonic path over
/// (16384, 128000) x k in (8, 32, 40, 64): 1024 and 2048 win or tie every
/// cell, and the fast path is then 1.7x to 8.0x whatever k is. They pull
/// against each other — a finer split is more parallel in pass one and
/// more work for pass two, which is a single workgroup.
const TOPK_CHUNK: usize = 1024;
/// Candidates pass two may be handed, since it runs in one workgroup.
const TOPK_MERGE: usize = 2048;

/// torch's scatter, in place: `dst[c with c[dim] := index[c]] = src[c]`
/// for every position `c` of `index` (int64, bounds-checked by the caller,
/// non-negative). `src` has `dst`'s dtype and is at least `index`'s size in
/// every dim; `dst` is at least `index`'s size in every dim but `dim`.
/// Bumps `dst`'s version.
pub fn scatter(dst: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Result<()> {
    scatter_inner(dst, dim, index, src, index.shape(), false)
}

/// `dst[index[i], ...] = src[i, ...]` along `dim` — torch's `index_copy_`.
///
/// The same kernel as [`scatter`] with the index read by the `dim`
/// coordinate alone, which is what a 1-D index means. Expanding it into a
/// full-shaped index first cost a reshape, an expand and then the scatter:
/// three dispatches to write one row of a ring buffer, and a decode step
/// writes six a layer.
pub fn index_copy(dst: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Result<()> {
    if index.dim() != 1 {
        return Err(Error::Invalid(format!(
            "index_copy_(): Index is supposed to be a vector, got {}D",
            index.dim()
        )));
    }
    scatter_inner(dst, dim, index, src, src.shape(), true)
}

fn scatter_inner(dst: &Tensor, dim: usize, index: &Tensor, src: &Tensor, shape: &[usize], select: bool) -> Result<()> {
    require(index.dtype(), DType::I64, "scatter index")?;
    require(src.dtype(), dst.dtype(), "scatter source")?;
    let n: usize = shape.iter().product();
    let mut push = vec![
        u32_of(n, "numel")?,
        shape.len() as u32,
        dim as u32,
        dst.dtype().words() as u32,
        u32_of(src.offset(), "offset")?,
        u32_of(index.offset(), "offset")?,
        u32_of(dst.offset(), "offset")?,
        u32::from(select),
    ];
    push.extend(padded(shape, "scatter")?);
    push.extend(padded(&contiguous_strides(src.shape()), "scatter")?);
    push.extend(padded(&contiguous_strides(dst.shape()), "scatter")?);
    SCATTER.launch_flat(dst.device(), &[Arg::Storage(src), Arg::Storage(index), Arg::Storage(dst)], n, &push)?;
    dst.bump_version();
    Ok(())
}

/// An inclusive scan of `x` along `dim` (reversed: from the end), into a new
/// tensor. Cumsum takes float32 or int64; LogCumsumExp float32.
pub fn scan(x: &Tensor, dim: usize, op: Scan, reverse: bool) -> Result<Tensor> {
    let is_int = match (x.dtype(), op) {
        (DType::F32, _) => 0,
        (DType::I64, Scan::Cumsum) => 1,
        (other, _) => return Err(Error::Invalid(format!("{op:?}: unsupported dtype {}", other.name()))),
    };
    let shape = x.shape();
    if dim >= shape.len() {
        return Err(Error::Shape(format!("scan: dim {dim} out of range for shape {shape:?}")));
    }
    let (outer, len, inner): (usize, usize, usize) = (shape[..dim].iter().product(), shape[dim], shape[dim + 1..].iter().product());
    let out = Tensor::empty_typed(x.device(), shape, x.dtype())?;
    let push = [
        u32_of(outer, "outer")?,
        u32_of(len, "len")?,
        u32_of(inner, "inner")?,
        op as u32,
        reverse as u32,
        is_int,
        u32_of(x.offset(), "offset")?,
        0,
    ];
    SCAN.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], outer * inner, &push)?;
    Ok(out)
}

/// Batched matrix product of float32 tensors: `op(a) @ op(b)` over the
/// last two dims, where `op` transposes when asked (`a` is then stored
/// (..., K, M), `b` (..., N, K)). Batch dims broadcast as in torch, up to
/// [`MAX_BATCH_DIMS`]; the result is (broadcast batch..., M, N).
pub fn bmm(a: &Tensor, b: &Tensor, trans_a: bool, trans_b: bool) -> Result<Tensor> {
    require_float("matmul", a.dtype())?;
    require_float("matmul", b.dtype())?;
    if a.dim() < 2 || b.dim() < 2 {
        return Err(Error::Shape("bmm: both operands need at least 2 dims".into()));
    }
    let (ra, ca) = (a.shape()[a.dim() - 2], a.shape()[a.dim() - 1]);
    let (rb, cb) = (b.shape()[b.dim() - 2], b.shape()[b.dim() - 1]);
    let (m, ka) = if trans_a { (ca, ra) } else { (ra, ca) };
    let (kb, n) = if trans_b { (cb, rb) } else { (rb, cb) };
    if ka != kb {
        return Err(Error::Shape(format!(
            "Expected size for first two dimensions of batch2 tensor to be: [{ka}] but got: [{kb}] \
             (matmul of {:?} and {:?})",
            a.shape(),
            b.shape()
        )));
    }
    let (a_batch, b_batch) = (&a.shape()[..a.dim() - 2], &b.shape()[..b.dim() - 2]);
    let batch_shape = broadcast_shape(a_batch, b_batch)?;
    if batch_shape.len() > MAX_BATCH_DIMS {
        return Err(Error::NotImplemented(format!("matmul with more than {MAX_BATCH_DIMS} batch dimensions")));
    }
    let batch: usize = batch_shape.iter().product();
    let mut out_shape = batch_shape.clone();
    out_shape.extend([m, n]);
    let out = Tensor::empty(a.device(), &out_shape)?;
    if batch == 0 || m == 0 || n == 0 {
        return Ok(out);
    }
    if ka == 0 {
        return Tensor::zeros(a.device(), &out_shape);
    }
    if batch > 65_535 {
        return Err(Error::NotImplemented(format!("matmul over {batch} batch elements (the limit is 65535)")));
    }
    // element strides of each operand's batch dims, 0 where it broadcasts
    let strides = |own: &[usize], matrix: usize| -> Vec<usize> {
        let lead = batch_shape.len() - own.len();
        let mut st = vec![0; batch_shape.len()];
        let mut step = matrix;
        for d in (0..batch_shape.len()).rev() {
            if d >= lead {
                let size = own[d - lead];
                st[d] = if size == 1 { 0 } else { step };
                step *= size;
            }
        }
        st
    };
    let (sa, sb) = (strides(a_batch, ra * ca), strides(b_batch, rb * cb));
    let pad = |v: &[usize], fill: usize| -> Result<Vec<u32>> {
        let mut out = v.iter().map(|&x| u32_of(x, "matmul")).collect::<Result<Vec<_>>>()?;
        out.resize(MAX_BATCH_DIMS, fill as u32);
        Ok(out)
    };
    let mut push = vec![
        u32_of(m, "M")?,
        u32_of(n, "N")?,
        u32_of(ka, "K")?,
        trans_a as u32,
        trans_b as u32,
        u32_of(a.offset(), "offset")?,
        u32_of(b.offset(), "offset")?,
        0,
        batch_shape.len() as u32,
    ];
    push.extend(pad(&batch_shape, 1)?);
    push.extend(pad(&sa, 0)?);
    push.extend(pad(&sb, 0)?);
    let threads = [u32_of(n.next_multiple_of(16), "N")?, u32_of(m.next_multiple_of(16), "M")?, batch as u32];
    BMM.launch(a.device(), &[Arg::Storage(a), Arg::Storage(b), Arg::Storage(&out)], threads, &push)?;
    Ok(out)
}

/// A workgroup per row, for `rows` rows (a 2-D grid past 65535).
fn row_grid(rows: usize) -> Result<[u32; 3]> {
    let gx = rows.min(65_535);
    Ok([u32_of(gx * 256, "rows")?, u32_of(rows.div_ceil(gx), "rows")?, 1])
}

/// layer_norm over the last `n` elements of each row of float32 `x`:
/// `(x - mean) * rstd * weight + bias`, with the biased variance and
/// `rstd = 1 / sqrt(var + eps)`, as torch. Returns the output and the
/// per-row `(mean, rstd)` pairs, (rows, 2), for [`layer_norm_backward`].
pub fn layer_norm(x: &Tensor, n: usize, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Result<(Tensor, Tensor)> {
    require_float("layer_norm", x.dtype())?;
    let rows = if n == 0 { 0 } else { x.numel() / n };
    let y = Tensor::empty(x.device(), x.shape())?;
    let stats = Tensor::empty(x.device(), &[rows, 2])?;
    if rows == 0 {
        return Ok((y, stats));
    }
    for (t, what) in [(weight, "weight"), (bias, "bias")] {
        if let Some(t) = t {
            require_float("layer_norm", t.dtype())?;
            if t.numel() != n {
                return Err(Error::Shape(format!("layer_norm: {what} has {} elements, expected {n}", t.numel())));
            }
        }
    }
    let affine = weight.is_some() as u32 | (bias.is_some() as u32) << 1;
    let (w, b) = (weight.unwrap_or(x), bias.unwrap_or(x));
    let push = [
        u32_of(rows, "rows")?,
        u32_of(n, "n")?,
        eps.to_bits(),
        affine,
        u32_of(x.offset(), "offset")?,
        u32_of(w.offset(), "offset")?,
        u32_of(b.offset(), "offset")?,
    ];
    let args = [Arg::Storage(x), Arg::Storage(w), Arg::Storage(b), Arg::Storage(&y), Arg::Storage(&stats)];
    LAYER_NORM.launch(x.device(), &args, row_grid(rows)?, &push)?;
    Ok((y, stats))
}

/// rms_norm over the last `n` elements of each row of float32 `x`, torch's
/// formula `h * (1 / sqrt(mean(h^2) + eps)) * weight`, in one kernel. `h`
/// is `x`, or `x + residual` when a residual is given, and is then returned
/// too (a decoder layer's add-then-normalize). No autograd: F.rms_norm
/// composes grilly ops when a gradient is needed.
pub fn rms_norm(
    x: &Tensor,
    n: usize,
    residual: Option<&Tensor>,
    weight: Option<&Tensor>,
    eps: f32,
) -> Result<(Tensor, Option<Tensor>)> {
    require_float("rms_norm", x.dtype())?;
    if n == 0 || !x.numel().is_multiple_of(n) {
        return Err(Error::Shape(format!("rms_norm: {:?} is not rows of {n}", x.shape())));
    }
    if let Some(r) = residual {
        require_float("rms_norm", r.dtype())?;
        if r.shape() != x.shape() {
            return Err(Error::Shape(format!("rms_norm: residual {:?} does not match input {:?}", r.shape(), x.shape())));
        }
    }
    if let Some(w) = weight {
        require_float("rms_norm", w.dtype())?;
        if w.numel() != n {
            return Err(Error::Shape(format!("rms_norm: weight has {} elements, expected {n}", w.numel())));
        }
    }
    let rows = x.numel() / n;
    let y = Tensor::empty(x.device(), x.shape())?;
    let h = residual.map(|_| Tensor::empty(x.device(), x.shape())).transpose()?;
    if rows == 0 {
        return Ok((y, h));
    }
    let flags = weight.is_some() as u32 | (residual.is_some() as u32) << 1;
    let (r, w) = (residual.unwrap_or(x), weight.unwrap_or(x));
    let push = [
        u32_of(rows, "rows")?,
        u32_of(n, "n")?,
        eps.to_bits(),
        flags,
        u32_of(x.offset(), "offset")?,
        u32_of(r.offset(), "offset")?,
        u32_of(w.offset(), "offset")?,
    ];
    let args = [Arg::Storage(x), Arg::Storage(r), Arg::Storage(w), Arg::Tensor(&y), Arg::Tensor(h.as_ref().unwrap_or(&y))];
    RMS_NORM.launch(x.device(), &args, row_grid(rows)?, &push)?;
    Ok((y, h))
}

/// The input gradient of [`layer_norm`] from the output gradient `g`, and,
/// with `want_gxhat`, `g * xhat` per element (its column sums over rows are
/// the weight's gradient).
pub fn layer_norm_backward(
    g: &Tensor,
    x: &Tensor,
    weight: Option<&Tensor>,
    stats: &Tensor,
    n: usize,
    want_gxhat: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let rows = stats.shape()[0];
    let dx = Tensor::empty(x.device(), x.shape())?;
    let gxhat = if want_gxhat { Some(Tensor::empty(x.device(), x.shape())?) } else { None };
    if rows == 0 {
        return Ok((dx, gxhat));
    }
    let w = weight.unwrap_or(x);
    let push = [
        u32_of(rows, "rows")?,
        u32_of(n, "n")?,
        weight.is_some() as u32,
        want_gxhat as u32,
        u32_of(g.offset(), "offset")?,
        u32_of(x.offset(), "offset")?,
        u32_of(w.offset(), "offset")?,
    ];
    let gx_arg = gxhat.as_ref().unwrap_or(&dx);
    let args = [Arg::Storage(g), Arg::Storage(x), Arg::Storage(w), Arg::Storage(stats), Arg::Storage(&dx), Arg::Storage(gx_arg)];
    LAYER_NORM_BACKWARD.launch(x.device(), &args, row_grid(rows)?, &push)?;
    Ok((dx, gxhat))
}
