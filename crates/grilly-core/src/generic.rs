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
pub const REDUCE: Kernel = Kernel { shader: "g2-reduce", bindings: 2, push_words: 6, local_size: [256, 1, 1] };
pub const REDUCE_ROWS: Kernel = Kernel { shader: "g2-reduce-rows", bindings: 2, push_words: 10, local_size: [256, 1, 1] };
pub const COPY: Kernel = Kernel { shader: "g2-copy", bindings: 2, push_words: 4 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };

pub const CAST: Kernel = Kernel { shader: "g2-cast", bindings: 2, push_words: 5, local_size: [256, 1, 1] };
pub const BINARY_INT: Kernel = Kernel { shader: "g2-binary-int", bindings: 3, push_words: 7 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const WHERE: Kernel = Kernel { shader: "g2-where", bindings: 4, push_words: 7 + 4 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const REDUCE_INT: Kernel = Kernel { shader: "g2-reduce-int", bindings: 2, push_words: 6, local_size: [256, 1, 1] };

pub const INDEX: Kernel = Kernel { shader: "g2-index", bindings: 3, push_words: 9 + 2 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const INDEX_ADD: Kernel = Kernel { shader: "g2-index-add", bindings: 3, push_words: 9 + 2 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const INDEX_ADD_ATOMIC: Kernel = Kernel { shader: "g2-index-add-atomic", bindings: 3, push_words: 9 + 2 * MAX_DIMS as u32, local_size: [256, 1, 1] };

pub const ARG_REDUCE: Kernel = Kernel { shader: "g2-argreduce", bindings: 2, push_words: 7, local_size: [256, 1, 1] };
pub const SORT_INIT: Kernel = Kernel { shader: "g2-sort-init", bindings: 3, push_words: 5, local_size: [256, 1, 1] };
pub const SORT_STEP: Kernel = Kernel { shader: "g2-sort-step", bindings: 2, push_words: 7, local_size: [256, 1, 1] };
pub const SORT_TAKE: Kernel = Kernel { shader: "g2-sort-take", bindings: 2, push_words: 3, local_size: [256, 1, 1] };

pub const SCATTER: Kernel = Kernel { shader: "g2-scatter", bindings: 3, push_words: 8 + 3 * MAX_DIMS as u32, local_size: [256, 1, 1] };
pub const SCAN: Kernel = Kernel { shader: "g2-scan", bindings: 2, push_words: 8, local_size: [256, 1, 1] };

pub const BMM: Kernel = Kernel { shader: "g2-bmm", bindings: 3, push_words: 9 + 3 * MAX_BATCH_DIMS as u32, local_size: [16, 16, 1] };

pub const ALL: &[Kernel] = &[
    UNARY, BINARY, REDUCE, COPY, CAST, BINARY_INT, WHERE, REDUCE_INT, INDEX, INDEX_ADD, INDEX_ADD_ATOMIC,
    ARG_REDUCE, SORT_INIT, SORT_STEP, SORT_TAKE, SCATTER, SCAN, BMM, REDUCE_ROWS,
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
    } else if len > 0 && outer * inner < COLUMN_THREADS {
        // too few outputs to fill the GPU with a thread each: make the
        // reduced dim contiguous, then reduce rows
        let moved = Tensor::empty(x.device(), &[outer, inner, len])?;
        copy_strided(x, x.offset(), &[len * inner, 1, inner], &moved, 0, &[inner * len, len, 1], &[outer, inner, len])?;
        reduce_rows(&moved, 0, outer * inner, len, op, &out)?;
    } else {
        // a thread per output; neighbouring outputs read neighbouring addresses
        let push = [
            u32_of(outer, "outer")?,
            u32_of(len, "len")?,
            u32_of(inner, "inner")?,
            op as u32,
            u32_of(x.offset(), "offset")?,
            0,
        ];
        REDUCE.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], outer * inner, &push)?;
    }
    Ok(out)
}

/// Below this many outputs, a strided (non-last-dim) reduction is copied
/// to contiguous rows first rather than given one thread per output.
const COLUMN_THREADS: usize = 8192;

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
    let push = [u32_of(outer, "outer")?, u32_of(len, "len")?, u32_of(inner, "inner")?, op as u32, u32_of(x.offset(), "offset")?, 0];
    REDUCE_INT.launch_flat(x.device(), &[Arg::Storage(x), Arg::Storage(&out)], outer * inner, &push)?;
    Ok(out)
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

/// torch's scatter, in place: `dst[c with c[dim] := index[c]] = src[c]`
/// for every position `c` of `index` (int64, bounds-checked by the caller,
/// non-negative). `src` has `dst`'s dtype and is at least `index`'s size in
/// every dim; `dst` is at least `index`'s size in every dim but `dim`.
/// Bumps `dst`'s version.
pub fn scatter(dst: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Result<()> {
    require(index.dtype(), DType::I64, "scatter index")?;
    require(src.dtype(), dst.dtype(), "scatter source")?;
    let mut push = vec![
        u32_of(index.numel(), "numel")?,
        index.dim() as u32,
        dim as u32,
        dst.dtype().words() as u32,
        u32_of(src.offset(), "offset")?,
        u32_of(index.offset(), "offset")?,
        u32_of(dst.offset(), "offset")?,
        0,
    ];
    push.extend(padded(index.shape(), "scatter")?);
    push.extend(padded(&contiguous_strides(src.shape()), "scatter")?);
    push.extend(padded(&contiguous_strides(dst.shape()), "scatter")?);
    SCATTER.launch_flat(dst.device(), &[Arg::Storage(src), Arg::Storage(index), Arg::Storage(dst)], index.numel(), &push)?;
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
