//! Shape ops. torch returns views for transpose, permute, expand and
//! indexing; grilly has no strided tensors yet, so these return copies
//! marked as view copies (in-place writes through them are refused rather
//! than silently missing the original). Gradients are torch's.

use grilly_core::generic::{self as k, contiguous_strides, IndexMode};
use grilly_core::{DType, Error, Result, Tensor};

use crate::math::record;

fn strides_of(t: &Tensor) -> Vec<usize> {
    contiguous_strides(t.shape())
}

/// Untracked: `t` materialized at `shape` under torch's broadcasting rule.
pub fn broadcast_raw(t: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if t.shape() == shape {
        return Ok(t.clone());
    }
    let ok = t.dim() <= shape.len()
        && t.shape().iter().rev().zip(shape.iter().rev()).all(|(&a, &b)| a == b || a == 1);
    if !ok {
        return Err(Error::Shape(format!(
            "The expanded size of the tensor must match the existing size: cannot expand {:?} to {shape:?}",
            t.shape()
        )));
    }
    let own = strides_of(t);
    let lead = shape.len() - t.dim();
    let src: Vec<usize> = (0..shape.len())
        .map(|i| if i < lead || t.shape()[i - lead] == 1 { 0 } else { own[i - lead] })
        .collect();
    let out = Tensor::empty_typed(t.device(), shape, t.dtype())?;
    k::copy_strided(t, t.offset(), &src, &out, 0, &contiguous_strides(shape), shape)?;
    Ok(out)
}

/// `x.expand(shape)` (already resolved: no -1 entries).
pub fn expand(x: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let mut out = broadcast_raw(x, shape)?;
    if out.same_tensor(x) {
        return Ok(out);
    }
    out.mark_view_copy();
    let in_shape = x.shape().to_vec();
    record("ExpandBackward0", &[x], &mut out, &[], false, move |g, _, _| Ok(vec![Some(k::sum_to_shape(g, &in_shape)?)]))?;
    Ok(out)
}

fn permute_raw(x: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let own = strides_of(x);
    let shape: Vec<usize> = dims.iter().map(|&d| x.shape()[d]).collect();
    let src: Vec<usize> = dims.iter().map(|&d| own[d]).collect();
    let out = Tensor::empty_typed(x.device(), &shape, x.dtype())?;
    k::copy_strided(x, x.offset(), &src, &out, 0, &contiguous_strides(&shape), &shape)?;
    Ok(out)
}

fn permute_named(x: &Tensor, dims: &[usize], name: &'static str) -> Result<Tensor> {
    let mut seen = vec![false; x.dim()];
    if dims.len() != x.dim() || dims.iter().any(|&d| d >= x.dim() || std::mem::replace(&mut seen[d], true)) {
        return Err(Error::Shape(format!("permute: {dims:?} is not a permutation of the dims of a {}-d tensor", x.dim())));
    }
    let mut out = permute_raw(x, dims)?;
    out.mark_view_copy();
    let mut inverse = vec![0; dims.len()];
    for (i, &d) in dims.iter().enumerate() {
        inverse[d] = i;
    }
    record(name, &[x], &mut out, &[], false, move |g, _, _| Ok(vec![Some(permute_raw(g, &inverse)?)]))?;
    Ok(out)
}

/// `x.permute(dims)`.
pub fn permute(x: &Tensor, dims: &[usize]) -> Result<Tensor> {
    permute_named(x, dims, "PermuteBackward0")
}

/// `x.transpose(d0, d1)`.
pub fn transpose(x: &Tensor, d0: usize, d1: usize) -> Result<Tensor> {
    let mut dims: Vec<usize> = (0..x.dim()).collect();
    if d0 >= x.dim() || d1 >= x.dim() {
        return Err(Error::Shape(format!("transpose: dims {d0}, {d1} out of range for a {}-d tensor", x.dim())));
    }
    dims.swap(d0, d1);
    permute_named(x, &dims, "TransposeBackward0")
}

/// Read the elements at `offset + c . strides` for every coordinate `c` of
/// `shape` (offset relative to `x`'s first element): torch's basic indexing
/// and slicing. `name` is the grad_fn torch reports.
pub fn strided_read(x: &Tensor, offset: usize, strides: &[usize], shape: &[usize], name: &'static str) -> Result<Tensor> {
    let mut out = Tensor::empty_typed(x.device(), shape, x.dtype())?;
    k::copy_strided(x, x.offset() + offset, strides, &out, 0, &contiguous_strides(shape), shape)?;
    out.mark_view_copy();
    let (in_shape, strides, shape) = (x.shape().to_vec(), strides.to_vec(), shape.to_vec());
    record(name, &[x], &mut out, &[], false, move |g, _, _| {
        let grad = Tensor::zeros(g.device(), &in_shape)?;
        k::copy_strided(g, g.offset(), &contiguous_strides(&shape), &grad, offset, &strides, &shape)?;
        Ok(vec![Some(grad)])
    })?;
    Ok(out)
}

/// `x.narrow(dim, start, length)`.
pub fn narrow(x: &Tensor, dim: usize, start: usize, length: usize) -> Result<Tensor> {
    if dim >= x.dim() || start + length > x.shape()[dim] {
        return Err(Error::Shape(format!(
            "start ({start}) + length ({length}) exceeds dimension size of {:?} at dim {dim}",
            x.shape()
        )));
    }
    let own = strides_of(x);
    let mut shape = x.shape().to_vec();
    shape[dim] = length;
    strided_read(x, start * own[dim], &own, &shape, "SliceBackward0")
}

/// `x.select(dim, index)`: the slice at `index`, with `dim` removed.
pub fn select(x: &Tensor, dim: usize, index: usize) -> Result<Tensor> {
    if dim >= x.dim() || index >= x.shape()[dim] {
        return Err(Error::Shape(format!("index {index} is out of bounds for dimension {dim} of {:?}", x.shape())));
    }
    let own = strides_of(x);
    let mut shape = x.shape().to_vec();
    let mut strides = own.clone();
    shape.remove(dim);
    strides.remove(dim);
    strided_read(x, index * own[dim], &strides, &shape, "SelectBackward0")
}

/// Write `value` (broadcast to `shape`) into `dst` at `offset + c . strides`:
/// torch's `dst[index] = value`. An in-place op: torch's rules apply.
pub fn write_region(dst: &Tensor, offset: usize, strides: &[usize], shape: &[usize], value: &Tensor) -> Result<()> {
    dst.check_inplace(&[value])?;
    // torch converts the value to the destination's dtype.
    let v = broadcast_raw(&value.cast(dst.dtype())?, shape)?;
    k::write_strided(&v, v.offset(), &contiguous_strides(shape), dst, dst.offset() + offset, strides, shape)
}

fn check_cat(xs: &[&Tensor], dim: usize, what: &str) -> Result<()> {
    let first = xs.first().ok_or_else(|| Error::Invalid(format!("{what}(): expected a non-empty list of Tensors")))?;
    if xs.iter().any(|t| t.dtype() != first.dtype()) {
        return Err(Error::NotImplemented(format!("{what}() of tensors with different dtypes (type promotion)")));
    }
    for t in xs {
        let same = t.dim() == first.dim()
            && t.shape().iter().zip(first.shape()).enumerate().all(|(i, (a, b))| a == b || (what == "cat" && i == dim));
        if !same {
            return Err(Error::Shape(format!(
                "{what}(): tensors must have the same shape{} (got {:?} and {:?})",
                if what == "cat" { " except in the concatenating dimension" } else { "" },
                first.shape(),
                t.shape()
            )));
        }
    }
    Ok(())
}

/// `torch.cat(xs, dim)`.
pub fn cat(xs: &[&Tensor], dim: usize) -> Result<Tensor> {
    check_cat(xs, dim, "cat")?;
    if dim >= xs[0].dim() {
        return Err(Error::Shape(format!("cat: dim {dim} out of range for {}-d tensors", xs[0].dim())));
    }
    let mut shape = xs[0].shape().to_vec();
    shape[dim] = xs.iter().map(|t| t.shape()[dim]).sum();
    let mut out = Tensor::empty_typed(xs[0].device(), &shape, xs[0].dtype())?;
    let out_strides = contiguous_strides(&shape);
    let mut starts = Vec::with_capacity(xs.len());
    let mut at = 0;
    for t in xs {
        starts.push((at, t.shape()[dim]));
        k::copy_strided(t, t.offset(), &strides_of(t), &out, at * out_strides[dim], &out_strides, t.shape())?;
        at += t.shape()[dim];
    }
    record("CatBackward0", xs, &mut out, &[], false, move |g, _, needs| {
        starts
            .iter()
            .zip(needs)
            .map(|(&(start, len), &need)| if need { narrow(g, dim, start, len).map(|t| Some(t.detach())) } else { Ok(None) })
            .collect()
    })?;
    Ok(out)
}

/// `torch.stack(xs, dim)`.
pub fn stack(xs: &[&Tensor], dim: usize) -> Result<Tensor> {
    check_cat(xs, dim, "stack")?;
    if dim > xs[0].dim() {
        return Err(Error::Shape(format!("stack: dim {dim} out of range for {}-d tensors", xs[0].dim())));
    }
    let mut shape = xs[0].shape().to_vec();
    shape.insert(dim, xs.len());
    let mut out = Tensor::empty_typed(xs[0].device(), &shape, xs[0].dtype())?;
    let out_strides = contiguous_strides(&shape);
    let mut dst_strides = out_strides.clone();
    dst_strides.remove(dim);
    for (i, t) in xs.iter().enumerate() {
        k::copy_strided(t, t.offset(), &strides_of(t), &out, i * out_strides[dim], &dst_strides, t.shape())?;
    }
    record("StackBackward0", xs, &mut out, &[], false, move |g, _, needs| {
        needs
            .iter()
            .enumerate()
            .map(|(i, &need)| if need { select(g, dim, i).map(|t| Some(t.detach())) } else { Ok(None) })
            .collect()
    })?;
    Ok(out)
}

/// `torch.matmul(a, b)` for the cases grilly supports: vectors, matrices,
/// batched matrices with equal batch dims, and a batched left operand
/// against a matrix. Broadcast batch dims are not implemented yet.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    use crate::math::{binary, sum};
    use grilly_core::generic::Binary;
    match (a.dim(), b.dim()) {
        (0, _) | (_, 0) => Err(Error::Invalid("both arguments to matmul need to be at least 1D".into())),
        (1, 1) => sum(&binary(a, b, Binary::Mul)?, &[], false),
        (1, _) => {
            // a row vector: (1, K) @ b, then drop that dim
            let r = matmul(&a.reshape(&[1, a.shape()[0]])?, b)?;
            let mut shape = r.shape().to_vec();
            shape.remove(shape.len() - 2);
            r.reshape(&shape)
        }
        (_, 1) => {
            let r = matmul(a, &b.reshape(&[b.shape()[0], 1])?)?;
            r.reshape(&a.shape()[..a.dim() - 1])
        }
        // NOT `linear(a, b^T)`: transposing b materializes a copy of the
        // whole matrix, which costs more than the matmul reads. The kernel
        // takes b as it is (measured: faster than linear even before the
        // copy is counted, and bit-identical).
        _ => bmm(a, b, false),
    }
}

/// `a @ b^T` over the last two dims, without materializing the transpose.
///
/// This is what `x @ w.t()` means, and what torch hands BLAS as a transpose
/// flag. Building `w.t()` first would copy the whole matrix with a stride
/// that defeats the cache — measured at 6.1 GB/s against 196 for the same
/// data the other way round — and then the matmul reads it again.
///
/// A 2-D `b` is exactly `linear(a, b)`, which has a kernel shaped for the
/// one-row case that decode lives in.
pub fn matmul_t(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match (a.dim(), b.dim()) {
        (0, _) | (_, 0) | (_, 1) => Err(Error::Invalid(
            "matmul with a transposed operand needs both arguments to be at least 2-D".into(),
        )),
        (1, _) => {
            let r = matmul_t(&a.reshape(&[1, a.shape()[0]])?, b)?;
            let mut shape = r.shape().to_vec();
            shape.remove(shape.len() - 2);
            r.reshape(&shape)
        }
        (_, 2) => crate::linear(a, b, None),
        _ => bmm(a, b, true),
    }
}

/// Batched (and batch-broadcast) matmul as one kernel and one autograd
/// node. With `trans_b` it computes `a @ b^T` instead, reading `b` in place
/// rather than through a materialized transpose.
///
/// The gradients follow from which one it computed:
/// - `a @ b`  : dA = g @ b^T, dB = a^T @ g
/// - `a @ b^T`: dA = g @ b,   dB = g^T @ a
fn bmm(a: &Tensor, b: &Tensor, trans_b: bool) -> Result<Tensor> {
    let mut out = k::bmm(a, b, false, trans_b)?;
    let (sa, sb) = (a.shape().to_vec(), b.shape().to_vec());
    let name = if trans_b { "BmmTransposedBackward0" } else { "BmmBackward0" };
    record(name, &[a, b], &mut out, &[a, b], false, move |g, s, needs| {
        let grad_a = if needs[0] {
            Some(k::sum_to_shape(&k::bmm(g, &s[1], false, !trans_b)?, &sa)?)
        } else {
            None
        };
        let grad_b = if needs[1] {
            let raw = if trans_b { k::bmm(g, &s[0], true, false)? } else { k::bmm(&s[0], g, true, false)? };
            Some(k::sum_to_shape(&raw, &sb)?)
        } else {
            None
        };
        Ok(vec![grad_a, grad_b])
    })?;
    Ok(out)
}

// ── indexing with index tensors ─────────────────────────────────────────

/// Index tensors may be int64 or int32 (torch accepts both).
fn as_index(index: &Tensor, op: &str) -> Result<Tensor> {
    match index.dtype() {
        DType::I64 => Ok(index.clone()),
        DType::I32 => index.cast(DType::I64),
        _ => Err(Error::Invalid(format!("{op}(): Expected dtype int32/int64 for index"))),
    }
}

/// The values of `index` lie in [lo, size) (lo = -size when negative indices
/// count from the end, else 0); otherwise `error(bad value)`.
fn check_bounds(index: &Tensor, size: usize, allow_negative: bool, error: impl Fn(i64) -> Error) -> Result<()> {
    let lo = if allow_negative { -(size as i64) } else { 0 };
    if let Some((min, max)) = k::index_bounds(index)? {
        if min < lo {
            return Err(error(min));
        }
        if max >= size as i64 {
            return Err(error(max));
        }
    }
    Ok(())
}

fn indexed(src: &Tensor, dim: usize, index: &Tensor, mode: IndexMode, name: &'static str) -> Result<Tensor> {
    let mut out = k::index_gather(src, dim, index, mode)?;
    let (idx, in_shape) = (index.detach(), src.shape().to_vec());
    record(name, &[src], &mut out, &[], false, move |g, _, _| {
        Ok(vec![Some(k::index_add_backward(g, dim, &idx, mode, &in_shape)?)])
    })?;
    Ok(out)
}

/// `torch.gather(input, dim, index)`.
pub fn gather(input: &Tensor, dim: usize, index: &Tensor) -> Result<Tensor> {
    let index = as_index(index, "gather")?;
    let size = *input.shape().get(dim).ok_or_else(|| Error::Shape(format!("gather: dim {dim} out of range")))?;
    check_bounds(&index, size, false, |v| {
        Error::Invalid(format!("index {v} is out of bounds for dimension {dim} with size {size}"))
    })?;
    indexed(input, dim, &index, IndexMode::Gather, "GatherBackward0")
}

/// `torch.index_select(input, dim, index)`.
pub fn index_select(input: &Tensor, dim: usize, index: &Tensor) -> Result<Tensor> {
    let index = as_index(index, "index_select")?;
    let size = *input.shape().get(dim).ok_or_else(|| Error::Shape(format!("index_select: dim {dim} out of range")))?;
    check_bounds(&index, size, false, |v| {
        Error::Invalid(format!("INDICES element is out of DATA bounds, id={v} axis_dim={size}"))
    })?;
    indexed(input, dim, &index, IndexMode::Select, "IndexSelectBackward0")
}

/// Advanced indexing `x[index]` along `dim` (a tensor index, any shape):
/// negative values count from the end, as in torch.
pub fn index_tensor(input: &Tensor, dim: usize, index: &Tensor) -> Result<Tensor> {
    let index = as_index(index, "index")?;
    let size = input.shape()[dim];
    check_bounds(&index, size, true, |v| {
        Error::Index(format!("index {v} is out of bounds for dimension {dim} with size {size}"))
    })?;
    let flat = index.detach().reshape(&[index.numel()])?;
    let picked = indexed(input, dim, &flat, IndexMode::Select, "IndexBackward0")?;
    let mut shape = input.shape()[..dim].to_vec();
    shape.extend_from_slice(index.shape());
    shape.extend_from_slice(&input.shape()[dim + 1..]);
    picked.reshape(&shape)
}

/// `F.embedding(input, weight, padding_idx)`: rows of `weight`. The
/// padding row gets no gradient, as in torch.
pub fn embedding(input: &Tensor, weight: &Tensor, padding_idx: Option<usize>) -> Result<Tensor> {
    let index = as_index(input, "embedding")?;
    if weight.dim() != 2 {
        return Err(Error::Shape("'weight' must be 2-D".into()));
    }
    let (rows, width) = (weight.shape()[0], weight.shape()[1]);
    check_bounds(&index, rows, false, |_| Error::Index("index out of range in self".into()))?;
    let flat = index.detach().reshape(&[index.numel()])?;
    let mut out = k::index_gather(weight, 0, &flat, IndexMode::Select)?;
    let w_shape = weight.shape().to_vec();
    record("EmbeddingBackward0", &[weight], &mut out, &[], false, move |g, _, _| {
        let grad = k::index_add_backward(g, 0, &flat, IndexMode::Select, &w_shape)?;
        if let Some(p) = padding_idx {
            let zeros = Tensor::zeros(grad.device(), &[width])?;
            k::copy_strided(&zeros, 0, &[1], &grad, p * width, &[1], &[width])?;
        }
        Ok(vec![Some(grad)])
    })?;
    let mut shape = index.shape().to_vec();
    shape.push(width);
    out.reshape(&shape)
}

// ── extremes and ordering ───────────────────────────────────────────────

/// argmax (`max`) / argmin along `dim` as int64: the first extreme, NaN
/// counting as the extreme, as in torch.
pub fn argmax(x: &Tensor, dim: usize, keepdim: bool, max: bool) -> Result<Tensor> {
    check_reducible(x, dim, if max { "argmax" } else { "argmin" })?;
    let src = match x.dtype() {
        DType::F32 | DType::I64 => x.detach(),
        _ => x.detach().cast(DType::I64)?,
    };
    let idx = k::arg_reduce(&src, dim, max)?;
    if keepdim {
        Ok(idx)
    } else {
        idx.reshape(&dropped(x.shape(), dim))
    }
}

/// torch's IndexError for reducing an empty (or missing) dimension.
fn check_reducible(x: &Tensor, dim: usize, op: &str) -> Result<()> {
    match x.shape().get(dim) {
        Some(0) => Err(Error::Index(format!("{op}(): Expected reduction dim {dim} to have non-zero size."))),
        Some(_) => Ok(()),
        None => Err(Error::Index(format!("{op}(): dim {dim} out of range for {} dimensions", x.dim()))),
    }
}

fn dropped(shape: &[usize], dim: usize) -> Vec<usize> {
    let mut s = shape.to_vec();
    s.remove(dim);
    s
}

/// `x.max(dim)` / `x.min(dim)`: (values, int64 indices). The values are
/// gathered at the indices, so the gradient goes to the chosen element,
/// as torch's max(dim) backward does.
pub fn max_dim(x: &Tensor, dim: usize, keepdim: bool, max: bool) -> Result<(Tensor, Tensor)> {
    check_reducible(x, dim, if max { "max" } else { "min" })?;
    let idx = argmax(x, dim, true, max)?;
    let values = indexed(x, dim, &idx, IndexMode::Gather, if max { "MaxBackward0" } else { "MinBackward0" })?;
    if keepdim {
        return Ok((values, idx));
    }
    let shape = dropped(x.shape(), dim);
    Ok((values.reshape(&shape)?, idx.reshape(&shape)?))
}

/// The first `take` elements of `x` in sorted order along `dim`, and their
/// positions: `sort` (take = the whole dim) and `topk`. Equal values keep
/// their original order and NaN sorts as the largest value, so the result
/// is deterministic; values are gathered from `x`, so they keep its dtype
/// and gradients are torch's. int32 and bool sort as int64 keys.
pub fn sorted(x: &Tensor, dim: usize, descending: bool, take: usize, name: &'static str) -> Result<(Tensor, Tensor)> {
    if x.dim() == 0 {
        let (v, i) = sorted(&x.reshape(&[1])?, 0, descending, take, name)?;
        return Ok((v.reshape(&[])?, i.reshape(&[])?));
    }
    let keys = match x.dtype() {
        DType::F32 | DType::I64 => x.detach(),
        _ => x.detach().cast(DType::I64)?,
    };
    let last = x.dim() - 1;
    let moved = if dim == last { keys } else { transpose(&keys, dim, last)? };
    let pos = k::sort_positions(&moved, descending, take)?;
    let idx = if dim == last { pos } else { transpose(&pos, dim, last)? };
    let values = indexed(x, dim, &idx, IndexMode::Gather, name)?;
    Ok((values, idx))
}

// ── scatter ─────────────────────────────────────────────────────────────

/// torch's argument checks for scatter, with its messages.
fn check_scatter(dst: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
    let index = as_index(index, "scatter")?;
    if src.dtype() != dst.dtype() {
        return Err(Error::Invalid("scatter(): Expected self.dtype to be equal to src.dtype".into()));
    }
    if index.dim() != dst.dim() || (src.dim() != dst.dim() && index.numel() > 0) {
        return Err(Error::Invalid("Index tensor must have the same number of dimensions as self tensor".into()));
    }
    let fits = (0..index.dim()).all(|d| index.shape()[d] <= src.shape()[d] && (d == dim || index.shape()[d] <= dst.shape()[d]));
    if !fits {
        return Err(Error::Invalid(format!(
            "Expected index {:?} to be no larger than self {:?} apart from dimension {dim} and to be no larger size than src {:?}",
            index.shape(),
            dst.shape(),
            src.shape()
        )));
    }
    let size = dst.shape()[dim];
    check_bounds(&index, size, false, |v| {
        Error::Invalid(format!("index {v} is out of bounds for dimension {dim} with size {size}"))
    })?;
    Ok(index.detach())
}

/// A contiguous copy of `x` in new storage.
fn copied(x: &Tensor) -> Result<Tensor> {
    let out = Tensor::empty_typed(x.device(), x.shape(), x.dtype())?;
    let strides = contiguous_strides(x.shape());
    k::copy_strided(x, x.offset(), &strides, &out, 0, &strides, x.shape())?;
    Ok(out)
}

/// The gradient reaching `src` through a scatter: the output gradient at
/// the scattered positions, zero where `src` extends past the index. (torch
/// returns `grad.gather(dim, index)` and so raises when `src` is larger than
/// the index; the zero-padded gradient is the correct one.)
fn scatter_src_grad(g: &Tensor, dim: usize, index: &Tensor, src_shape: &[usize]) -> Result<Tensor> {
    let picked = k::index_gather(g, dim, index, IndexMode::Gather)?;
    if picked.shape() == src_shape {
        return Ok(picked);
    }
    let out = Tensor::zeros(g.device(), src_shape)?;
    k::copy_strided(&picked, 0, &contiguous_strides(picked.shape()), &out, 0, &contiguous_strides(src_shape), picked.shape())?;
    Ok(out)
}

/// `torch.scatter(input, dim, index, src)`: a copy of `input` with `src`
/// written at the indexed positions. Differentiable in `input` (zero at the
/// overwritten positions) and `src` (the gradient gathered back).
pub fn scatter(input: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
    let index = check_scatter(input, dim, index, src)?;
    let mut out = copied(input)?;
    k::scatter(&out, dim, &index, src)?;
    let src_shape = src.shape().to_vec();
    record("ScatterBackward0", &[input, src], &mut out, &[], false, move |g, _, needs| {
        let grad_input = if needs[0] {
            let copy = copied(g)?;
            k::scatter(&copy, dim, &index, &Tensor::zeros(g.device(), index.shape())?)?;
            Some(copy)
        } else {
            None
        };
        let grad_src = if needs[1] { Some(scatter_src_grad(g, dim, &index, &src_shape)?) } else { None };
        Ok(vec![grad_input, grad_src])
    })?;
    Ok(out)
}

/// `dst.scatter_(dim, index, src)`, written into `dst`'s storage. Returns
/// the tensor the caller should now hold as `dst`: the same storage, and,
/// when `src` requires grad, a ScatterBackward0 node, so the gradient
/// reaches `src` as in torch. `dst` itself may not be part of a graph
/// (grilly's in-place rule; see docs/autograd.md).
pub fn scatter_(dst: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
    dst.check_inplace(&[])?;
    let index = check_scatter(dst, dim, index, src)?;
    k::scatter(dst, dim, &index, src)?;
    let mut out = dst.detach();
    let src_shape = src.shape().to_vec();
    record("ScatterBackward0", &[src], &mut out, &[], false, move |g, _, _| {
        Ok(vec![Some(scatter_src_grad(g, dim, &index, &src_shape)?)])
    })?;
    Ok(out)
}
