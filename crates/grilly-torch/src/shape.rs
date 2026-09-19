//! Shape ops. torch returns views for transpose, permute, expand and
//! indexing; grilly has no strided tensors yet, so these return copies
//! marked as view copies (in-place writes through them are refused rather
//! than silently missing the original). Gradients are torch's.

use grilly_core::generic::{self as k, contiguous_strides};
use grilly_core::{Error, Result, Tensor};

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
    let out = Tensor::empty(t.device(), shape)?;
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
    let out = Tensor::empty(x.device(), &shape)?;
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
    let mut out = Tensor::empty(x.device(), shape)?;
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
    let v = broadcast_raw(value, shape)?;
    k::write_strided(&v, v.offset(), &contiguous_strides(shape), dst, dst.offset() + offset, strides, shape)
}

fn check_cat(xs: &[&Tensor], dim: usize, what: &str) -> Result<()> {
    let first = xs.first().ok_or_else(|| Error::Invalid(format!("{what}(): expected a non-empty list of Tensors")))?;
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
    let mut out = Tensor::empty(xs[0].device(), &shape)?;
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
    let mut out = Tensor::empty(xs[0].device(), &shape)?;
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
        (1, 1) => sum(&binary(a, b, Binary::Mul)?, &[], false),
        (1, 2) => {
            let r = matmul(&a.reshape(&[1, a.shape()[0]])?, b)?;
            r.reshape(&[b.shape()[1]])
        }
        (_, 1) => {
            let r = matmul(a, &b.reshape(&[b.shape()[0], 1])?)?;
            r.reshape(&a.shape()[..a.dim() - 1])
        }
        (_, 2) => crate::linear(a, &transpose(b, 0, 1)?, None),
        (ad, bd) if ad == bd && a.shape()[..ad - 2] == b.shape()[..bd - 2] => {
            let batch: usize = a.shape()[..ad - 2].iter().product();
            let (m, kk) = (a.shape()[ad - 2], a.shape()[ad - 1]);
            let n = b.shape()[bd - 1];
            let a3 = a.reshape(&[batch, m, kk])?;
            let b3 = b.reshape(&[batch, b.shape()[bd - 2], n])?;
            let parts = (0..batch).map(|i| matmul(&select(&a3, 0, i)?, &select(&b3, 0, i)?)).collect::<Result<Vec<_>>>()?;
            let refs: Vec<&Tensor> = parts.iter().collect();
            let mut shape = a.shape()[..ad - 2].to_vec();
            shape.extend([m, n]);
            stack(&refs, 0)?.reshape(&shape)
        }
        _ => Err(Error::NotImplemented(format!(
            "matmul with broadcast batch dimensions ({:?} @ {:?})",
            a.shape(),
            b.shape()
        ))),
    }
}
