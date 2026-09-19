//! Differentiable elementwise math and reductions on grilly-core's generic
//! kernels, with torch's broadcasting and torch's derivative formulas.

use grilly_core::autograd::{self, Backward, SavedTensor};
use grilly_core::generic::{self as k, Binary, IntBinary, Reduce, Scan, Unary};
use grilly_core::{DType, Error, Result, Tensor};

use crate::shape;

type GradFn = Box<dyn FnMut(&Tensor, &[Tensor], &[bool]) -> Result<Vec<Option<Tensor>>> + Send>;

/// A backward formula: tensors saved at forward time and a function of
/// (output gradient, saved tensors, which inputs need a gradient).
pub(crate) struct Formula {
    saved: Vec<SavedTensor>,
    f: GradFn,
}

impl Backward for Formula {
    fn apply(&mut self, grads: &[Option<Tensor>], needs: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let g = grads[0].as_ref().expect("single-output formula");
        let saved = self.saved.iter_mut().map(SavedTensor::get).collect::<Result<Vec<_>>>()?;
        (self.f)(g, &saved, needs)
    }

    fn release(&mut self) {
        self.saved.iter_mut().for_each(SavedTensor::release);
    }
}

/// Record a single-output op. `save` lists inputs to keep; with `save_out`
/// the output is kept after them.
pub(crate) fn record(
    name: &'static str,
    inputs: &[&Tensor],
    out: &mut Tensor,
    save: &[&Tensor],
    save_out: bool,
    f: impl FnMut(&Tensor, &[Tensor], &[bool]) -> Result<Vec<Option<Tensor>>> + Send + 'static,
) -> Result<()> {
    autograd::record(name, inputs, &mut [out], |outs| {
        let mut saved = save.iter().map(|t| SavedTensor::save(t)).collect::<Result<Vec<_>>>()?;
        if save_out {
            saved.push(SavedTensor::save(outs[0])?);
        }
        Ok(Formula { saved, f: Box::new(f) })
    })
}

/// `Some(f())` when input `i` needs a gradient.
fn need(needs: &[bool], i: usize, f: impl FnOnce() -> Result<Tensor>) -> Result<Option<Tensor>> {
    if needs[i] { f().map(Some) } else { Ok(None) }
}

// Untracked helpers for formulas (backward runs with grad mode off anyway).
fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    k::binary(a, b, Binary::Mul)
}
fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    k::binary(a, b, Binary::Div)
}
fn scale(a: &Tensor, c: f32) -> Result<Tensor> {
    k::unary(a, Unary::MulScalar, c, 0.0)
}
fn neg(a: &Tensor) -> Result<Tensor> {
    k::unary(a, Unary::Neg, 0.0, 0.0)
}

// ── binary ───────────────────────────────────────────────────────────────

/// `op(a, b)` with torch broadcasting, differentiable. Comparisons return
/// 1.0 / 0.0 masks with no gradient (torch returns bool tensors).
pub fn binary(a: &Tensor, b: &Tensor, op: Binary) -> Result<Tensor> {
    let mut out = k::binary(a, b, op)?;
    let (sa, sb) = (a.shape().to_vec(), b.shape().to_vec());
    let down = move |t: Tensor, s: &[usize]| k::sum_to_shape(&t, s);
    match op {
        Binary::Add => record("AddBackward0", &[a, b], &mut out, &[], false, move |g, _, n| {
            Ok(vec![need(n, 0, || down(g.clone(), &sa))?, need(n, 1, || down(g.clone(), &sb))?])
        })?,
        Binary::Sub => record("SubBackward0", &[a, b], &mut out, &[], false, move |g, _, n| {
            Ok(vec![need(n, 0, || down(g.clone(), &sa))?, need(n, 1, || down(neg(g)?, &sb))?])
        })?,
        Binary::Mul => record("MulBackward0", &[a, b], &mut out, &[a, b], false, move |g, s, n| {
            Ok(vec![need(n, 0, || down(mul(g, &s[1])?, &sa))?, need(n, 1, || down(mul(g, &s[0])?, &sb))?])
        })?,
        Binary::Div => record("DivBackward0", &[a, b], &mut out, &[a, b], false, move |g, s, n| {
            let ga = div(g, &s[1])?;
            let gb = need(n, 1, || down(neg(&mul(&ga, &div(&s[0], &s[1])?)?)?, &sb))?;
            Ok(vec![need(n, 0, || down(ga.clone(), &sa))?, gb])
        })?,
        Binary::Pow => record("PowBackward1", &[a, b], &mut out, &[a, b], true, move |g, s, n| {
            let (base, expo, result) = (&s[0], &s[1], &s[2]);
            let ga = need(n, 0, || {
                let lower = k::binary(base, &k::unary(expo, Unary::AddScalar, -1.0, 0.0)?, Binary::Pow)?;
                down(mul(g, &mul(expo, &lower)?)?, &sa)
            })?;
            // torch zeroes the exponent's gradient where the base is 0.
            let gb = need(n, 1, || {
                let safe = k::binary(base, &k::binary(base, &Tensor::zeros(base.device(), &[])?, Binary::Eq)?, Binary::Add)?;
                down(mul(g, &mul(result, &k::unary(&safe, Unary::Log, 0.0, 0.0)?)?)?, &sb)
            })?;
            Ok(vec![ga, gb])
        })?,
        Binary::Maximum | Binary::Minimum => {
            let (name, win) = if op == Binary::Maximum { ("MaximumBackward0", Binary::Gt) } else { ("MinimumBackward0", Binary::Lt) };
            record(name, &[a, b], &mut out, &[a, b], false, move |g, s, n| {
                // The larger (smaller) input gets the gradient; ties split it, as in torch.
                let tie = scale(&k::binary(&s[0], &s[1], Binary::Eq)?, 0.5)?;
                let wa = k::binary(&k::binary(&s[0], &s[1], win)?, &tie, Binary::Add)?;
                let wb = k::unary(&wa, Unary::RsubScalar, 1.0, 0.0)?;
                Ok(vec![need(n, 0, || down(mul(g, &wa)?, &sa))?, need(n, 1, || down(mul(g, &wb)?, &sb))?])
            })?
        }
        Binary::Eq | Binary::Ne | Binary::Gt | Binary::Ge | Binary::Lt | Binary::Le => {}
    }
    Ok(out)
}

// ── unary and scalar ─────────────────────────────────────────────────────

/// `f(x; a, b)`, differentiable (see [`Unary`] for what `a`, `b` mean).
pub fn unary(x: &Tensor, op: Unary, a: f32, b: f32) -> Result<Tensor> {
    let mut out = k::unary(x, op, a, b)?;
    let o = &mut out;
    match op {
        Unary::Neg => record("NegBackward0", &[x], o, &[], false, |g, _, _| Ok(vec![Some(neg(g)?)]))?,
        Unary::Abs => record("AbsBackward0", &[x], o, &[x], false, |g, s, _| {
            Ok(vec![Some(mul(g, &k::unary(&s[0], Unary::Sign, 0.0, 0.0)?)?)])
        })?,
        Unary::Sign => record("SignBackward0", &[x], o, &[], false, |g, _, _| {
            Ok(vec![Some(Tensor::zeros(g.device(), g.shape())?)])
        })?,
        Unary::Exp => record("ExpBackward0", &[x], o, &[], true, |g, s, _| Ok(vec![Some(mul(g, &s[0])?)]))?,
        Unary::Log => record("LogBackward0", &[x], o, &[x], false, |g, s, _| Ok(vec![Some(div(g, &s[0])?)]))?,
        Unary::Sqrt => record("SqrtBackward0", &[x], o, &[], true, |g, s, _| {
            Ok(vec![Some(div(g, &scale(&s[0], 2.0)?)?)])
        })?,
        Unary::Rsqrt => record("RsqrtBackward0", &[x], o, &[], true, |g, s, _| {
            let cube = k::unary(&s[0], Unary::PowScalar, 3.0, 0.0)?;
            Ok(vec![Some(scale(&mul(g, &cube)?, -0.5)?)])
        })?,
        Unary::Tanh => record("TanhBackward0", &[x], o, &[], true, |g, s, _| {
            let d = k::unary(&k::unary(&s[0], Unary::PowScalar, 2.0, 0.0)?, Unary::RsubScalar, 1.0, 0.0)?;
            Ok(vec![Some(mul(g, &d)?)])
        })?,
        Unary::Sigmoid => record("SigmoidBackward0", &[x], o, &[], true, |g, s, _| {
            let d = mul(&s[0], &k::unary(&s[0], Unary::RsubScalar, 1.0, 0.0)?)?;
            Ok(vec![Some(mul(g, &d)?)])
        })?,
        Unary::AddScalar => record("AddBackward1", &[x], o, &[], false, |g, _, _| Ok(vec![Some(g.clone())]))?,
        Unary::MulScalar => record("MulBackward1", &[x], o, &[], false, move |g, _, _| Ok(vec![Some(scale(g, a)?)]))?,
        Unary::RsubScalar => record("RsubBackward1", &[x], o, &[], false, |g, _, _| Ok(vec![Some(neg(g)?)]))?,
        Unary::RdivScalar => record("DivBackward0", &[x], o, &[x], true, |g, s, _| {
            Ok(vec![Some(neg(&mul(g, &div(&s[1], &s[0])?)?)?)])
        })?,
        Unary::PowScalar => record("PowBackward0", &[x], o, &[x], false, move |g, s, _| {
            if a == 0.0 {
                return Ok(vec![Some(Tensor::zeros(g.device(), g.shape())?)]);
            }
            let lower = k::unary(&s[0], Unary::PowScalar, a - 1.0, 0.0)?;
            Ok(vec![Some(scale(&mul(g, &lower)?, a)?)])
        })?,
        Unary::Clamp => record("ClampBackward1", &[x], o, &[x], false, move |g, s, _| {
            Ok(vec![Some(mul(g, &k::unary(&s[0], Unary::ClampMask, a, b)?)?)])
        })?,
        Unary::Sin => record("SinBackward0", &[x], o, &[x], false, |g, s, _| {
            Ok(vec![Some(mul(g, &k::unary(&s[0], Unary::Cos, 0.0, 0.0)?)?)])
        })?,
        Unary::Cos => record("CosBackward0", &[x], o, &[x], false, |g, s, _| {
            Ok(vec![Some(neg(&mul(g, &k::unary(&s[0], Unary::Sin, 0.0, 0.0)?)?)?)])
        })?,
        Unary::Gelu => record("GeluBackward0", &[x], o, &[x], false, |g, s, _| {
            Ok(vec![Some(mul(g, &k::unary(&s[0], Unary::GeluGrad, 0.0, 0.0)?)?)])
        })?,
        // internal helpers and predicates: not differentiable
        Unary::ClampMask | Unary::GeluGrad | Unary::IsFinite | Unary::IsInf | Unary::IsNan => {}
    }
    Ok(out)
}

// ── reductions ───────────────────────────────────────────────────────────

fn check_dims(x: &Tensor, dims: &[usize]) -> Result<Vec<usize>> {
    let mut d: Vec<usize> = if dims.is_empty() { (0..x.dim()).collect() } else { dims.to_vec() };
    d.sort_unstable();
    d.dedup();
    if !dims.is_empty() && d.len() != dims.len() {
        return Err(Error::Invalid("dim appears multiple times in the list of dims".into()));
    }
    if let Some(&bad) = d.iter().find(|&&v| v >= x.dim().max(1)) {
        return Err(Error::Shape(format!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {bad})",
            -(x.dim() as isize),
            x.dim() as isize - 1
        )));
    }
    Ok(d)
}

/// Shape with every dim in `dims` set to 1.
fn kept_shape(shape: &[usize], dims: &[usize]) -> Vec<usize> {
    shape.iter().enumerate().map(|(i, &s)| if dims.contains(&i) { 1 } else { s }).collect()
}

fn reduce_raw(x: &Tensor, dims: &[usize], op: Reduce, keepdim: bool) -> Result<Tensor> {
    if x.dim() == 0 {
        return k::reduce_all(x, op); // a 0-d tensor reduces to itself
    }
    let shape = x.shape();
    let mut final_shape: Vec<usize> = Vec::with_capacity(shape.len());
    for (i, &s) in shape.iter().enumerate() {
        if !dims.contains(&i) {
            final_shape.push(s);
        } else if keepdim {
            final_shape.push(1);
        }
    }
    // adjacent reduced dims (layer_norm's trailing dims, a full sum) merge
    // into one, so they take one pass
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &d in dims {
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == d => *end = d,
            _ => runs.push((d, d)),
        }
    }
    let mut t = x.detach();
    for &(first, last) in runs.iter().rev() {
        let mut merged: Vec<usize> = t.shape()[..first].to_vec();
        merged.push(t.shape()[first..=last].iter().product());
        merged.extend_from_slice(&t.shape()[last + 1..]);
        t = k::reduce(&t.reshape(&merged)?, first, op, true)?;
        let mut kept: Vec<usize> = t.shape()[..first].to_vec();
        kept.extend(std::iter::repeat_n(1, last - first + 1));
        kept.extend_from_slice(&t.shape()[first + 1..]);
        t = t.reshape(&kept)?;
    }
    t.reshape(&final_shape)
}

/// `torch.sum(x, dim=dims, keepdim)`; empty `dims` sums everything.
pub fn sum(x: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor> {
    let d = check_dims(x, dims)?;
    let mut out = reduce_raw(x, &d, Reduce::Sum, keepdim)?;
    let (in_shape, keep) = (x.shape().to_vec(), kept_shape(x.shape(), &d));
    let name = if dims.is_empty() { "SumBackward0" } else { "SumBackward1" };
    record(name, &[x], &mut out, &[], false, move |g, _, _| {
        Ok(vec![Some(shape::broadcast_raw(&g.detach().reshape(&keep)?, &in_shape)?)])
    })?;
    Ok(out)
}

/// `torch.mean(x, dim=dims, keepdim)`; empty `dims` averages everything.
pub fn mean(x: &Tensor, dims: &[usize], keepdim: bool) -> Result<Tensor> {
    let d = check_dims(x, dims)?;
    let mut out = reduce_raw(x, &d, Reduce::Mean, keepdim)?;
    let count: usize = d.iter().map(|&i| x.shape().get(i).copied().unwrap_or(1)).product();
    let (in_shape, keep) = (x.shape().to_vec(), kept_shape(x.shape(), &d));
    let name = if dims.is_empty() { "MeanBackward0" } else { "MeanBackward1" };
    record(name, &[x], &mut out, &[], false, move |g, _, _| {
        let spread = shape::broadcast_raw(&g.detach().reshape(&keep)?, &in_shape)?;
        Ok(vec![Some(scale(&spread, 1.0 / count as f32)?)])
    })?;
    Ok(out)
}

/// `torch.amax` / `torch.amin`: the gradient is shared evenly among ties.
pub fn amax(x: &Tensor, dims: &[usize], keepdim: bool, max: bool) -> Result<Tensor> {
    let d = check_dims(x, dims)?;
    let op = if max { Reduce::Max } else { Reduce::Min };
    let mut out = reduce_raw(x, &d, op, keepdim)?;
    let (in_shape, keep, dd) = (x.shape().to_vec(), kept_shape(x.shape(), &d), d.clone());
    let name = if max { "AmaxBackward0" } else { "AminBackward0" };
    record(name, &[x], &mut out, &[x], true, move |g, s, _| {
        let best = shape::broadcast_raw(&s[1].detach().reshape(&keep)?, &in_shape)?;
        let hit = k::binary(&s[0], &best, Binary::Eq)?;
        let count = shape::broadcast_raw(&reduce_raw(&hit, &dd, Reduce::Sum, true)?, &in_shape)?;
        let spread = shape::broadcast_raw(&g.detach().reshape(&keep)?, &in_shape)?;
        Ok(vec![Some(div(&mul(&spread, &hit)?, &count)?)])
    })?;
    Ok(out)
}

// ── composites: differentiable through the ops they are made of ──────────

/// `torch.softmax(x, dim)`.
pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let m = reduce_raw(&x.detach(), &check_dims(x, &[dim])?, Reduce::Max, true)?;
    let e = unary(&binary(x, &m, Binary::Sub)?, Unary::Exp, 0.0, 0.0)?;
    binary(&e, &sum(&e, &[dim], true)?, Binary::Div)
}

/// `torch.log_softmax(x, dim)`.
pub fn log_softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    let m = reduce_raw(&x.detach(), &check_dims(x, &[dim])?, Reduce::Max, true)?;
    let shifted = binary(x, &m, Binary::Sub)?;
    let lse = unary(&sum(&unary(&shifted, Unary::Exp, 0.0, 0.0)?, &[dim], true)?, Unary::Log, 0.0, 0.0)?;
    binary(&shifted, &lse, Binary::Sub)
}

/// `F.normalize(x, p=2, dim, eps)`: `x / max(||x||_2, eps)` along `dim`.
pub fn normalize(x: &Tensor, dim: usize, eps: f32) -> Result<Tensor> {
    let sq = unary(x, Unary::PowScalar, 2.0, 0.0)?;
    let norm = unary(&sum(&sq, &[dim], true)?, Unary::Sqrt, 0.0, 0.0)?;
    let denom = unary(&norm, Unary::Clamp, eps, f32::INFINITY)?;
    binary(x, &denom, Binary::Div)
}

/// `F.layer_norm` over the trailing `normalized` dims (biased variance).
pub fn layer_norm(x: &Tensor, normalized: usize, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    if normalized == 0 || normalized > x.dim() {
        return Err(Error::Shape(format!("layer_norm over {normalized} dims of a {}-d input", x.dim())));
    }
    let dims: Vec<usize> = (x.dim() - normalized..x.dim()).collect();
    let centred = binary(x, &mean(x, &dims, true)?, Binary::Sub)?;
    let var = mean(&unary(&centred, Unary::PowScalar, 2.0, 0.0)?, &dims, true)?;
    let inv = unary(&unary(&var, Unary::AddScalar, eps, 0.0)?, Unary::Rsqrt, 0.0, 0.0)?;
    let mut y = binary(&centred, &inv, Binary::Mul)?;
    if let Some(w) = weight {
        y = binary(&y, w, Binary::Mul)?;
    }
    if let Some(b) = bias {
        y = binary(&y, b, Binary::Add)?;
    }
    Ok(y)
}

/// `F.silu(x) = x * sigmoid(x)`.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    binary(x, &unary(x, Unary::Sigmoid, 0.0, 0.0)?, Binary::Mul)
}

// ── integers, booleans, comparisons, selection ──────────────────────────

/// `isfinite` / `isinf` / `isnan`: a bool tensor. Integer and bool tensors
/// are always finite, as in torch.
pub fn predicate(x: &Tensor, op: Unary) -> Result<Tensor> {
    if !x.dtype().is_floating() {
        let value = if op == Unary::IsFinite { 1.0 } else { 0.0 };
        return Tensor::full_typed(x.device(), x.shape(), value, DType::Bool);
    }
    k::unary(x, op, 0.0, 0.0)?.cast(DType::Bool)
}

fn int_op(op: Binary) -> Result<IntBinary> {
    Ok(match op {
        Binary::Add => IntBinary::Add,
        Binary::Sub => IntBinary::Sub,
        Binary::Mul => IntBinary::Mul,
        Binary::Maximum => IntBinary::Maximum,
        Binary::Minimum => IntBinary::Minimum,
        Binary::Eq => IntBinary::Eq,
        Binary::Ne => IntBinary::Ne,
        Binary::Gt => IntBinary::Gt,
        Binary::Ge => IntBinary::Ge,
        Binary::Lt => IntBinary::Lt,
        Binary::Le => IntBinary::Le,
        Binary::Div | Binary::Pow => {
            return Err(Error::NotImplemented(format!("{op:?} on integer tensors")))
        }
    })
}

/// A comparison of two tensors of the same dtype (callers promote first):
/// a bool tensor, not differentiable.
pub fn compare(a: &Tensor, b: &Tensor, op: Binary) -> Result<Tensor> {
    if a.dtype().is_floating() {
        return k::binary(a, b, op)?.cast(DType::Bool);
    }
    k::binary_int(&a.cast(DType::I64)?, &b.cast(DType::I64)?, int_op(op)?, true)
}

/// Integer arithmetic (+, -, *, maximum, minimum) or bitwise/logical ops on
/// integer or bool tensors, computed exactly in int64 and returned in
/// `out` (the promoted dtype chosen by the caller).
pub fn integer(a: &Tensor, b: &Tensor, op: IntBinary, out: DType) -> Result<Tensor> {
    let r = k::binary_int(&a.cast(DType::I64)?, &b.cast(DType::I64)?, op, false)?;
    r.cast(out)
}

/// Integer sum / max / min over dims (all when empty), computed in int64;
/// torch's `sum` of integers or bools is int64.
pub fn integer_reduce(x: &Tensor, dims: &[usize], op: Reduce, keepdim: bool) -> Result<Tensor> {
    let mut d: Vec<usize> = if dims.is_empty() { (0..x.dim()).collect() } else { dims.to_vec() };
    d.sort_unstable();
    d.dedup();
    let mut t = x.cast(DType::I64)?;
    if x.dim() == 0 {
        return Ok(t);
    }
    for &dim in d.iter().rev() {
        t = k::reduce_int(&t, dim, op, true)?;
    }
    if keepdim {
        return Ok(t);
    }
    let shape: Vec<usize> = x.shape().iter().enumerate().filter(|(i, _)| !d.contains(i)).map(|(_, &s)| s).collect();
    t.reshape(&shape)
}

/// `torch.where(cond, a, b)` with broadcasting; `a` and `b` share a dtype
/// (callers promote first). The gradient goes to the branch that was taken.
pub fn where_(cond: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let mut out = k::where_select(cond, a, b)?;
    if !a.dtype().is_floating() {
        return Ok(out);
    }
    let (sa, sb) = (a.shape().to_vec(), b.shape().to_vec());
    record("WhereBackward0", &[a, b], &mut out, &[cond], false, move |g, s, n| {
        let zero = Tensor::zeros(g.device(), &[])?;
        Ok(vec![
            need(n, 0, || k::sum_to_shape(&k::where_select(&s[0], g, &zero)?, &sa))?,
            need(n, 1, || k::sum_to_shape(&k::where_select(&s[0], &zero, g)?, &sb))?,
        ])
    })?;
    Ok(out)
}

// ── scans ────────────────────────────────────────────────────────────────

/// `torch.cumsum(x, dim)`: float32 stays float32 (compensated summation);
/// integers and bools sum exactly as int64, as in torch. The gradient is
/// the gradient's cumsum taken from the end.
pub fn cumsum(x: &Tensor, dim: usize) -> Result<Tensor> {
    let src = match x.dtype() {
        DType::F32 | DType::I64 => x.detach(),
        _ => x.cast(DType::I64)?,
    };
    let mut out = k::scan(&src, dim, Scan::Cumsum, false)?;
    record("CumsumBackward0", &[x], &mut out, &[], false, move |g, _, _| {
        Ok(vec![Some(k::scan(g, dim, Scan::Cumsum, true)?)])
    })?;
    Ok(out)
}

/// `torch.logcumsumexp(x, dim)`: `log(sum_{j<=i} exp(x_j))`, stably, with
/// torch's backward (FunctionsManual.cpp `logcumsumexp_backward`): the
/// gradient's positive and negative parts go through reversed
/// log-cumsum-exps, so nothing overflows.
pub fn logcumsumexp(x: &Tensor, dim: usize) -> Result<Tensor> {
    if !x.dtype().is_floating() {
        return Err(Error::Invalid(format!("\"logcumsumexp_out_cpu\" not implemented for '{}'", type_label(x.dtype()))));
    }
    let mut out = k::scan(x, dim, Scan::LogCumsumExp, false)?;
    record("LogcumsumexpBackward0", &[x], &mut out, &[x], true, move |g, s, _| {
        let (x, y) = (&s[0], &s[1]);
        if g.numel() <= 1 || g.shape()[dim] == 1 {
            return Ok(vec![Some(g.clone())]);
        }
        let dev = g.device();
        let (zero, lowest) = (Tensor::zeros(dev, &[])?, Tensor::full(dev, &[], f32::MIN)?);
        let log_abs = k::unary(&k::unary(g, Unary::Abs, 0.0, 0.0)?, Unary::Log, 0.0, 0.0)?;
        let part = |op: Binary| -> Result<Tensor> {
            let mask = k::binary(g, &zero, op)?.cast(DType::Bool)?;
            let log_part = k::where_select(&mask, &log_abs, &lowest)?;
            let r = k::scan(&k::binary(&log_part, y, Binary::Sub)?, dim, Scan::LogCumsumExp, true)?;
            k::unary(&k::binary(&r, x, Binary::Add)?, Unary::Exp, 0.0, 0.0)
        };
        Ok(vec![Some(k::binary(&part(Binary::Gt)?, &part(Binary::Lt)?, Binary::Sub)?)])
    })?;
    Ok(out)
}

/// torch's capitalized scalar-type name, as in its "not implemented for" errors.
fn type_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "Float",
        DType::I64 => "Long",
        DType::I32 => "Int",
        DType::Bool => "Bool",
    }
}
