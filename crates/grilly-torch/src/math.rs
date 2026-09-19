//! Differentiable elementwise math and reductions on grilly-core's generic
//! kernels, with torch's broadcasting and torch's derivative formulas.

use grilly_core::autograd::{self, Backward, SavedTensor};
use grilly_core::generic::{self as k, Binary, Reduce, Unary};
use grilly_core::{Error, Result, Tensor};

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
        Unary::Gelu => record("GeluBackward0", &[x], o, &[x], false, |g, s, _| {
            Ok(vec![Some(mul(g, &k::unary(&s[0], Unary::GeluGrad, 0.0, 0.0)?)?)])
        })?,
        Unary::ClampMask | Unary::GeluGrad => {} // internal helpers, not differentiable
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
    let mut t = x.clone();
    if x.dim() == 0 {
        return k::reduce_all(x, op); // a 0-d tensor reduces to itself
    }
    for &d in dims.iter().rev() {
        t = k::reduce(&t, d, op, true)?;
    }
    if keepdim {
        return Ok(t);
    }
    let shape: Vec<usize> = x.shape().iter().enumerate().filter(|(i, _)| !dims.contains(i)).map(|(_, &s)| s).collect();
    t.detach().reshape(&shape)
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
