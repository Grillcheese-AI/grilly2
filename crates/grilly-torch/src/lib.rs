//! torch-compatible ops on grilly-core. Names, argument order and semantics
//! follow `torch` / `torch.nn.functional`. Every op records its backward
//! formula for autograd.

pub mod kernels;

use grilly_core::autograd::{self, Backward, SavedTensor};
use grilly_core::{Arg, Error, Result, Tensor};

pub use grilly_core::add;

fn u32_dim(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Shape(format!("{what} = {n} exceeds u32")))
}

// ── relu ─────────────────────────────────────────────────────────────────

struct ReluBackward {
    /// relu's output: `out > 0` exactly where `x > 0`.
    result: SavedTensor,
}

impl Backward for ReluBackward {
    fn apply(&mut self, grads: &[Option<Tensor>], _: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let g = grads[0].as_ref().expect("single output");
        let result = &self.result.get()?;
        let grad_in = Tensor::empty(g.device(), g.shape())?;
        let n = u32_dim(g.numel(), "numel")?;
        if n > 0 {
            kernels::RELU_BACKWARD.launch(
                g.device(),
                &[Arg::Tensor(g), Arg::Tensor(result), Arg::Tensor(&grad_in)],
                [n, 1, 1],
                &[n],
            )?;
        }
        Ok(vec![Some(grad_in)])
    }

    fn release(&mut self) {
        self.result.release();
    }
}

/// `torch.relu(input)`.
pub fn relu(input: &Tensor) -> Result<Tensor> {
    let mut out = Tensor::empty(input.device(), input.shape())?;
    let n = u32_dim(input.numel(), "numel")?;
    if n > 0 {
        kernels::RELU.launch(input.device(), &[Arg::Tensor(input), Arg::Tensor(&out)], [n, 1, 1], &[n])?;
    }
    autograd::record("ReluBackward0", &[input], &mut [&mut out], |outs| {
        Ok(ReluBackward { result: SavedTensor::save(outs[0])? })
    })?;
    Ok(out)
}

// ── linear ───────────────────────────────────────────────────────────────

/// grilly's own node: torch decomposes linear (AddmmBackward0, MmBackward0,
/// ViewBackward0 depending on shapes), grilly runs it as one op.
struct LinearBackward {
    input: SavedTensor,
    weight: SavedTensor,
}

impl LinearBackward {
    fn pass(x: &Tensor, w: &Tensor, pass: u32, g: &Tensor, out: &Tensor, threads: [u32; 2]) -> Result<()> {
        let (in_f, out_f) = (w.shape()[1], w.shape()[0]);
        // From the leading dims, not numel / in_f: that is 0 when in_f is 0.
        let rows: usize = x.shape()[..x.dim() - 1].iter().product();
        // Each pass writes one output; the other two output slots are bound
        // to that same tensor and never touched.
        kernels::LINEAR_BACKWARD.launch(
            g.device(),
            &[Arg::Tensor(g), Arg::Tensor(x), Arg::Tensor(w), Arg::Tensor(out), Arg::Tensor(out), Arg::Tensor(out)],
            [threads[0], threads[1], 1],
            &[u32_dim(rows, "rows")?, u32_dim(in_f, "in_features")?, u32_dim(out_f, "out_features")?, pass],
        )
    }
}

impl Backward for LinearBackward {
    fn apply(&mut self, grads: &[Option<Tensor>], needs: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let g = grads[0].as_ref().expect("single output");
        let (x, w) = (self.input.get()?, self.weight.get()?);
        let dev = g.device();
        let (out_f, in_f) = (w.shape()[0], w.shape()[1]);
        let rows: usize = x.shape()[..x.dim() - 1].iter().product();
        // An empty reduction contributes zero, as in torch.
        let empty_sum = |shape: &[usize], reduced: usize| -> Result<Option<Tensor>> {
            if reduced == 0 { Tensor::zeros(dev, shape).map(Some) } else { Ok(None) }
        };

        let mut result = vec![None; needs.len()];
        if needs[0] {
            result[0] = Some(match empty_sum(x.shape(), out_f)? {
                Some(z) => z,
                None => {
                    let gx = Tensor::empty(dev, x.shape())?;
                    if gx.numel() > 0 {
                        Self::pass(&x, &w, 0, g, &gx, [u32_dim(in_f, "in")?, u32_dim(rows, "rows")?])?;
                    }
                    gx
                }
            });
        }
        if needs[1] {
            result[1] = Some(match empty_sum(w.shape(), rows)? {
                Some(z) => z,
                None => {
                    let gw = Tensor::empty(dev, w.shape())?;
                    if gw.numel() > 0 {
                        Self::pass(&x, &w, 1, g, &gw, [u32_dim(in_f, "in")?, u32_dim(out_f, "out")?])?;
                    }
                    gw
                }
            });
        }
        if needs.len() > 2 && needs[2] {
            result[2] = Some(match empty_sum(&[out_f], rows)? {
                Some(z) => z,
                None => {
                    let gb = Tensor::empty(dev, &[out_f])?;
                    if out_f > 0 {
                        Self::pass(&x, &w, 2, g, &gb, [u32_dim(out_f, "out")?, 1])?;
                    }
                    gb
                }
            });
        }
        Ok(result)
    }

    fn release(&mut self) {
        self.input.release();
        self.weight.release();
    }
}

/// `torch.nn.functional.linear(input, weight, bias=None)`:
/// `input @ weight.T + bias` over the last dimension of `input`.
///
/// `input` is `(*, in_features)`, `weight` is `(out_features, in_features)`,
/// `bias` is `(out_features,)`.
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (Some(&in_features), [out_features, w_in]) = (input.shape().last(), weight.shape()) else {
        return Err(Error::Shape(format!(
            "linear: expected input (*, in_features) and weight (out_features, in_features), got {:?} and {:?}",
            input.shape(),
            weight.shape()
        )));
    };
    let out_features = *out_features;
    if *w_in != in_features {
        return Err(Error::Shape(format!(
            "linear: input has {in_features} features but weight expects {w_in} ({:?} vs {:?})",
            input.shape(),
            weight.shape()
        )));
    }
    if let Some(b) = bias {
        if b.shape() != [out_features] {
            return Err(Error::Shape(format!(
                "linear: bias must be ({out_features},), got {:?}",
                b.shape()
            )));
        }
    }

    let mut out_shape = input.shape().to_vec();
    *out_shape.last_mut().unwrap() = out_features;
    let rows: usize = input.shape()[..input.dim() - 1].iter().product();

    let mut out = {
        if rows == 0 || out_features == 0 {
            Tensor::empty(input.device(), &out_shape)?
        } else if in_features == 0 {
            // Empty reduction: torch returns the bias, or zeros.
            match bias {
                Some(b) => {
                    let host = b.to_vec()?;
                    Tensor::from_slice(input.device(), &host.repeat(rows), &out_shape)?
                }
                None => Tensor::zeros(input.device(), &out_shape)?,
            }
        } else {
            let out = Tensor::empty(input.device(), &out_shape)?;
            let (rows, k, n) = (u32_dim(rows, "rows")?, u32_dim(in_features, "in_features")?, u32_dim(out_features, "out_features")?);
            // With no bias the shader never reads binding 2; bind the weight there.
            let bias_arg = Arg::Tensor(bias.unwrap_or(weight));
            kernels::LINEAR.launch(
                input.device(),
                &[Arg::Tensor(input), Arg::Tensor(weight), bias_arg, Arg::Tensor(&out)],
                [n, rows, 1],
                &[rows, k, n, bias.is_some() as u32],
            )?;
            out
        }
    };

    let mut inputs = vec![input, weight];
    inputs.extend(bias);
    autograd::record("LinearBackward0", &inputs, &mut [&mut out], |_| {
        Ok(LinearBackward { input: SavedTensor::save(input)?, weight: SavedTensor::save(weight)? })
    })?;
    Ok(out)
}
