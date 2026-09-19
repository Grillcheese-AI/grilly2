//! torch-compatible ops on grilly-core. Names, argument order and semantics
//! follow `torch` / `torch.nn.functional`.

pub mod kernels;

use grilly_core::{Arg, Error, Result, Tensor};

fn u32_dim(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n).map_err(|_| Error::Shape(format!("{what} = {n} exceeds u32")))
}

/// `torch.relu(input)`.
pub fn relu(input: &Tensor) -> Result<Tensor> {
    let out = Tensor::empty(input.device(), input.shape())?;
    let n = u32_dim(input.numel(), "numel")?;
    if n > 0 {
        kernels::RELU.launch(input.device(), &[Arg::Tensor(input), Arg::Tensor(&out)], [n, 1, 1], &[n])?;
    }
    Ok(out)
}

/// `torch.add(input, other)` for tensors of the same shape.
/// Broadcasting and `alpha` are not implemented yet.
pub fn add(input: &Tensor, other: &Tensor) -> Result<Tensor> {
    if input.shape() != other.shape() {
        return Err(Error::Shape(format!(
            "add: shapes {:?} and {:?} differ (broadcasting is not implemented yet)",
            input.shape(),
            other.shape()
        )));
    }
    let out = Tensor::empty(input.device(), input.shape())?;
    let n = u32_dim(input.numel(), "numel")?;
    if n > 0 {
        kernels::ADD.launch(
            input.device(),
            &[Arg::Tensor(input), Arg::Tensor(other), Arg::Tensor(&out)],
            [n, 1, 1],
            &[n],
        )?;
    }
    Ok(out)
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
    if *w_in != in_features {
        return Err(Error::Shape(format!(
            "linear: input has {in_features} features but weight expects {w_in} ({:?} vs {:?})",
            input.shape(),
            weight.shape()
        )));
    }
    if let Some(b) = bias {
        if b.shape() != [*out_features] {
            return Err(Error::Shape(format!(
                "linear: bias must be ({out_features},), got {:?}",
                b.shape()
            )));
        }
    }

    let mut out_shape = input.shape().to_vec();
    *out_shape.last_mut().unwrap() = *out_features;
    let out = Tensor::empty(input.device(), &out_shape)?;
    let rows: usize = input.shape()[..input.dim() - 1].iter().product();
    if rows == 0 || *out_features == 0 {
        return Ok(out);
    }
    if in_features == 0 {
        // Empty reduction: torch returns the bias, or zeros.
        return match bias {
            Some(b) => {
                let host = b.to_vec()?;
                Tensor::from_slice(input.device(), &host.repeat(rows), &out_shape)
            }
            None => Tensor::zeros(input.device(), &out_shape),
        };
    }

    let (rows, k, n) = (u32_dim(rows, "rows")?, u32_dim(in_features, "in_features")?, u32_dim(*out_features, "out_features")?);
    // With no bias the shader never reads binding 2; bind the weight there.
    let bias_arg = Arg::Tensor(bias.unwrap_or(weight));
    kernels::LINEAR.launch(
        input.device(),
        &[Arg::Tensor(input), Arg::Tensor(weight), bias_arg, Arg::Tensor(&out)],
        [n, rows, 1],
        &[rows, k, n, bias.is_some() as u32],
    )?;
    Ok(out)
}
