//! The few ops the core itself needs (the engine sums gradients). Their
//! kernels are declared here, once, like every other kernel.

use crate::autograd::{self, Backward};
use crate::error::{Error, Result};
use crate::kernel::{Arg, Kernel};
use crate::tensor::Tensor;

/// `out[i] = a[i] + b[i]`. Bindings: a, b, out. Push: {n}.
pub const ADD: Kernel = Kernel { shader: "fnn-residual", bindings: 3, push_words: 1, local_size: [256, 1, 1] };

/// `a[i] += b[i]`. Bindings: a (read-write), b. Push: {n}.
pub const ADD_INPLACE: Kernel = Kernel { shader: "elementwise-add", bindings: 2, push_words: 1, local_size: [256, 1, 1] };

pub const ALL: &[Kernel] = &[ADD, ADD_INPLACE];

fn numel_u32(t: &Tensor) -> Result<u32> {
    u32::try_from(t.numel()).map_err(|_| Error::Shape(format!("{} elements exceed u32", t.numel())))
}

fn same_shape(op: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::Shape(format!(
            "{op}: shapes {:?} and {:?} differ (broadcasting is not implemented yet)",
            a.shape(),
            b.shape()
        )));
    }
    Ok(())
}

struct AddBackward;

impl Backward for AddBackward {
    fn apply(&mut self, grads: &[Option<Tensor>], needs: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let g = grads[0].as_ref().expect("single output");
        Ok(needs.iter().map(|&n| n.then(|| g.clone())).collect())
    }
}

/// `torch.add(input, other)` for tensors of the same shape.
pub fn add(input: &Tensor, other: &Tensor) -> Result<Tensor> {
    same_shape("add", input, other)?;
    let mut out = Tensor::empty(input.device(), input.shape())?;
    let n = numel_u32(input)?;
    if n > 0 {
        ADD.launch(input.device(), &[Arg::Tensor(input), Arg::Tensor(other), Arg::Tensor(&out)], [n, 1, 1], &[n])?;
    }
    autograd::record("AddBackward0", &[input, other], &mut [&mut out], |_| Ok(AddBackward))?;
    Ok(out)
}

/// `dst += src` in place, untracked. Used to accumulate `.grad` and by
/// `Tensor::add_`.
pub(crate) fn add_into(dst: &Tensor, src: &Tensor) -> Result<()> {
    same_shape("add_", dst, src)?;
    let n = numel_u32(dst)?;
    if n > 0 {
        ADD_INPLACE.launch(dst.device(), &[Arg::Tensor(dst), Arg::Tensor(src)], [n, 1, 1], &[n])?;
    }
    dst.bump_version();
    Ok(())
}
