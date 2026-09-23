//! Sparse block-code VSA: `(..., k, l)` float32, k blocks of l.
//!
//! The algebra is IBM's NVSA (Hersche et al., 2023), and it is the binding
//! algebra CubbyLLM decided on. A discrete code is one-hot per block; a
//! continuous one is a distribution over each block's positions. Binding is
//! per-block circular convolution, which keeps that structure exactly: bind
//! two one-hot blocks and the result is one-hot at the sum of their
//! positions, mod l.
//!
//! Only binding needs a kernel of its own. Similarity, bundling and
//! everything else are the framework's ordinary ops over the same tensors,
//! which is the point of porting this onto grilly rather than beside it.
//!
//! **Tolerance.** For one-hot inputs `bind` is exact: one term of each sum
//! is 1.0 and the rest are 0.0, so the result is an exact one-hot whatever
//! order they are added in. That makes it bit-for-bit equal to v1's
//! `blockcode_bind`, which got there by finding the hot positions. For
//! continuous inputs it is a float32 sum of l products, and the reference
//! is the same convolution in float64.

use grilly_core::{Arg, Error, Kernel, Result, Tensor};

/// Bindings: a, b, out. Push: elements, k, l, unbind, the two batch
/// strides (0 broadcasts), and the two element offsets.
const BIND: Kernel = Kernel { shader: "g2-blockcode-bind", bindings: 3, push_words: 8, local_size: [256, 1, 1] };

pub const KERNELS: &[Kernel] = &[BIND];

/// `(vectors, k, l)` for a block-code tensor, which is `(..., k, l)` with
/// any number of leading batch dimensions — including none.
fn shape_of(t: &Tensor, what: &str) -> Result<(usize, usize, usize)> {
    let shape = t.shape();
    if shape.len() < 2 {
        return Err(Error::Shape(format!("{what}: a block code is (..., k, l), got {shape:?}")));
    }
    let (k, l) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    if k == 0 || l == 0 {
        return Err(Error::Shape(format!("{what}: a block code is (..., k, l) with k, l > 0, got {shape:?}")));
    }
    Ok((t.numel() / (k * l), k, l))
}

fn convolve(a: &Tensor, b: &Tensor, unbind: bool) -> Result<Tensor> {
    let what = if unbind { "blockcode unbind" } else { "blockcode bind" };
    let (a_vecs, k, l) = shape_of(a, what)?;
    let (b_vecs, bk, bl) = shape_of(b, what)?;
    if (bk, bl) != (k, l) {
        return Err(Error::Shape(format!(
            "{what}: block shapes differ, ({k}, {l}) and ({bk}, {bl})"
        )));
    }
    // One operand may be a single code broadcast over the other's batch,
    // which is what binding a whole codebook against one key looks like.
    let (vecs, wide) = match (a_vecs, b_vecs) {
        (x, y) if x == y => (x, a),
        (1, y) => (y, b),
        (x, 1) => (x, a),
        (x, y) => {
            return Err(Error::Shape(format!(
                "{what}: {x} and {y} codes cannot be broadcast against each other"
            )))
        }
    };
    let mut shape = wide.shape().to_vec();
    if shape.len() < 2 {
        shape = vec![k, l];
    }
    let out = Tensor::empty(a.device(), &shape)?;
    let stride = |n: usize| if n == 1 { 0 } else { k * l };
    BIND.launch_flat(
        a.device(),
        &[Arg::Storage(a), Arg::Storage(b), Arg::Tensor(&out)],
        vecs * k * l,
        &[
            (vecs * k * l) as u32,
            k as u32,
            l as u32,
            u32::from(unbind),
            stride(a_vecs) as u32,
            stride(b_vecs) as u32,
            a.offset() as u32,
            b.offset() as u32,
        ],
    )?;
    Ok(out)
}

/// Per-block circular convolution: `bind(a, b)`.
///
/// Association. For one-hot blocks the hot position of the result is the
/// sum of the operands' hot positions, mod l, so binding is invertible by
/// [`unbind`] and exact.
pub fn bind(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    convolve(a, b, false)
}

/// Per-block circular correlation: `unbind(bind(x, key), key) == x`.
pub fn unbind(composite: &Tensor, key: &Tensor) -> Result<Tensor> {
    convolve(composite, key, true)
}
