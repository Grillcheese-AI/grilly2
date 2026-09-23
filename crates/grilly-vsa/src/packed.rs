//! Packed binary hypervectors: `dim` bits held 32 to a word.
//!
//! The other VSA algebra, and **not** a variant of the block codes in
//! [`crate::block_code`] (CLAUDE.md §8.3). Here a hypervector is a bit
//! string, 32x smaller than one float per dimension, and the algebra is:
//!
//! - bind — XOR, its own inverse, so unbinding is binding again. That is
//!   `bitwise_xor` on the packed words; there is no kernel here for it,
//!   because the framework already has one;
//! - bundle — a bit-wise majority vote, which is superposition;
//! - similarity — Hamming: `1 - distance / dim`. 1.0 identical, 0.5 for two
//!   unrelated vectors (not 0.0, which is the complement), so a similarity
//!   near 0.5 means "nothing in common" and is the number to compare
//!   against;
//! - permute — a cyclic rotation of the bits, which gives a sequence
//!   position its own role.
//!
//! Words are held as `i32` tensors, since that is grilly's 32-bit integer;
//! the shaders read the same bits as `uint`. The operations are integer
//! throughout, so every result here is exact — there is no tolerance to
//! state, and a test that needs one has found a bug.

use grilly_core::{Arg, DType, Error, Kernel, Result, Tensor};

/// Bindings: query, codebook, distances. Push: entries, words, two offsets.
const HAMMING: Kernel = Kernel { shader: "g2-hdc-hamming", bindings: 3, push_words: 4, local_size: [256, 1, 1] };
/// Bindings: vectors, out. Push: vectors, words, offset.
const BUNDLE: Kernel = Kernel { shader: "g2-hdc-bundle", bindings: 2, push_words: 3, local_size: [256, 1, 1] };
/// Bindings: source, out. Push: words out, words per vector, dim, shift, offset.
const PERMUTE: Kernel = Kernel { shader: "g2-hdc-permute", bindings: 2, push_words: 5, local_size: [256, 1, 1] };

pub const KERNELS: &[Kernel] = &[HAMMING, BUNDLE, PERMUTE];

fn require_words(t: &Tensor, what: &str) -> Result<()> {
    if t.dtype() != DType::I32 {
        return Err(Error::Invalid(format!(
            "{what}: packed hypervectors are int32 words, got {:?}",
            t.dtype()
        )));
    }
    Ok(())
}

/// Hamming distance from `query` (`words`,) to every row of `codebook`
/// (n, `words`), as int32.
pub fn hamming(query: &Tensor, codebook: &Tensor) -> Result<Tensor> {
    require_words(query, "hamming")?;
    require_words(codebook, "hamming")?;
    let words = query.numel();
    if codebook.dim() != 2 || codebook.shape()[1] != words {
        return Err(Error::Shape(format!(
            "hamming: codebook must be (n, {words}) to match the query, got {:?}",
            codebook.shape()
        )));
    }
    let n = codebook.shape()[0];
    let out = Tensor::empty_typed(query.device(), &[n], DType::I32)?;
    HAMMING.launch_flat(
        query.device(),
        &[Arg::Storage(query), Arg::Storage(codebook), Arg::Storage(&out)],
        n,
        &[n as u32, words as u32, query.offset() as u32, codebook.offset() as u32],
    )?;
    Ok(out)
}

/// Bit-wise majority vote over the rows of `vectors` (n, `words`).
///
/// A tie — only an even `n` can produce one — goes to 0, which is v1's rule.
pub fn bundle(vectors: &Tensor) -> Result<Tensor> {
    require_words(vectors, "bundle")?;
    if vectors.dim() != 2 || vectors.shape()[0] == 0 {
        return Err(Error::Shape(format!(
            "bundle: expected (n, words) with n > 0, got {:?}",
            vectors.shape()
        )));
    }
    let (n, words) = (vectors.shape()[0], vectors.shape()[1]);
    let out = Tensor::empty_typed(vectors.device(), &[words], DType::I32)?;
    BUNDLE.launch_flat(
        vectors.device(),
        &[Arg::Storage(vectors), Arg::Storage(&out)],
        words,
        &[n as u32, words as u32, vectors.offset() as u32],
    )?;
    Ok(out)
}

/// Rotate the bits of each packed vector by `shift`, cyclically within
/// `dim` bits: `out bit i = in bit (i - shift) mod dim`.
pub fn permute(x: &Tensor, dim: usize, shift: i64) -> Result<Tensor> {
    require_words(x, "permute")?;
    let words = *x.shape().last().unwrap_or(&0);
    if words == 0 || dim == 0 || dim > words * 32 || dim <= (words - 1) * 32 {
        return Err(Error::Shape(format!(
            "permute: {dim} bits do not fit {words} words (a word holds 32, and the last one must be used)"
        )));
    }
    let shift = shift.rem_euclid(dim as i64) as u32;
    let out = Tensor::empty_typed(x.device(), x.shape(), DType::I32)?;
    PERMUTE.launch_flat(
        x.device(),
        &[Arg::Storage(x), Arg::Storage(&out)],
        x.numel(),
        &[x.numel() as u32, words as u32, dim as u32, shift, x.offset() as u32],
    )?;
    Ok(out)
}
