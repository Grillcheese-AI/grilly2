//! Fused block-sparse attention: flex_attention's kernels
//! (docs/flex-attention.md). The mask arrives as 32x32 tile lists built
//! once by `create_block_mask` (python/grilly/nn/attention/flex_attention.py).

use grilly_core::generic::{self as k, Binary, Reduce};
use grilly_core::{require_float, Arg, DType, Error, Result, Tensor};

use crate::kernels::{FLASH_BWD_KV, FLASH_BWD_Q, FLASH_FWD};
use crate::math::record;

/// The kernels' tile: 32 queries x 32 keys.
pub const TILE: usize = 32;
/// The largest head dim (of queries/keys, and of values) the kernels take.
pub const MAX_HEAD_DIM: usize = 128;

/// A mask as tile lists (int32 tensors), for `mb` mask batches and `mh`
/// mask heads (each 1, or the batch / head count): the bit mask, each query
/// tile's key tiles, and each key tile's query tiles.
#[derive(Clone)]
pub struct Tiles {
    pub bits: Tensor,
    pub kv_count: Tensor,
    pub kv_list: Tensor,
    pub q_count: Tensor,
    pub q_list: Tensor,
    pub mb: usize,
    pub mh: usize,
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

/// (B, H, L, S, D, Dv) of q (B, H, L, D), k (B, H, S, D), v (B, H, S, Dv).
fn dims(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<[usize; 6]> {
    for t in [q, k, v] {
        require_float("flex_attention", t.dtype())?;
        if t.dim() != 4 {
            return Err(Error::Shape("flex_attention: query, key and value must be 4-D".into()));
        }
    }
    let (b, h, l, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (s, dv) = (k.shape()[2], v.shape()[3]);
    if k.shape() != [b, h, s, d] || v.shape()[..3] != [b, h, s] {
        return Err(Error::Shape(format!(
            "flex_attention: shapes {:?}, {:?}, {:?} do not match",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    if d > MAX_HEAD_DIM || dv > MAX_HEAD_DIM {
        return Err(Error::NotImplemented(format!("fused attention with head dim > {MAX_HEAD_DIM}")));
    }
    Ok([b, h, l, s, d, dv])
}

/// The push constants shared by the three kernels.
fn push(dims: [usize; 6], tiles: &Tiles, scale: f32, offsets: &[usize]) -> Result<Vec<u32>> {
    let [b, h, l, s, d, dv] = dims;
    let mut words = vec![
        u32_of(b, "batch")?,
        u32_of(h, "heads")?,
        u32_of(l, "L")?,
        u32_of(s, "S")?,
        u32_of(d, "D")?,
        u32_of(dv, "Dv")?,
        scale.to_bits(),
        u32_of(tiles.mb, "mask batch")?,
        u32_of(tiles.mh, "mask heads")?,
        u32_of(l.div_ceil(TILE), "query tiles")?,
        u32_of(s.div_ceil(TILE), "key tiles")?,
    ];
    for &o in offsets {
        words.push(u32_of(o, "offset")?);
    }
    Ok(words)
}

/// `softmax(q k^T * scale, masked) @ v` for q (B, H, L, D), k (B, H, S, D),
/// v (B, H, S, Dv): the output (B, H, L, Dv) and the log-sum-exp (B, H, L).
/// Not recorded for autograd; see [`flex_attention`].
pub fn flash_forward(q: &Tensor, k: &Tensor, v: &Tensor, tiles: &Tiles, scale: f32) -> Result<(Tensor, Tensor)> {
    let dims = dims(q, k, v)?;
    let [b, h, l, _, _, dv] = dims;
    for t in [&tiles.bits, &tiles.kv_count, &tiles.kv_list, &tiles.q_count, &tiles.q_list] {
        if t.dtype() != DType::I32 {
            return Err(Error::Invalid("flex_attention: tile lists must be int32".into()));
        }
    }
    let out = Tensor::empty(q.device(), &[b, h, l, dv])?;
    let lse = Tensor::empty(q.device(), &[b, h, l])?;
    if b * h * l == 0 {
        return Ok((out, lse));
    }
    let push = push(dims, tiles, scale, &[q.offset(), k.offset(), v.offset()])?;
    let args = [
        Arg::Storage(q),
        Arg::Storage(k),
        Arg::Storage(v),
        Arg::Tensor(&tiles.bits),
        Arg::Tensor(&tiles.kv_count),
        Arg::Tensor(&tiles.kv_list),
        Arg::Storage(&out),
        Arg::Storage(&lse),
    ];
    let threads = [u32_of(l.div_ceil(TILE) * 256, "query tiles")?, u32_of(h, "heads")?, u32_of(b, "batch")?];
    FLASH_FWD.launch(q.device(), &args, threads, &push)?;
    Ok((out, lse))
}

/// The gradients (dq, dk, dv) of [`flash_forward`] from `dout`, the
/// forward's `out` and `lse`: two kernels, no atomics, deterministic.
#[allow(clippy::too_many_arguments)]
pub fn flash_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    out: &Tensor,
    lse: &Tensor,
    dout: &Tensor,
    tiles: &Tiles,
    scale: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = dims(q, k, v)?;
    let [b, h, l, s, d, dv] = dims;
    let dq = Tensor::zeros(q.device(), &[b, h, l, d])?;
    let dk = Tensor::zeros(q.device(), &[b, h, s, d])?;
    let dvv = Tensor::zeros(q.device(), &[b, h, s, dv])?;
    if b * h * l * s == 0 {
        return Ok((dq, dk, dvv));
    }
    // Drow = rowsum(dO * O), (B, H, L)
    let drow = k::reduce(&k::binary(dout, out, Binary::Mul)?, 3, Reduce::Sum, false)?;
    let push = push(dims, tiles, scale, &[q.offset(), k.offset(), v.offset(), dout.offset()])?;
    let common = [Arg::Storage(q), Arg::Storage(k), Arg::Storage(v), Arg::Storage(dout), Arg::Tensor(lse), Arg::Tensor(&drow)];
    let mut kv_args = common.to_vec();
    kv_args.extend([
        Arg::Tensor(&tiles.bits),
        Arg::Tensor(&tiles.q_count),
        Arg::Tensor(&tiles.q_list),
        Arg::Storage(&dk),
        Arg::Storage(&dvv),
    ]);
    let kv_threads = [u32_of(s.div_ceil(TILE) * 256, "key tiles")?, u32_of(h, "heads")?, u32_of(b, "batch")?];
    FLASH_BWD_KV.launch(q.device(), &kv_args, kv_threads, &push)?;
    let mut q_args = common.to_vec();
    q_args.extend([Arg::Tensor(&tiles.bits), Arg::Tensor(&tiles.kv_count), Arg::Tensor(&tiles.kv_list), Arg::Storage(&dq)]);
    let q_threads = [u32_of(l.div_ceil(TILE) * 256, "query tiles")?, u32_of(h, "heads")?, u32_of(b, "batch")?];
    FLASH_BWD_Q.launch(q.device(), &q_args, q_threads, &push)?;
    Ok((dq, dk, dvv))
}

/// flex_attention's fused path with autograd: (out, lse). The gradient
/// flows through `out` (FlexAttentionBackward0); `lse` is not
/// differentiable here.
pub fn flex_attention(q: &Tensor, k: &Tensor, v: &Tensor, tiles: &Tiles, scale: f32) -> Result<(Tensor, Tensor)> {
    let (mut out, lse) = flash_forward(q, k, v, tiles, scale)?;
    let tiles = tiles.clone();
    // saved: q, k, v, lse, then out (save_out)
    record("FlexAttentionBackward0", &[q, k, v], &mut out, &[q, k, v, &lse], true, move |g, s, needs| {
        let (dq, dk, dv) = flash_backward(&s[0], &s[1], &s[2], &s[4], &s[3], g, &tiles, scale)?;
        Ok(vec![needs[0].then_some(dq), needs[1].then_some(dk), needs[2].then_some(dv)])
    })?;
    Ok((out, lse))
}
