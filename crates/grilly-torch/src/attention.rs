//! Fused block-sparse attention: flex_attention's kernels
//! (docs/flex-attention.md). The mask arrives as 32x32 tile lists built
//! once by `create_block_mask` (python/grilly/nn/attention/flex_attention.py).

use grilly_core::{require_float, Arg, DType, Error, Result, Tensor};

use crate::kernels::FLASH_FWD;

/// The kernels' tile: 32 queries x 32 keys.
pub const TILE: usize = 32;
/// The largest head dim (of queries/keys, and of values) the kernels take.
pub const MAX_HEAD_DIM: usize = 128;

/// A mask as tile lists (int32), for `mb` mask batches and `mh` mask heads
/// (each 1, or the batch / head count).
pub struct Tiles<'a> {
    pub bits: &'a Tensor,
    pub kv_count: &'a Tensor,
    pub kv_list: &'a Tensor,
    pub mb: usize,
    pub mh: usize,
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

/// `softmax(q k^T * scale, masked) @ v` for q (B, H, L, D), k (B, H, S, D),
/// v (B, H, S, Dv): the output (B, H, L, Dv) and the log-sum-exp (B, H, L).
/// Not recorded for autograd.
pub fn flash_forward(q: &Tensor, k: &Tensor, v: &Tensor, tiles: &Tiles, scale: f32) -> Result<(Tensor, Tensor)> {
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
    for t in [tiles.bits, tiles.kv_count, tiles.kv_list] {
        if t.dtype() != DType::I32 {
            return Err(Error::Invalid("flex_attention: tile lists must be int32".into()));
        }
    }
    let (nq, nk) = (l.div_ceil(TILE), s.div_ceil(TILE));
    let out = Tensor::empty(q.device(), &[b, h, l, dv])?;
    let lse = Tensor::empty(q.device(), &[b, h, l])?;
    if b * h * l == 0 {
        return Ok((out, lse));
    }
    let push = [
        u32_of(b, "batch")?,
        u32_of(h, "heads")?,
        u32_of(l, "L")?,
        u32_of(s, "S")?,
        u32_of(d, "D")?,
        u32_of(dv, "Dv")?,
        scale.to_bits(),
        u32_of(tiles.mb, "mask batch")?,
        u32_of(tiles.mh, "mask heads")?,
        u32_of(nq, "query tiles")?,
        u32_of(nk, "key tiles")?,
        u32_of(q.offset(), "offset")?,
        u32_of(k.offset(), "offset")?,
        u32_of(v.offset(), "offset")?,
    ];
    let args = [
        Arg::Storage(q),
        Arg::Storage(k),
        Arg::Storage(v),
        Arg::Tensor(tiles.bits),
        Arg::Tensor(tiles.kv_count),
        Arg::Tensor(tiles.kv_list),
        Arg::Storage(&out),
        Arg::Storage(&lse),
    ];
    let threads = [u32_of(nq * 256, "query tiles")?, u32_of(h, "heads")?, u32_of(b, "batch")?];
    FLASH_FWD.launch(q.device(), &args, threads, &push)?;
    Ok((out, lse))
}
