//! A key/value cache that stays on the GPU (CLAUDE.md §11, step 5).
//!
//! v1's cache (`cpp/src/ops/kv_cache.cpp`) is not used. Its append and
//! decode take and return host arrays: every call uploads, runs a kernel,
//! and downloads. Its storage is lossy by design (low-rank projection,
//! FP8 / INT4). Its ops drive v1's pool and command batch directly, which
//! would be a second path to the GPU beside grilly's dispatch spine. Here
//! the cache is exact float32, preallocated per layer as
//! `(batch, kv_heads, max_len, head_dim)` in persistent memory:
//! - [`KvCache::append`] writes new keys and values in place at the layer's
//!   current length (`g2-copy`);
//! - [`KvCache::attend`] runs one query token against the filled positions,
//!   with grouped-query attention by index, so the cache is never copied or
//!   read back. One workgroup per (batch, head) leaves most of the card idle
//!   at batch 1, so with a long cache the positions are split across
//!   workgroups and merged ("flash decoding"): `g2-attn-decode` for one
//!   slice, `g2-attn-decode-split` + `g2-attn-decode-merge` for several.
//!
//! No autograd: this is the inference path.
//!
//! **Not capturable.** The fill level lives on the host: `append` writes at
//! a host-computed offset and `attend` passes the length as a push
//! constant. A capture would record both as they stood, and since the
//! tensor shapes never change nothing downstream would notice — every
//! replay would attend over the positions the capture saw. So a cache
//! operation refuses while a capture is open (`grilly_core::graph`) rather
//! than let a replay return a quietly stale answer. Making the cache
//! capturable means moving the length to a device-side counter the shaders
//! read, which is not done here.

use grilly_core::generic::write_strided;
use grilly_core::{require_float, Arg, DType, Device, Error, Kernel, MemClass, Result, Tensor};

use crate::ops::{rope_into, Heads, Rotary};

/// Bindings: q, k cache, v cache, scratch, out. Push: heads, kv_heads,
/// max_len, len, dim, scale, q_off, rows.
const ATTN_DECODE: Kernel = Kernel { shader: "g2-attn-decode", bindings: 5, push_words: 8, local_size: [256, 1, 1] };
/// Bindings: q, k cache, v cache, scratch, (m, l), partial out. Push: heads,
/// kv_heads, max_len, len, dim, scale, q_off, splits.
const ATTN_DECODE_SPLIT: Kernel = Kernel { shader: "g2-attn-decode-split", bindings: 6, push_words: 8, local_size: [256, 1, 1] };
/// Bindings: (m, l), partial out, out. Push: rows, splits, dim.
const ATTN_DECODE_MERGE: Kernel = Kernel { shader: "g2-attn-decode-merge", bindings: 3, push_words: 3, local_size: [256, 1, 1] };
pub const KERNELS: &[Kernel] = &[ATTN_DECODE, ATTN_DECODE_SPLIT, ATTN_DECODE_MERGE];

/// Cache positions per slice, at least: with fewer, a workgroup's 256
/// threads run dry and the split costs more than it wins. Measured: at 256
/// filled positions one workgroup per row is already the fastest.
const MIN_CHUNK: usize = 256;
/// Workgroups to aim for when splitting. Measured on an RX 6750 XT (40
/// compute units, 14 rows at batch 1); the win is flat from here to four
/// times as many slices (the table in docs/inference.md).
const TARGET_GROUPS: usize = 256;
/// The merge holds one weight per slice in a workgroup's threads.
const MAX_SPLITS: usize = 256;

/// How many slices to cut `len` positions into for `rows` (batch * head)
/// queries. 1 means the single-workgroup kernel.
fn splits_for(rows: usize, len: usize) -> usize {
    TARGET_GROUPS.div_ceil(rows.max(1)).clamp(1, (len / MIN_CHUNK).clamp(1, MAX_SPLITS))
}

/// Largest head dimension g2-attn-decode takes (its lanes per group).
pub const MAX_HEAD_DIM: usize = 256;

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

/// Keys and values of every layer, on the device.
pub struct KvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    lens: Vec<usize>,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
    max_len: usize,
}

impl KvCache {
    /// An empty cache for `layers` layers of `(batch, kv_heads, max_len,
    /// head_dim)` keys and values.
    pub fn new(device: &Device, layers: usize, batch: usize, kv_heads: usize, head_dim: usize, max_len: usize) -> Result<KvCache> {
        if kv_heads == 0 || head_dim == 0 || max_len == 0 || batch == 0 {
            return Err(Error::Shape("KV cache: batch, kv_heads, head_dim and max_len must be positive".into()));
        }
        if head_dim > MAX_HEAD_DIM {
            return Err(Error::NotImplemented(format!("a KV cache with head_dim {head_dim} (at most {MAX_HEAD_DIM})")));
        }
        let shape = [batch, kv_heads, max_len, head_dim];
        let alloc = || Tensor::empty_in(device, &shape, DType::F32, MemClass::Persistent);
        let (mut k, mut v) = (Vec::with_capacity(layers), Vec::with_capacity(layers));
        for _ in 0..layers {
            k.push(alloc()?);
            v.push(alloc()?);
        }
        Ok(KvCache { k, v, lens: vec![0; layers], batch, kv_heads, head_dim, max_len })
    }

    pub fn layers(&self) -> usize {
        self.lens.len()
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Positions filled in `layer` (Hugging Face's `get_seq_length`).
    pub fn len(&self, layer: usize) -> usize {
        self.lens.get(layer).copied().unwrap_or(0)
    }

    /// Whether no position of any layer is filled.
    pub fn is_empty(&self) -> bool {
        self.lens.iter().all(|&n| n == 0)
    }

    /// Device bytes held.
    pub fn nbytes(&self) -> usize {
        2 * self.layers() * self.batch * self.kv_heads * self.max_len * self.head_dim * 4
    }

    /// Forget every position (the buffers stay allocated).
    pub fn reset(&mut self) {
        self.lens.iter_mut().for_each(|n| *n = 0);
    }

    /// A cache operation cannot be recorded into a command graph; see the
    /// module header for why refusing is the only safe answer.
    fn not_during_capture(device: &Device, what: &str) -> Result<()> {
        if device.capturing() {
            return Err(Error::Invalid(format!(
                "KV cache {what}: a KV cache cannot be captured. Its fill level is a host value that                  the recorded commands carry — the write offset, and the length the decode shader                  reads — so every replay would use the length the capture saw, while the shapes stay                  the same and hide it. Keep the cache outside the captured region."
            )));
        }
        Ok(())
    }

    fn layer(&self, layer: usize) -> Result<()> {
        if layer >= self.layers() {
            return Err(Error::Index(format!("KV cache: layer {layer} out of range for {} layers", self.layers())));
        }
        Ok(())
    }

    /// Write `(batch, kv_heads, T, head_dim)` keys and values at `layer`'s
    /// next `T` positions.
    pub fn append(&mut self, layer: usize, keys: &Tensor, values: &Tensor) -> Result<()> {
        self.layer(layer)?;
        KvCache::not_during_capture(keys.device(), "append")?;
        let (b, h, d) = (self.batch, self.kv_heads, self.head_dim);
        for (what, t) in [("keys", keys), ("values", values)] {
            require_float("KV cache", t.dtype())?;
            match *t.shape() {
                [tb, th, _, td] if (tb, th, td) == (b, h, d) => {}
                _ => {
                    return Err(Error::Shape(format!(
                        "KV cache append: {what} must be ({b}, {h}, T, {d}), got {:?}",
                        t.shape()
                    )))
                }
            }
        }
        let t = keys.shape()[2];
        if values.shape()[2] != t {
            return Err(Error::Shape(format!("KV cache append: {t} keys but {} values", values.shape()[2])));
        }
        let pos = self.lens[layer];
        if pos + t > self.max_len {
            return Err(Error::Invalid(format!(
                "KV cache full: layer {layer} holds {pos} of {} positions, {t} more asked",
                self.max_len
            )));
        }
        let m = self.max_len;
        let src = [h * t * d, t * d, d, 1];
        let dst = [h * m * d, m * d, d, 1];
        write_strided(keys, keys.offset(), &src, &self.k[layer], pos * d, &dst, &[b, h, t, d])?;
        write_strided(values, values.offset(), &src, &self.v[layer], pos * d, &dst, &[b, h, t, d])?;
        self.lens[layer] = pos + t;
        Ok(())
    }

    /// Write a step's keys and values at `layer`'s next `T` positions,
    /// taking them from projections in their own `(batch, T, row)` layout:
    /// the keys are rotated into the cache by the rotary kernel and the
    /// values copied, with no transpose or slice in between. Two dispatches.
    pub fn append_rotary(&mut self, layer: usize, keys: &Heads<'_>, values: &Heads<'_>, rot: &Rotary<'_>) -> Result<()> {
        self.layer(layer)?;
        KvCache::not_during_capture(rot.cos.device(), "append_rotary")?;
        let (b, t, v_row) = values.layout()?;
        let (kb, kt, _) = keys.layout()?;
        let (h, d, m) = (self.kv_heads, self.head_dim, self.max_len);
        if (kb, kt) != (b, t) || (keys.heads, keys.dim) != (h, d) || (values.heads, values.dim) != (h, d) || b != self.batch {
            return Err(Error::Shape(format!(
                "KV cache append: expected {h} heads of {d} for batch {}, got keys {}x{} and values {}x{} for batch {b}",
                self.batch, keys.heads, keys.dim, values.heads, values.dim
            )));
        }
        let pos = self.lens[layer];
        if pos + t > m {
            return Err(Error::Invalid(format!(
                "KV cache full: layer {layer} holds {pos} of {m} positions, {t} more asked"
            )));
        }
        rope_into(keys, rot, &self.k[layer], pos * d, [h * m * d, m * d, d])?;
        write_strided(
            values.src,
            values.src.offset() + values.col,
            &[t * v_row, d, v_row, 1],
            &self.v[layer],
            pos * d,
            &[h * m * d, m * d, d, 1],
            &[b, h, t, d],
        )?;
        self.lens[layer] = pos + t;
        Ok(())
    }

    /// `softmax(q . K^T * scale) @ V` for one query token per head,
    /// `q` of shape `(batch, heads, 1, head_dim)`, against `layer`'s filled
    /// positions. `heads` must be a multiple of `kv_heads` (grouped-query
    /// attention). `scale` defaults to `1 / sqrt(head_dim)`.
    pub fn attend(&self, layer: usize, q: &Tensor, scale: Option<f32>) -> Result<Tensor> {
        self.attend_in(layer, q, scale, None)
    }

    /// [`KvCache::attend`] with the number of slices forced, for tests and
    /// the benchmark that set [`splits_for`].
    pub fn attend_splits(&self, layer: usize, q: &Tensor, scale: Option<f32>, splits: usize) -> Result<Tensor> {
        self.attend_in(layer, q, scale, Some(splits))
    }

    fn attend_in(&self, layer: usize, q: &Tensor, scale: Option<f32>, splits: Option<usize>) -> Result<Tensor> {
        self.layer(layer)?;
        KvCache::not_during_capture(q.device(), "attend")?;
        require_float("KV cache attention", q.dtype())?;
        let (b, d) = (self.batch, self.head_dim);
        let heads = match *q.shape() {
            [qb, hq, 1, qd] if qb == b && qd == d && hq % self.kv_heads == 0 && hq > 0 => hq,
            _ => {
                return Err(Error::Shape(format!(
                    "KV cache attention: query must be ({b}, heads, 1, {d}) with heads a multiple of {}, got {:?}",
                    self.kv_heads,
                    q.shape()
                )))
            }
        };
        let len = self.lens[layer];
        if len == 0 {
            return Err(Error::Invalid(format!("KV cache attention: layer {layer} is empty")));
        }
        let dev = q.device();
        let rows = b * heads;
        let out = Tensor::empty(dev, &[b, heads, 1, d])?;
        let scratch = Tensor::empty(dev, &[rows, self.max_len])?;
        let scale = scale.unwrap_or(1.0 / (d as f32).sqrt());
        let splits = match splits {
            Some(0) => return Err(Error::Invalid("KV cache attention: splits must be at least 1".into())),
            Some(n) if n > MAX_SPLITS => {
                return Err(Error::Invalid(format!("KV cache attention: at most {MAX_SPLITS} splits, got {n}")))
            }
            Some(n) => n.min(len),
            None => splits_for(rows, len),
        };
        let gx = rows.min(65_535);
        let mut push = [
            u32_of(heads, "heads")?,
            u32_of(self.kv_heads, "kv_heads")?,
            u32_of(self.max_len, "max_len")?,
            u32_of(len, "len")?,
            u32_of(d, "head_dim")?,
            scale.to_bits(),
            u32_of(q.offset(), "offset")?,
            u32_of(rows, "rows")?,
        ];
        if splits == 1 || rows > 65_535 {
            ATTN_DECODE.launch(
                dev,
                &[Arg::Storage(q), Arg::Tensor(&self.k[layer]), Arg::Tensor(&self.v[layer]), Arg::Tensor(&scratch), Arg::Tensor(&out)],
                [u32_of(gx * 256, "rows")?, u32_of(rows.div_ceil(gx), "rows")?, 1],
                &push,
            )?;
            return Ok(out);
        }
        // one workgroup per (row, slice), then one per row to merge them
        let ml = Tensor::empty(dev, &[rows, splits, 2])?;
        let partial = Tensor::empty(dev, &[rows, splits, d])?;
        push[7] = u32_of(splits, "splits")?;
        ATTN_DECODE_SPLIT.launch(
            dev,
            &[
                Arg::Storage(q),
                Arg::Tensor(&self.k[layer]),
                Arg::Tensor(&self.v[layer]),
                Arg::Tensor(&scratch),
                Arg::Tensor(&ml),
                Arg::Tensor(&partial),
            ],
            [u32_of(rows * 256, "rows")?, u32_of(splits, "splits")?, 1],
            &push,
        )?;
        ATTN_DECODE_MERGE.launch(
            dev,
            &[Arg::Tensor(&ml), Arg::Tensor(&partial), Arg::Tensor(&out)],
            [u32_of(gx * 256, "rows")?, u32_of(rows.div_ceil(gx), "rows")?, 1],
            &[u32_of(rows, "rows")?, u32_of(splits, "splits")?, u32_of(d, "head_dim")?],
        )?;
        Ok(out)
    }
}
