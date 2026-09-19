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
//! - [`KvCache::attend`] runs one query token against the filled positions
//!   (`g2-attn-decode`), with grouped-query attention by index, so the cache
//!   is never copied or read back.
//!
//! No autograd: this is the inference path.

use grilly_core::generic::write_strided;
use grilly_core::{require_float, Arg, DType, Device, Error, Kernel, MemClass, Result, Tensor};

/// Bindings: q, k cache, v cache, scratch, out. Push: heads, kv_heads,
/// max_len, len, dim, scale, q_off, rows.
const ATTN_DECODE: Kernel = Kernel { shader: "g2-attn-decode", bindings: 5, push_words: 8, local_size: [256, 1, 1] };
pub const KERNELS: &[Kernel] = &[ATTN_DECODE];

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

    /// `softmax(q . K^T * scale) @ V` for one query token per head,
    /// `q` of shape `(batch, heads, 1, head_dim)`, against `layer`'s filled
    /// positions. `heads` must be a multiple of `kv_heads` (grouped-query
    /// attention). `scale` defaults to `1 / sqrt(head_dim)`.
    pub fn attend(&self, layer: usize, q: &Tensor, scale: Option<f32>) -> Result<Tensor> {
        self.layer(layer)?;
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
        let gx = rows.min(65_535);
        let push = [
            u32_of(heads, "heads")?,
            u32_of(self.kv_heads, "kv_heads")?,
            u32_of(self.max_len, "max_len")?,
            u32_of(len, "len")?,
            u32_of(d, "head_dim")?,
            scale.to_bits(),
            u32_of(q.offset(), "offset")?,
            u32_of(rows, "rows")?,
        ];
        ATTN_DECODE.launch(
            dev,
            &[Arg::Storage(q), Arg::Tensor(&self.k[layer]), Arg::Tensor(&self.v[layer]), Arg::Tensor(&scratch), Arg::Tensor(&out)],
            [u32_of(gx * 256, "rows")?, u32_of(rows.div_ceil(gx), "rows")?, 1],
            &push,
        )?;
        Ok(out)
    }
}
