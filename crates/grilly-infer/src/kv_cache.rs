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
//! **The fill level is a device value.** A `(layers,)` cursor buffer holds
//! how many positions each layer has: `g2-kv-write` writes at it,
//! `g2-kv-advance` moves it on, and the decode shaders read their length
//! from it. Nothing about a step's position reaches the GPU as a push
//! constant, which is what lets a decode step be recorded once and replayed
//! (`grilly_core::graph`) — a pushed length would be baked into the
//! recorded command and, since the cache's shapes never change with it,
//! every replay would quietly attend over the positions the capture saw.
//!
//! `lens` mirrors the cursor on the host, for `len()` and for refusing an
//! overrun before it is recorded. It is exact whenever the caller drives the
//! cache itself, and goes stale under replay, which has no host: see
//! [`KvCache::len`].

use grilly_core::{require_float, Arg, DType, Device, Error, Kernel, MemClass, Result, Tensor};

use crate::ops::{Heads, Rotary};

/// Bindings: q, k cache, v cache, scratch, out. Push: heads, kv_heads,
/// max_len, len, dim, scale, q_off, rows.
const ATTN_DECODE: Kernel = Kernel { shader: "g2-attn-decode", bindings: 6, push_words: 8, local_size: [256, 1, 1] };
/// Bindings: q, k cache, v cache, scratch, (m, l), partial out. Push: heads,
/// kv_heads, max_len, len, dim, scale, q_off, splits.
const ATTN_DECODE_SPLIT: Kernel = Kernel { shader: "g2-attn-decode-split", bindings: 7, push_words: 8, local_size: [256, 1, 1] };
/// Bindings: (m, l), partial out, out. Push: rows, splits, dim.
const ATTN_DECODE_MERGE: Kernel = Kernel { shader: "g2-attn-decode-merge", bindings: 3, push_words: 3, local_size: [256, 1, 1] };
/// Bindings: source, cos, sin, cache, cursor. Push: n, T, heads, dim, layer,
/// max_len, src_off, rope, the source's and the cache's three strides, and
/// the absolute position of the cache's slot 0.
const KV_WRITE: Kernel = Kernel { shader: "g2-kv-write", bindings: 5, push_words: 15, local_size: [256, 1, 1] };
/// Bindings: cursor. Push: layer, T, max_len.
const KV_ADVANCE: Kernel = Kernel { shader: "g2-kv-advance", bindings: 1, push_words: 3, local_size: [1, 1, 1] };
pub const KERNELS: &[Kernel] = &[ATTN_DECODE, ATTN_DECODE_SPLIT, ATTN_DECODE_MERGE, KV_WRITE, KV_ADVANCE];

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
    /// `(layers,)` filled positions, on the device. The shaders' source of
    /// truth; `lens` is the host's mirror of it.
    cursor: Tensor,
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
        let cursor = Tensor::full_typed(device, &[layers], 0.0, DType::I32)?;
        Ok(KvCache { k, v, cursor, lens: vec![0; layers], batch, kv_heads, head_dim, max_len })
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

    /// Positions filled in `layer` (Hugging Face's `get_seq_length`), as the
    /// host last saw it.
    ///
    /// This is the host's mirror of the device cursor, advanced by `append`.
    /// Replaying a captured step advances the cursor without running any
    /// host code, so after replays this lags: it says what the cache held
    /// when the host last drove it. The device path never consults it. Use
    /// [`KvCache::device_len`] for the number the shaders are using.
    pub fn len(&self, layer: usize) -> usize {
        self.lens.get(layer).copied().unwrap_or(0)
    }

    /// Positions filled in `layer`, read back from the device cursor — the
    /// number the shaders act on, including after replays. A device read, so
    /// it costs a submit; `len` is the cheap answer when the host is driving.
    pub fn device_len(&self, layer: usize) -> Result<usize> {
        self.layer(layer)?;
        let bytes = self.cursor.to_bytes()?;
        let at = layer * 4;
        Ok(u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize)
    }

    /// The cursor buffer, for a caller that has to rotate a query at the
    /// cache's own position without reading it back (`grilly_infer::ops`).
    pub fn cursor(&self) -> &Tensor {
        &self.cursor
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
    pub fn reset(&mut self) -> Result<()> {
        self.lens.iter_mut().for_each(|n| *n = 0);
        self.cursor.fill_(0.0).map(|_| ())
    }

    /// Write `T` positions into `dst` at the layer's cursor, rotating them
    /// first when `rot` is given, and advance the cursor. `src_strides` and
    /// `dst_strides` are (batch, head, position) in elements.
    fn write_at_cursor(
        &self,
        layer: usize,
        src: &Tensor,
        src_off: usize,
        src_strides: [usize; 3],
        rot: Option<(&Tensor, &Tensor, u32)>,
        dst: &Tensor,
        dst_strides: [usize; 3],
        t: usize,
    ) -> Result<()> {
        let (b, h, d) = (self.batch, self.kv_heads, self.head_dim);
        let n = b * h * t * d;
        if n == 0 {
            return Ok(());
        }
        let dev = src.device();
        // A descriptor set has to be complete, so the unused cos/sin slots
        // take the source; `rope = 0` means the shader never reads them.
        let (cos, sin, pos_base) = rot.unwrap_or((src, src, 0));
        KV_WRITE.launch_flat(
            dev,
            &[Arg::Storage(src), Arg::Storage(cos), Arg::Storage(sin), Arg::Storage(dst), Arg::Storage(&self.cursor)],
            n,
            &[
                u32_of(n, "elements")?,
                u32_of(t, "T")?,
                u32_of(h, "heads")?,
                u32_of(d, "head_dim")?,
                u32_of(layer, "layer")?,
                u32_of(self.max_len, "max_len")?,
                u32_of(src_off, "offset")?,
                u32::from(rot.is_some()),
                u32_of(src_strides[0], "stride")?,
                u32_of(src_strides[1], "stride")?,
                u32_of(src_strides[2], "stride")?,
                u32_of(dst_strides[0], "stride")?,
                u32_of(dst_strides[1], "stride")?,
                u32_of(dst_strides[2], "stride")?,
                pos_base,
            ],
        )
    }

    /// Move the layer's cursor on by `t`, on the device. A separate dispatch
    /// so it lands after the writes that used the old value: every recorded
    /// command sits behind a full barrier.
    fn advance(&self, device: &Device, layer: usize, t: usize) -> Result<()> {
        KV_ADVANCE.launch(
            device,
            &[Arg::Storage(&self.cursor)],
            [1, 1, 1],
            &[u32_of(layer, "layer")?, u32_of(t, "T")?, u32_of(self.max_len, "max_len")?],
        )
    }

    /// Refuse an overrun before it is recorded. The host mirror is exact
    /// whenever the host is driving; a replay saturates in the shader
    /// instead, which is the most a graph can do without a host.
    fn room_for(&self, layer: usize, t: usize) -> Result<usize> {
        let pos = self.lens[layer];
        if pos + t > self.max_len {
            return Err(Error::Invalid(format!(
                "KV cache full: layer {layer} holds {pos} of {} positions, {t} more asked",
                self.max_len
            )));
        }
        Ok(pos)
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
        let pos = self.room_for(layer, t)?;
        let m = self.max_len;
        let src = [h * t * d, t * d, d];
        let dst = [h * m * d, m * d, d];
        self.write_at_cursor(layer, keys, keys.offset(), src, None, &self.k[layer], dst, t)?;
        self.write_at_cursor(layer, values, values.offset(), src, None, &self.v[layer], dst, t)?;
        self.advance(keys.device(), layer, t)?;
        self.lens[layer] = pos + t;
        Ok(())
    }

    /// Write a step's keys and values at `layer`'s next `T` positions,
    /// taking them from projections in their own `(batch, T, row)` layout:
    /// the keys are rotated into the cache by the rotary kernel and the
    /// values copied, with no transpose or slice in between. Two dispatches.
    pub fn append_rotary(&mut self, layer: usize, keys: &Heads<'_>, values: &Heads<'_>, rot: &Rotary<'_>) -> Result<()> {
        self.layer(layer)?;
        let (b, t, v_row) = values.layout()?;
        let (kb, kt, _) = keys.layout()?;
        let (h, d, m) = (self.kv_heads, self.head_dim, self.max_len);
        if (kb, kt) != (b, t) || (keys.heads, keys.dim) != (h, d) || (values.heads, values.dim) != (h, d) || b != self.batch {
            return Err(Error::Shape(format!(
                "KV cache append: expected {h} heads of {d} for batch {}, got keys {}x{} and values {}x{} for batch {b}",
                self.batch, keys.heads, keys.dim, values.heads, values.dim
            )));
        }
        let pos = self.room_for(layer, t)?;
        rot.check(t, d)?;
        let (_, _, k_row) = keys.layout()?;
        let dst = [h * m * d, m * d, d];
        // a projection's (batch, T, row) layout, `heads` columns at `col`,
        // read in place: batch stride T*row, head stride dim, position row
        self.write_at_cursor(
            layer,
            keys.src,
            keys.src.offset() + keys.col,
            [t * k_row, d, k_row],
            // slot 0 of this cache holds absolute position `pos0 - pos`, and
            // that stays true as the cursor moves: the caller advances pos0
            // with the cache. Fixed for the cache's lifetime, so a push
            // constant is safe where the slot itself would not be.
            Some((rot.cos, rot.sin, (rot.pos0 as u32).wrapping_sub(pos as u32))),
            &self.k[layer],
            dst,
            t,
        )?;
        self.write_at_cursor(
            layer,
            values.src,
            values.src.offset() + values.col,
            [t * v_row, d, v_row],
            None,
            &self.v[layer],
            dst,
            t,
        )?;
        self.advance(rot.cos.device(), layer, t)?;
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
        // The split count is a push constant, so a captured dispatch keeps
        // it; the slices are cut from the *device* length, so they always
        // cover the cache however full it is. Choosing it from max_len rather
        // than the current length is what makes that stable — an early step
        // simply leaves the later slices empty, and they return at once.
        let splits = match splits {
            Some(0) => return Err(Error::Invalid("KV cache attention: splits must be at least 1".into())),
            Some(n) if n > MAX_SPLITS => {
                return Err(Error::Invalid(format!("KV cache attention: at most {MAX_SPLITS} splits, got {n}")))
            }
            Some(n) => n,
            None => splits_for(rows, self.max_len),
        };
        let gx = rows.min(65_535);
        let mut push = [
            u32_of(heads, "heads")?,
            u32_of(self.kv_heads, "kv_heads")?,
            u32_of(self.max_len, "max_len")?,
            u32_of(layer, "layer")?,
            u32_of(d, "head_dim")?,
            scale.to_bits(),
            u32_of(q.offset(), "offset")?,
            u32_of(rows, "rows")?,
        ];
        if splits == 1 || rows > 65_535 {
            ATTN_DECODE.launch(
                dev,
                &[
                    Arg::Storage(q),
                    Arg::Tensor(&self.k[layer]),
                    Arg::Tensor(&self.v[layer]),
                    Arg::Tensor(&scratch),
                    Arg::Tensor(&out),
                    Arg::Storage(&self.cursor),
                ],
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
                Arg::Storage(&self.cursor),
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
