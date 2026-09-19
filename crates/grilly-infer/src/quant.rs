//! int8 weight-only quantization (CLAUDE.md §11, step 2), with the numerics
//! of torchao's `Int8WeightOnlyConfig` at its default per-row granularity:
//!
//! - `scale = max|w| / 127.5` per output row, at least float32 eps;
//! - `q = clamp(round_half_even(w * (1 / scale)), -128, 127)`;
//! - `linear(x) = (x @ q^T) * scale + bias`.
//!
//! The quantized values are torchao's exactly (both divisions correctly
//! rounded on the GPU). The weight stays int8 in device memory and is
//! dequantized inside the matmul kernel, never materialized as float32.
//! The weight is frozen; gradients flow to the input and the bias, as in
//! torchao.

use grilly_core::autograd::{self, Backward};
use grilly_core::generic::{self, copy_strided, Reduce};
use grilly_core::{require_float, Arg, DType, Error, Kernel, MemClass, Result, Tensor};

/// Bindings: w, q, scale. Push: rows, k, words, w_off.
const QUANTIZE: Kernel = Kernel { shader: "g2-quantize-q8", bindings: 3, push_words: 4, local_size: [256, 1, 1] };
/// Bindings: x, q, scale, bias, y. Push: m, k, n, vecs, x_off, b_off, has_bias, x_vec4.
const LINEAR: Kernel = Kernel { shader: "g2-linear-q8", bindings: 5, push_words: 8, local_size: [256, 1, 1] };
/// Bindings: x, q, scale, bias, y. Push: m, k, n, vecs, x_off, b_off, has_bias, pad.
/// The same result as LINEAR, for many rows: a 64x64 output tile per workgroup.
const LINEAR_TILED: Kernel = Kernel { shader: "g2-linear-q8-tiled", bindings: 5, push_words: 8, local_size: [256, 1, 1] };
/// Bindings: q, scale, w. Push: rows, k, words.
const DEQUANTIZE: Kernel = Kernel { shader: "g2-dequantize-q8", bindings: 3, push_words: 3, local_size: [256, 1, 1] };

/// Rows of x per workgroup of the linear kernel (ROWS in g2-linear-q8.glsl).
const LINEAR_ROWS: usize = 8;
/// Output features per workgroup of the linear kernel.
const LINEAR_COLS: usize = 4;

pub const KERNELS: &[Kernel] = &[QUANTIZE, LINEAR, LINEAR_TILED, DEQUANTIZE];

/// The output tile of g2-linear-q8-tiled (BM, BN there).
const TILE: usize = 64;
/// Rows from which tiling can win. Below it the decode kernel reads the
/// weight once for up to LINEAR_ROWS rows anyway.
const TILED_MIN_ROWS: usize = 16;
/// Tiles a shape must make before tiling is worth it. One tile is one
/// workgroup, so a narrow layer (k_proj at 128 rows is 4 tiles) would leave
/// most of the card idle; the decode kernel's finer grid does not. Measured
/// on an RX 6750 XT, 40 compute units (docs/inference.md).
const TILED_MIN_TILES: usize = 16;

/// Whether `linear` tiles a given shape.
fn tiled_for(m: usize, n: usize) -> bool {
    m >= TILED_MIN_ROWS && n.div_ceil(TILE) * m.div_ceil(TILE) >= TILED_MIN_TILES
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

/// Workgroups covering `groups` along x (and y beyond 65535), as threads.
fn grid(groups: usize, local: u32) -> Result<[u32; 2]> {
    let gx = groups.clamp(1, 65_535);
    Ok([u32_of(gx, "groups")? * local, u32_of(groups.div_ceil(gx), "groups")?])
}

/// A linear layer's weight, int8 per output row with a float32 scale, in
/// persistent device memory.
pub struct Int8Weight {
    /// (out, words) 32-bit words: four int8 values each, little-endian, a row
    /// padded with zeros to whole 16-byte vectors
    q: Tensor,
    /// (out,) float32
    scale: Tensor,
    out_features: usize,
    in_features: usize,
}

impl Int8Weight {
    /// Quantize a float32 `(out_features, in_features)` weight.
    pub fn quantize(weight: &Tensor) -> Result<Int8Weight> {
        require_float("int8 weight-only quantization", weight.dtype())?;
        let &[out_features, in_features] = weight.shape() else {
            return Err(Error::Shape(format!("expected a 2-D weight (out_features, in_features), got {:?}", weight.shape())));
        };
        if in_features == 0 {
            return Err(Error::Shape("cannot quantize a weight with no input features".into()));
        }
        let words = in_features.div_ceil(16) * 4;
        let dev = weight.device();
        let q = Tensor::empty_in(dev, &[out_features, words], DType::I32, MemClass::Persistent)?;
        let scale = Tensor::empty_in(dev, &[out_features], DType::F32, MemClass::Persistent)?;
        if out_features > 0 {
            let [gx, gy] = grid(out_features, 256)?;
            let push =
                [u32_of(out_features, "out_features")?, u32_of(in_features, "in_features")?, u32_of(words, "words")?, u32_of(weight.offset(), "offset")?];
            QUANTIZE.launch(dev, &[Arg::Storage(weight), Arg::Tensor(&q), Arg::Tensor(&scale)], [gx, gy, 1], &push)?;
        }
        Ok(Int8Weight { q, scale, out_features, in_features })
    }

    /// `(out_features, in_features)`.
    pub fn shape(&self) -> [usize; 2] {
        [self.out_features, self.in_features]
    }

    /// The per-row scales, `(out_features,)` float32.
    pub fn scale(&self) -> &Tensor {
        &self.scale
    }

    /// Device bytes held: the int8 rows (with their padding) and the scales.
    pub fn nbytes(&self) -> usize {
        self.q.numel() * 4 + self.scale.numel() * 4
    }

    /// The int8 values, row-major `(out_features, in_features)`, on the host.
    pub fn qdata(&self) -> Result<Vec<i8>> {
        let bytes = self.q.to_bytes()?;
        let row = self.q.shape()[1] * 4;
        Ok((0..self.out_features).flat_map(|r| bytes[r * row..r * row + self.in_features].iter().map(|&b| b as i8)).collect())
    }

    /// One weight holding `parts`' rows in order: a fused q/k/v or gate/up
    /// projection. The scales are per row, so a fused linear computes each
    /// row exactly as the separate ones did.
    pub fn cat(parts: &[&Int8Weight]) -> Result<Int8Weight> {
        let Some(first) = parts.first() else {
            return Err(Error::Shape("Int8Weight::cat: no weights".into()));
        };
        let in_features = first.in_features;
        if parts.iter().any(|p| p.in_features != in_features) {
            return Err(Error::Shape(format!(
                "Int8Weight::cat: in_features differ ({:?})",
                parts.iter().map(|p| p.in_features).collect::<Vec<_>>()
            )));
        }
        let words = first.q.shape()[1];
        let out_features: usize = parts.iter().map(|p| p.out_features).sum();
        let dev = first.q.device();
        let q = Tensor::empty_in(dev, &[out_features, words], DType::I32, MemClass::Persistent)?;
        let scale = Tensor::empty_in(dev, &[out_features], DType::F32, MemClass::Persistent)?;
        let mut row = 0;
        for p in parts {
            copy_strided(&p.q, p.q.offset(), &[words, 1], &q, row * words, &[words, 1], &[p.out_features, words])?;
            copy_strided(&p.scale, p.scale.offset(), &[1], &scale, row, &[1], &[p.out_features])?;
            row += p.out_features;
        }
        Ok(Int8Weight { q, scale, out_features, in_features })
    }

    /// The weight as float32: `q * scale` per row (torchao's dequantize).
    pub fn dequantize(&self) -> Result<Tensor> {
        dequantize(&self.q, &self.scale, self.out_features, self.in_features)
    }

    /// `input @ weight^T + bias` over the last dimension of `input`, with the
    /// weight dequantized in the kernel. Recorded for autograd with respect
    /// to `input` and `bias`; the weight is frozen.
    pub fn linear(&self, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        require_float("linear", input.dtype())?;
        let (n, k) = (self.out_features, self.in_features);
        if input.shape().last() != Some(&k) {
            return Err(Error::Shape(format!("linear: input {:?} does not end in in_features = {k}", input.shape())));
        }
        if let Some(b) = bias {
            require_float("linear", b.dtype())?;
            if b.shape() != [n] {
                return Err(Error::Shape(format!("linear: bias must be ({n},), got {:?}", b.shape())));
            }
        }
        let mut out_shape = input.shape().to_vec();
        *out_shape.last_mut().unwrap() = n;
        let m = input.numel() / k.max(1);
        let dev = input.device();
        let mut y = Tensor::empty(dev, &out_shape)?;
        let mut inputs = vec![input];
        inputs.extend(bias);
        autograd::record("LinearInt8Backward0", &inputs, &mut [&mut y], |_| {
            Ok(LinearInt8Backward { q: self.q.clone(), scale: self.scale.clone(), n, k, input_shape: input.shape().to_vec() })
        })?;
        if m == 0 || n == 0 {
            return Ok(y);
        }
        let tiled = tiled_for(m, n);
        let mut push = [
            u32_of(m, "rows")?,
            u32_of(k, "in_features")?,
            u32_of(n, "out_features")?,
            u32_of(self.q.shape()[1] / 4, "vectors")?,
            u32_of(input.offset(), "offset")?,
            u32_of(bias.map_or(0, |b| b.offset()), "offset")?,
            bias.is_some() as u32,
            (input.offset().is_multiple_of(4) && k.is_multiple_of(4)) as u32,
        ];
        // With no bias the kernel never reads binding 3; bind the scales there.
        let b = bias.map_or(Arg::Tensor(&self.scale), Arg::Storage);
        let args = [Arg::Storage(input), Arg::Tensor(&self.q), Arg::Tensor(&self.scale), b, Arg::Tensor(&y)];
        if tiled {
            push[7] = 0;
            let groups = [u32_of(n.div_ceil(TILE) * 256, "out_features")?, u32_of(m.div_ceil(TILE), "rows")?, 1];
            LINEAR_TILED.launch(dev, &args, groups, &push)?;
        } else {
            let [gx, gy] = grid(n.div_ceil(LINEAR_COLS), 256)?;
            let gz = u32_of(m.div_ceil(LINEAR_ROWS), "rows")?;
            LINEAR.launch(dev, &args, [gx, gy, gz], &push)?;
        }
        Ok(y)
    }
}

fn dequantize(q: &Tensor, scale: &Tensor, n: usize, k: usize) -> Result<Tensor> {
    let dev = q.device();
    let w = Tensor::empty(dev, &[n, k])?;
    let push = [u32_of(n, "out_features")?, u32_of(k, "in_features")?, u32_of(q.shape()[1], "words")?];
    DEQUANTIZE.launch_flat(dev, &[Arg::Tensor(q), Arg::Tensor(scale), Arg::Tensor(&w)], w.numel(), &push)?;
    Ok(w)
}

/// grad_input = grad @ dequantize(weight); grad_bias = grad summed over the
/// leading dims. The dequantized weight exists only during the backward.
struct LinearInt8Backward {
    q: Tensor,
    scale: Tensor,
    n: usize,
    k: usize,
    input_shape: Vec<usize>,
}

impl Backward for LinearInt8Backward {
    fn apply(&mut self, grads: &[Option<Tensor>], needs: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let g = grads[0].as_ref().expect("single output");
        let m = g.numel() / self.n.max(1);
        let g2 = g.reshape(&[m, self.n])?;
        let mut result = vec![None; needs.len()];
        if needs[0] {
            let w = dequantize(&self.q, &self.scale, self.n, self.k)?;
            result[0] = Some(generic::bmm(&g2, &w, false, false)?.reshape(&self.input_shape)?);
        }
        if needs.len() > 1 && needs[1] {
            result[1] = Some(generic::reduce(&g2, 0, Reduce::Sum, false)?);
        }
        Ok(result)
    }
}
