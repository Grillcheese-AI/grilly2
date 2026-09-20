//! int4 weight-only quantization (CLAUDE.md §11, step 2's "int4 if it can be
//! done well"), with the numerics of torchao's `Int4WeightOnlyConfig` at its
//! default group size of 128:
//!
//! - `scale = (max - min) / 15` over each group of `group_size` values along
//!   the row, at least float32 eps;
//! - `zero  = min + 8 * scale` (torchao's float zero-point domain);
//! - `q     = clamp(round_half_even((w - min) / scale), 0, 15)`;
//! - `dequantize(q) = (q - 8) * scale + zero = q * scale + min`.
//!
//! Group-wise and asymmetric is what makes int4 usable: one scale for a
//! whole row leaves four bits to cover every outlier in it.
//!
//! The weight stays int4 in device memory, two values per byte, and is
//! dequantized inside the matmul kernel. It is frozen, as in torchao;
//! gradients flow to the input and the bias.
//!
//! Against `Int8Weight`: half the bytes, and a scale and zero per group
//! rather than one scale per row. The scale cannot leave the dot product,
//! so the linear scales each group's partial sum as it finishes.

use grilly_core::autograd::{self, Backward};
use grilly_core::generic::{self, copy_strided, Reduce};
use grilly_core::{require_float, Arg, DType, Error, Kernel, MemClass, Result, Tensor};

/// Bindings: w, q, scale, zero. Push: rows, k, group, words, groups, w_off.
const QUANTIZE: Kernel = Kernel { shader: "g2-quantize-q4", bindings: 4, push_words: 6, local_size: [256, 1, 1] };
/// Bindings: x, q, scale, zero, bias, y. Push: m, k, n, vecs, group, groups, x_off, b_off, has_bias, x_vec4.
const LINEAR: Kernel = Kernel { shader: "g2-linear-q4", bindings: 6, push_words: 10, local_size: [256, 1, 1] };
/// Bindings: q, scale, zero, w. Push: rows, k, group, words, groups.
const DEQUANTIZE: Kernel = Kernel { shader: "g2-dequantize-q4", bindings: 4, push_words: 5, local_size: [256, 1, 1] };

pub const KERNELS: &[Kernel] = &[QUANTIZE, LINEAR, DEQUANTIZE];

/// torchao's default `group_size`.
pub const DEFAULT_GROUP_SIZE: usize = 128;
/// Output features per workgroup of the linear kernel.
const LINEAR_COLS: usize = 4;
/// Rows of x per workgroup of the linear kernel (ROWS in g2-linear-q4.glsl).
const LINEAR_ROWS: usize = 8;

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

fn grid(groups: usize, local: u32) -> Result<[u32; 2]> {
    let gx = groups.clamp(1, 65_535);
    Ok([u32_of(gx, "groups")? * local, u32_of(groups.div_ceil(gx), "groups")?])
}

/// A linear layer's weight, int4 with a float32 scale and zero point per
/// group, in persistent device memory.
pub struct Int4Weight {
    /// (out, words) 32-bit words: eight 4-bit levels each, low nibble first
    q: Tensor,
    /// (out, groups) float32
    scale: Tensor,
    /// (out, groups) float32
    zero: Tensor,
    out_features: usize,
    in_features: usize,
    group_size: usize,
}

impl Int4Weight {
    /// Quantize a float32 `(out_features, in_features)` weight.
    pub fn quantize(weight: &Tensor, group_size: usize) -> Result<Int4Weight> {
        require_float("int4 weight-only quantization", weight.dtype())?;
        let &[out_features, in_features] = weight.shape() else {
            return Err(Error::Shape(format!("expected a 2-D weight (out_features, in_features), got {:?}", weight.shape())));
        };
        if in_features == 0 {
            return Err(Error::Shape("cannot quantize a weight with no input features".into()));
        }
        // 32 values per 16-byte vector, and a vector must lie in one group
        if group_size == 0 || !group_size.is_multiple_of(32) {
            return Err(Error::Invalid(format!("int4 group_size must be a positive multiple of 32, got {group_size}")));
        }
        let groups = in_features.div_ceil(group_size);
        let words = in_features.div_ceil(32) * 4;  // whole 16-byte vectors
        let dev = weight.device();
        let q = Tensor::empty_in(dev, &[out_features, words], DType::I32, MemClass::Persistent)?;
        let scale = Tensor::empty_in(dev, &[out_features, groups], DType::F32, MemClass::Persistent)?;
        let zero = Tensor::empty_in(dev, &[out_features, groups], DType::F32, MemClass::Persistent)?;
        if out_features > 0 {
            let push = [
                u32_of(out_features, "out_features")?,
                u32_of(in_features, "in_features")?,
                u32_of(group_size, "group_size")?,
                u32_of(words, "words")?,
                u32_of(groups, "groups")?,
                u32_of(weight.offset(), "offset")?,
            ];
            QUANTIZE.launch_flat(
                dev,
                &[Arg::Storage(weight), Arg::Tensor(&q), Arg::Tensor(&scale), Arg::Tensor(&zero)],
                out_features * groups,
                &push,
            )?;
        }
        Ok(Int4Weight { q, scale, zero, out_features, in_features, group_size })
    }

    /// `(out_features, in_features)`.
    pub fn shape(&self) -> [usize; 2] {
        [self.out_features, self.in_features]
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Device bytes held by the levels, scales and zero points.
    pub fn nbytes(&self) -> usize {
        (self.q.numel() + self.scale.numel() + self.zero.numel()) * 4
    }

    /// The packed levels, as bytes: two values per byte, the earlier one in
    /// the low nibble.
    pub fn qdata(&self) -> Result<Vec<u8>> {
        let bytes = self.q.to_bytes()?;
        let row = self.q.shape()[1] * 4;
        Ok((0..self.out_features)
            .flat_map(|r| bytes[r * row..r * row + self.in_features.div_ceil(2)].to_vec())
            .collect())
    }

    pub fn scale(&self) -> &Tensor {
        &self.scale
    }

    pub fn zero(&self) -> &Tensor {
        &self.zero
    }

    /// An int4 weight from levels and parameters computed elsewhere — by
    /// GPTQ, say, which chooses them to cancel a layer's output error
    /// rather than each weight's own (`grilly.infer.gptq`).
    ///
    /// `q` holds the packed levels as this type stores them: `(out, words)`
    /// 32-bit words, eight 4-bit levels each, low nibble first, rows padded
    /// to whole 16-byte vectors. `scale` and `zero` are `(out, groups)`.
    /// The three are copied into persistent memory, as `quantize` allocates.
    pub fn from_parts(q: &Tensor, scale: &Tensor, zero: &Tensor, in_features: usize, group_size: usize) -> Result<Int4Weight> {
        if !group_size.is_multiple_of(32) || group_size == 0 {
            return Err(Error::Invalid(format!("int4 group_size must be a positive multiple of 32, got {group_size}")));
        }
        let &[out_features, words] = q.shape() else {
            return Err(Error::Shape(format!("expected packed levels (out_features, words), got {:?}", q.shape())));
        };
        let groups = in_features.div_ceil(group_size);
        if words != in_features.div_ceil(32) * 4 {
            return Err(Error::Shape(format!(
                "packed levels must be {} words per row for in_features {in_features}, got {words}",
                in_features.div_ceil(32) * 4
            )));
        }
        if scale.shape() != [out_features, groups] || zero.shape() != [out_features, groups] {
            return Err(Error::Shape(format!(
                "scale and zero must be ({out_features}, {groups}), got {:?} and {:?}",
                scale.shape(),
                zero.shape()
            )));
        }
        if q.dtype() != DType::I32 {
            return Err(Error::Invalid(format!("packed levels must be int32 words, got {:?}", q.dtype())));
        }
        require_float("int4 weight", scale.dtype())?;
        require_float("int4 weight", zero.dtype())?;
        let dev = q.device();
        let held = |src: &Tensor, shape: &[usize], dtype| -> Result<Tensor> {
            let out = Tensor::empty_in(dev, shape, dtype, MemClass::Persistent)?;
            copy_strided(src, src.offset(), &[shape[1], 1], &out, 0, &[shape[1], 1], shape)?;
            Ok(out)
        };
        Ok(Int4Weight {
            q: held(q, &[out_features, words], DType::I32)?,
            scale: held(scale, &[out_features, groups], DType::F32)?,
            zero: held(zero, &[out_features, groups], DType::F32)?,
            out_features,
            in_features,
            group_size,
        })
    }

    /// One weight holding `parts`' rows in order: a fused q/k/v or gate/up
    /// projection. Each row keeps its own groups' scales and zero points, so
    /// the rows compute exactly what they did apart.
    pub fn cat(parts: &[&Int4Weight]) -> Result<Int4Weight> {
        let Some(first) = parts.first() else {
            return Err(Error::Shape("Int4Weight::cat: no weights".into()));
        };
        let (in_features, group_size) = (first.in_features, first.group_size);
        if parts.iter().any(|p| p.in_features != in_features || p.group_size != group_size) {
            return Err(Error::Shape(format!(
                "Int4Weight::cat: in_features or group_size differ ({:?})",
                parts.iter().map(|p| (p.in_features, p.group_size)).collect::<Vec<_>>()
            )));
        }
        let (words, groups) = (first.q.shape()[1], first.scale.shape()[1]);
        let out_features: usize = parts.iter().map(|p| p.out_features).sum();
        let dev = first.q.device();
        let q = Tensor::empty_in(dev, &[out_features, words], DType::I32, MemClass::Persistent)?;
        let scale = Tensor::empty_in(dev, &[out_features, groups], DType::F32, MemClass::Persistent)?;
        let zero = Tensor::empty_in(dev, &[out_features, groups], DType::F32, MemClass::Persistent)?;
        let mut row = 0;
        for p in parts {
            let rows = p.out_features;
            copy_strided(&p.q, p.q.offset(), &[words, 1], &q, row * words, &[words, 1], &[rows, words])?;
            copy_strided(&p.scale, p.scale.offset(), &[groups, 1], &scale, row * groups, &[groups, 1], &[rows, groups])?;
            copy_strided(&p.zero, p.zero.offset(), &[groups, 1], &zero, row * groups, &[groups, 1], &[rows, groups])?;
            row += rows;
        }
        Ok(Int4Weight { q, scale, zero, out_features, in_features, group_size })
    }

    /// The float32 weight the kernels see.
    pub fn dequantize(&self) -> Result<Tensor> {
        let dev = self.q.device();
        let w = Tensor::empty(dev, &[self.out_features, self.in_features])?;
        let push = [
            u32_of(self.out_features, "out_features")?,
            u32_of(self.in_features, "in_features")?,
            u32_of(self.group_size, "group_size")?,
            u32_of(self.q.shape()[1], "words")?,
            u32_of(self.scale.shape()[1], "groups")?,
        ];
        DEQUANTIZE.launch_flat(
            dev,
            &[Arg::Tensor(&self.q), Arg::Tensor(&self.scale), Arg::Tensor(&self.zero), Arg::Tensor(&w)],
            w.numel(),
            &push,
        )?;
        Ok(w)
    }

    /// `input @ weight^T + bias`, with the weight dequantized in the kernel.
    /// Recorded for autograd with respect to `input` and `bias`; the weight
    /// is frozen.
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
        let (q, scale, zero) = (self.q.clone(), self.scale.clone(), self.zero.clone());
        let (group, shape) = (self.group_size, input.shape().to_vec());
        autograd::record("LinearInt4Backward0", &inputs, &mut [&mut y], move |_| {
            Ok(LinearInt4Backward { q, scale, zero, n, k, group, input_shape: shape })
        })?;
        if m == 0 || n == 0 {
            return Ok(y);
        }
        let [gx, gy] = grid(n.div_ceil(LINEAR_COLS), 256)?;
        let gz = u32_of(m.div_ceil(LINEAR_ROWS), "rows")?;
        let push = [
            u32_of(m, "rows")?,
            u32_of(k, "in_features")?,
            u32_of(n, "out_features")?,
            u32_of(self.q.shape()[1] / 4, "vectors")?,
            u32_of(self.group_size, "group_size")?,
            u32_of(self.scale.shape()[1], "groups")?,
            u32_of(input.offset(), "offset")?,
            u32_of(bias.map_or(0, |b| b.offset()), "offset")?,
            bias.is_some() as u32,
            (input.offset().is_multiple_of(4) && k.is_multiple_of(4)) as u32,
        ];
        // With no bias the kernel never reads binding 4; bind the scales there.
        let b = bias.map_or(Arg::Tensor(&self.scale), Arg::Storage);
        LINEAR.launch(
            dev,
            &[Arg::Storage(input), Arg::Tensor(&self.q), Arg::Tensor(&self.scale), Arg::Tensor(&self.zero), b, Arg::Tensor(&y)],
            [gx, gy, gz],
            &push,
        )?;
        Ok(y)
    }
}

/// grad_input = grad @ dequantize(weight); grad_bias = grad summed over the
/// leading dims. The dequantized weight exists only during the backward.
struct LinearInt4Backward {
    q: Tensor,
    scale: Tensor,
    zero: Tensor,
    n: usize,
    k: usize,
    group: usize,
    input_shape: Vec<usize>,
}

impl Backward for LinearInt4Backward {
    fn apply(&mut self, grads: &[Option<Tensor>], needs: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let g = grads[0].as_ref().expect("single output");
        let m = g.numel() / self.n.max(1);
        let g2 = g.reshape(&[m, self.n])?;
        let mut result = vec![None; needs.len()];
        if needs[0] {
            let w = Int4Weight {
                q: self.q.clone(),
                scale: self.scale.clone(),
                zero: self.zero.clone(),
                out_features: self.n,
                in_features: self.k,
                group_size: self.group,
            }
            .dequantize()?;
            result[0] = Some(generic::bmm(&g2, &w, false, false)?.reshape(&self.input_shape)?);
        }
        if needs.len() > 1 && needs[1] {
            result[1] = Some(generic::reduce(&g2, 0, Reduce::Sum, false)?);
        }
        Ok(result)
    }
}
