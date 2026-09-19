//! Fused kernels for a decoder's step, each matching the ops it replaces
//! bit for bit (CLAUDE.md §9 case (a)): the rotary embedding and SwiGLU.
//! They read projections in their own `(batch, T, row)` layout, so no
//! transpose or slice copy is made to feed them.

use grilly_core::{require_float, Arg, Error, Kernel, Result, Tensor};

/// Bindings: src, cos, sin, dst. Push: n, t, heads, dim, row, col, src_off,
/// pos0, dst_off, ds_b, ds_h, ds_t.
pub(crate) const ROPE: Kernel = Kernel { shader: "g2-rope", bindings: 4, push_words: 12, local_size: [256, 1, 1] };
/// Bindings: gate, up, out. Push: n, width, g_off, g_row, g_col, u_off, u_row, u_col.
const SWIGLU: Kernel = Kernel { shader: "g2-swiglu", bindings: 3, push_words: 8, local_size: [256, 1, 1] };
pub const KERNELS: &[Kernel] = &[ROPE, SWIGLU];

pub(crate) fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Shape(format!("{what} = {v} exceeds u32")))
}

/// A projection's output `(batch, T, row)`: the heads it holds start at
/// column `col`.
pub struct Heads<'a> {
    pub src: &'a Tensor,
    pub col: usize,
    pub heads: usize,
    pub dim: usize,
}

impl Heads<'_> {
    /// (batch, T, row), checked against the heads it should hold.
    pub(crate) fn layout(&self) -> Result<(usize, usize, usize)> {
        require_float("rotary embedding", self.src.dtype())?;
        let &[b, t, row] = self.src.shape() else {
            return Err(Error::Shape(format!("expected a (batch, T, features) projection, got {:?}", self.src.shape())));
        };
        if !self.dim.is_multiple_of(2) || self.col + self.heads * self.dim > row {
            return Err(Error::Shape(format!(
                "{} heads of {} (even) from column {} do not fit rows of {row}",
                self.heads, self.dim, self.col
            )));
        }
        Ok((b, t, row))
    }
}

/// The rotary tables `(positions, dim)`, and the first position used.
pub struct Rotary<'a> {
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub pos0: usize,
}

impl Rotary<'_> {
    pub(crate) fn check(&self, t: usize, dim: usize) -> Result<()> {
        for table in [self.cos, self.sin] {
            require_float("rotary embedding", table.dtype())?;
            match *table.shape() {
                [positions, d] if d == dim && self.pos0 + t <= positions => {}
                _ => {
                    return Err(Error::Shape(format!(
                        "rotary tables {:?} do not cover positions {}..{} of dim {dim}",
                        table.shape(),
                        self.pos0,
                        self.pos0 + t
                    )))
                }
            }
        }
        Ok(())
    }
}

/// Record the rotary kernel writing `x` to `dst` at element offset
/// `dst_off` with strides `(batch, head, T)` (dim is contiguous).
pub(crate) fn rope_into(x: &Heads, rot: &Rotary, dst: &Tensor, dst_off: usize, strides: [usize; 3]) -> Result<()> {
    let (b, t, row) = x.layout()?;
    rot.check(t, x.dim)?;
    let n = b * t * x.heads * x.dim;
    if n == 0 {
        return Ok(());
    }
    let push = [
        u32_of(n, "elements")?,
        u32_of(t, "T")?,
        u32_of(x.heads, "heads")?,
        u32_of(x.dim, "dim")?,
        u32_of(row, "row")?,
        u32_of(x.col, "col")?,
        u32_of(x.src.offset(), "offset")?,
        u32_of(rot.pos0, "position")?,
        u32_of(dst_off, "offset")?,
        u32_of(strides[0], "stride")?,
        u32_of(strides[1], "stride")?,
        u32_of(strides[2], "stride")?,
    ];
    ROPE.launch_flat(
        x.src.device(),
        &[Arg::Storage(x.src), Arg::Storage(rot.cos), Arg::Storage(rot.sin), Arg::Storage(dst)],
        n,
        &push,
    )
}

/// The rotated heads as a new `(batch, heads, T, dim)` tensor.
pub fn rope(x: &Heads, rot: &Rotary) -> Result<Tensor> {
    let (b, t, _) = x.layout()?;
    let out = Tensor::empty(x.src.device(), &[b, x.heads, t, x.dim])?;
    rope_into(x, rot, &out, 0, [x.heads * t * x.dim, t * x.dim, x.dim])?;
    Ok(out)
}

/// `silu(gate) * up`, `(batch, T, width)`, reading `width` columns of each
/// from its source (`(batch, T, row)`) at its column.
pub fn swiglu(gate: &Tensor, gate_col: usize, up: &Tensor, up_col: usize, width: usize) -> Result<Tensor> {
    let mut rows = 0;
    let mut lead = Vec::new();
    for (t, col) in [(gate, gate_col), (up, up_col)] {
        require_float("swiglu", t.dtype())?;
        let Some((&row, l)) = t.shape().split_last() else {
            return Err(Error::Shape("swiglu: a 0-d source".into()));
        };
        if col + width > row {
            return Err(Error::Shape(format!("swiglu: columns {col}..{} do not fit rows of {row}", col + width)));
        }
        let r = l.iter().product::<usize>();
        if !lead.is_empty() && lead != l {
            return Err(Error::Shape(format!("swiglu: gate {:?} and up {:?} disagree", gate.shape(), up.shape())));
        }
        lead = l.to_vec();
        rows = r;
    }
    let mut shape = lead;
    shape.push(width);
    let out = Tensor::empty(gate.device(), &shape)?;
    let n = rows * width;
    if n > 0 {
        let row = |t: &Tensor| *t.shape().last().unwrap();
        let push = [
            u32_of(n, "elements")?,
            u32_of(width, "width")?,
            u32_of(gate.offset(), "offset")?,
            u32_of(row(gate), "row")?,
            u32_of(gate_col, "col")?,
            u32_of(up.offset(), "offset")?,
            u32_of(row(up), "row")?,
            u32_of(up_col, "col")?,
        ];
        SWIGLU.launch_flat(gate.device(), &[Arg::Storage(gate), Arg::Storage(up), Arg::Tensor(&out)], n, &push)?;
    }
    Ok(out)
}
