//! One integrate-and-fire timestep, forward and backward.
//!
//! ```text
//! h  = decay * (v - rest) + rest + in_scale * x
//! s  = h >= threshold
//! v' = soft ? h - threshold * s : h * (1 - s) + rest * s
//! ```
//!
//! `decay` and `in_scale` carry the neuron type: IF is `(1, 1)`, LIF is
//! `(1 - 1/tau, 1)`, and LIF with `decay_input` is `(1 - 1/tau, 1/tau)`.
//! One kernel covers all of them because they differ only in two numbers.
//!
//! **Tolerance.** The fused step is the same arithmetic in the same order
//! as the composed one, so `grilly.snn`'s test asserts them bit-exact
//! against each other (CLAUDE.md §9(a)): there is no reassociation here to
//! excuse a difference.
//!
//! `tau` is a push constant, so a **learnable** `tau` gets no gradient
//! from this path. `ParametricLIFNode` keeps the composed ops for that
//! reason, and the two are tested to agree.

use grilly_core::autograd::{self, Backward, SavedTensor};
use grilly_core::{require_float, Arg, Error, Kernel, Result, Tensor};

/// Bindings: v, x, spikes, v', h. Push: n, decay, in_scale, rest,
/// threshold, soft, v_off, x_off, want_h.
const STEP: Kernel = Kernel { shader: "g2-snn-lif-step", bindings: 5, push_words: 9, local_size: [256, 1, 1] };
/// Bindings: h, spikes, dL/ds, dL/dv', dL/dx, dL/dv. Push: n, decay,
/// in_scale, rest, threshold, soft, detach_reset, kind, alpha, has_gs,
/// has_gv, pad.
const BACKWARD: Kernel = Kernel { shader: "g2-snn-lif-step-backward", bindings: 6, push_words: 12, local_size: [256, 1, 1] };

pub const KERNELS: &[Kernel] = &[STEP, BACKWARD];

/// Which surrogate stands in for the step's derivative on the way back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surrogate {
    ATan,
    Sigmoid,
    FastSigmoid,
}

impl Surrogate {
    fn code(self) -> u32 {
        match self {
            Surrogate::ATan => 0,
            Surrogate::Sigmoid => 1,
            Surrogate::FastSigmoid => 2,
        }
    }
}

/// What happens to a neuron that just fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reset {
    /// Back to `rest`, throwing the overshoot away.
    Hard,
    /// Subtract the threshold, keeping it — so a strongly driven neuron
    /// comes back sooner.
    Soft,
}

/// A neuron's constants for one step.
#[derive(Clone, Copy, Debug)]
pub struct StepConfig {
    /// `1 - 1/tau`; 1.0 for a non-leaky neuron.
    pub decay: f32,
    /// What the input current is multiplied by: `1/tau`, or 1.0.
    pub in_scale: f32,
    /// The resting potential a hard reset returns to. 0.0 for a soft one.
    pub rest: f32,
    pub threshold: f32,
    pub reset: Reset,
    /// Cut the gradient through the reset, shortening the path back
    /// through time.
    pub detach_reset: bool,
    pub surrogate: Surrogate,
    /// The surrogate's width.
    pub alpha: f32,
}

impl StepConfig {
    fn soft(&self) -> u32 {
        u32::from(self.reset == Reset::Soft)
    }

    fn rest(&self) -> f32 {
        if self.reset == Reset::Soft {
            0.0
        } else {
            self.rest
        }
    }
}

/// One timestep: `(spikes, next membrane)`.
///
/// `v` and `x` must be the same shape; the outputs take it.
pub fn step(v: &Tensor, x: &Tensor, config: StepConfig) -> Result<(Tensor, Tensor)> {
    require_float("snn step", v.dtype())?;
    require_float("snn step", x.dtype())?;
    if v.shape() != x.shape() {
        return Err(Error::Shape(format!(
            "snn step: the membrane is {:?} and the input is {:?}; they must match",
            v.shape(),
            x.shape()
        )));
    }
    let dev = v.device();
    let n = v.numel();
    let mut spikes = Tensor::empty(dev, v.shape())?;
    let mut next = Tensor::empty(dev, v.shape())?;
    // `h` is only for the backward. Under inference the kernel is told not
    // to write it and the binding goes to a tensor it never touches.
    let wants_grad = autograd::is_grad_enabled() && (v.requires_grad() || x.requires_grad());
    let h = if wants_grad { Tensor::empty(dev, v.shape())? } else { spikes.clone() };

    autograd::record("SnnLifStepBackward0", &[v, x], &mut [&mut spikes, &mut next], |outs| {
        Ok(LifStepBackward {
            h: SavedTensor::save(&h)?,
            spikes: SavedTensor::save(outs[0])?,
            shape: v.shape().to_vec(),
            config,
        })
    })?;
    if n == 0 {
        return Ok((spikes, next));
    }
    STEP.launch_flat(
        dev,
        &[Arg::Storage(v), Arg::Storage(x), Arg::Tensor(&spikes), Arg::Tensor(&next), Arg::Tensor(&h)],
        n,
        &[
            n as u32,
            config.decay.to_bits(),
            config.in_scale.to_bits(),
            config.rest().to_bits(),
            config.threshold.to_bits(),
            config.soft(),
            v.offset() as u32,
            x.offset() as u32,
            u32::from(wants_grad),
        ],
    )?;
    Ok((spikes, next))
}

/// Holds the pre-reset membrane and the spikes, which is everything the
/// backward needs: the input and the old membrane are not kept alive.
///
/// Through [`SavedTensor`], so the two behave like any other op's saved
/// values — freed after the backward unless the graph is retained, and
/// refusing if something wrote into them in place meanwhile.
struct LifStepBackward {
    h: SavedTensor,
    spikes: SavedTensor,
    shape: Vec<usize>,
    config: StepConfig,
}

impl Backward for LifStepBackward {
    fn apply(&mut self, grads: &[Option<Tensor>], needs: &[bool]) -> Result<Vec<Option<Tensor>>> {
        let (h, spikes) = (self.h.get()?, self.spikes.get()?);
        let dev = h.device();
        let n = h.numel();
        if grads.iter().all(Option::is_none) || n == 0 {
            return Ok(vec![None; needs.len()]);
        }
        let g_x = Tensor::empty(dev, &self.shape)?;
        let g_v = Tensor::empty(dev, &self.shape)?;
        // The kernel reads a binding only when its `has_` flag is set, so an
        // absent gradient binds the spikes rather than allocating zeros.
        let g_spike = grads[0].as_ref().unwrap_or(&spikes);
        let g_next = grads[1].as_ref().unwrap_or(&spikes);
        let c = self.config;
        BACKWARD.launch_flat(
            dev,
            &[
                Arg::Tensor(&h),
                Arg::Tensor(&spikes),
                Arg::Tensor(g_spike),
                Arg::Tensor(g_next),
                Arg::Tensor(&g_x),
                Arg::Tensor(&g_v),
            ],
            n,
            &[
                n as u32,
                c.decay.to_bits(),
                c.in_scale.to_bits(),
                c.rest().to_bits(),
                c.threshold.to_bits(),
                c.soft(),
                u32::from(c.detach_reset),
                c.surrogate.code(),
                c.alpha.to_bits(),
                u32::from(grads[0].is_some()),
                u32::from(grads[1].is_some()),
                0,
            ],
        )?;
        let mut out = vec![None; needs.len()];
        if needs[0] {
            out[0] = Some(g_v);
        }
        if needs[1] {
            out[1] = Some(g_x);
        }
        Ok(out)
    }

    fn release(&mut self) {
        self.h.release();
        self.spikes.release();
    }
}
