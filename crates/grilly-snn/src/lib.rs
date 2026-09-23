//! Spiking neural networks: the neuron timestep, fused.
//!
//! A spiking neuron's step is three small stages — charge the membrane,
//! fire if it crossed the threshold, reset it — and every one of them is
//! elementwise over the same tensor. Composed from ordinary ops that is
//! **12 dispatches**, each reading and writing the whole thing. Here it is
//! one, and the backward is one more.
//!
//! The arithmetic was never the cost. Measured on an RX 6750 XT, the step
//! alone in a loop:
//!
//! | shape | composed | fused | |
//! |---|---:|---:|---:|
//! | (64, 256) | 12 dispatches, 0.200 ms | 1, 0.013 ms | 15.2x |
//! | (256, 1024) | 12, 0.228 ms | 1, 0.014 ms | 16.4x |
//! | (1024, 4096) | 12, 0.841 ms | 1, 0.068 ms | 12.4x |
//!
//! Small shapes are host-bound — twelve launches at ~10 µs each — and
//! large ones are memory-bound, twelve passes over 16 MB against one. GPU
//! time at (1024, 4096) goes 0.710 ms to 0.060 ms. Fusing removes both
//! costs, which is why this is a crate and not a Python helper.
//!
//! What is **not** here is the rest of `grilly.snn` — the surrogate
//! gradients as objects, place and time cells, spike coding, the sparse
//! expansion. Those are ordinary tensor expressions and they stay in
//! Python, the same way `grilly.vsa`'s bipolar and holographic algebras
//! did. A crate earns its place by holding a kernel.

pub mod lif;

pub use lif::{step, Reset, StepConfig, Surrogate};
