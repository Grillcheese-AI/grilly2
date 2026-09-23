//! grilly-core: the device, GPU memory, the single kernel-dispatch path,
//! tensor storage and autograd. Everything a module (`grilly-torch`,
//! `grilly-vsa`, ...) needs from the GPU, and nothing more.
//!
//! Execution model: work is recorded into an open command batch on the
//! device and runs when the host next needs data (a download, an upload, or
//! an explicit [`Device::synchronize`]). Consecutive commands are separated
//! by a full barrier, so they observe each other's writes in order.

pub mod autograd;
pub mod fault;
mod device;
mod dtype;
mod error;
pub mod fusion;
pub mod generic;
pub mod graph;
mod kernel;
pub mod ops;
pub mod rng;
mod tensor;

pub use device::{Buffer, Caps, Device, MemClass, SubmitStats};
pub use dtype::{require_float, DType};
pub use error::{Error, Result};
pub use kernel::{Arg, Kernel};
pub use ops::add;
pub use tensor::Tensor;
