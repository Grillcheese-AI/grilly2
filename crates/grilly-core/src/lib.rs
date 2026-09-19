//! grilly-core: the device, GPU memory, the single kernel-dispatch path, and
//! tensor storage. Everything a module (`grilly-torch`, `grilly-vsa`, ...)
//! needs from the GPU, and nothing more.
//!
//! Execution model: launches are recorded into an open command batch on the
//! device and run when the host next needs data (a download, an upload, or an
//! explicit [`Device::synchronize`]). Consecutive launches are separated by a
//! full compute barrier, so they observe each other's writes in order.

mod device;
mod error;
mod kernel;
mod tensor;

pub use device::{Buffer, Caps, Device, MemClass};
pub use error::{Error, Result};
pub use kernel::{Arg, Kernel};
pub use tensor::Tensor;
