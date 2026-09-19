//! Inference machinery for grilly2 (CLAUDE.md §11): weight loading, int8
//! weight-only quantization and GPU sampling now; the decode loop to follow.
//! It depends on `grilly-core` alone, so it is usable without any model-zoo
//! adapter.

pub mod quant;
pub mod safetensors;
pub mod sampling;

pub use crate::quant::Int8Weight;
pub use crate::safetensors::{LoadError, SafeTensorsFile};
pub use crate::sampling::Sampling;
