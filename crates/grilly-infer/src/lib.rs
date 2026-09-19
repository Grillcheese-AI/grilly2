//! Inference machinery for grilly2 (CLAUDE.md §11): weight loading and int8
//! weight-only quantization now; GPU sampling and the decode loop to follow.
//! It depends on `grilly-core` alone, so it is usable without any model-zoo
//! adapter.

pub mod quant;
pub mod safetensors;

pub use crate::quant::Int8Weight;
pub use crate::safetensors::{LoadError, SafeTensorsFile};
