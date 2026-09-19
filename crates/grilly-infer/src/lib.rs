//! Inference machinery for grilly2 (CLAUDE.md §11): weight loading now;
//! quantization, GPU sampling and the decode loop to follow. It depends on
//! `grilly-core` alone, so it is usable without any model-zoo adapter.

pub mod safetensors;

pub use crate::safetensors::{LoadError, SafeTensorsFile};
