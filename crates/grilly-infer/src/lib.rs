//! Inference machinery for grilly2 (CLAUDE.md §11): weight loading, int8
//! weight-only quantization, GPU sampling and a GPU-resident KV cache; the
//! decode loop to follow. It depends on `grilly-core` alone, so it is usable
//! without any model-zoo adapter.

pub mod kv_cache;
pub mod quant;
pub mod safetensors;
pub mod sampling;

pub use crate::kv_cache::KvCache;
pub use crate::quant::Int8Weight;
pub use crate::safetensors::{LoadError, SafeTensorsFile};
pub use crate::sampling::Sampling;
