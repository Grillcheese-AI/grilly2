//! Inference machinery for grilly2 (CLAUDE.md §11): weight loading, int8
//! and int4 weight-only quantization, GPU sampling, a GPU-resident KV cache
//! and the decode loop. It depends on `grilly-core` alone, so it is usable
//! without any model-zoo adapter.

pub mod kv_cache;
pub mod ops;
pub mod quant;
pub mod quant4;
pub mod safetensors;
pub mod sampling;

pub use crate::kv_cache::KvCache;
pub use crate::ops::{rope, swiglu, Heads, Rotary};
pub use crate::quant::Int8Weight;
pub use crate::quant4::Int4Weight;
pub use crate::safetensors::{LoadError, SafeTensorsFile};
pub use crate::sampling::Sampling;
