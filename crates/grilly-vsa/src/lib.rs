//! Vector-symbolic architectures on the GPU.
//!
//! Two algebras live here, and they are **not** two implementations of one
//! thing (CLAUDE.md §8.3):
//!
//! - [`block_code`] — IBM NVSA sparse block codes. Float32, `(..., k, l)`:
//!   k blocks of l, each block one-hot (discrete) or a distribution over
//!   positions (continuous). Binding is per-block circular convolution.
//! - packed binary hypervectors — bits, 32 to a word. Binding is XOR,
//!   bundling is a majority vote, similarity is Hamming.
//!
//! They take different parameters, hold different dtypes and mean different
//! things by "bind". v1 let a capability check for one gate the other, which
//! is where the confusion started; nothing here couples them.

pub mod block_code;

pub use block_code::{bind, unbind};
