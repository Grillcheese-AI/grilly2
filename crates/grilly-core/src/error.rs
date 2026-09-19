use std::fmt;

/// Errors surfaced by grilly. Every failure reported by the GPU layer
/// arrives here with its message; nothing is swallowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The GPU layer refused or failed an operation.
    Device(String),
    /// Tensor shapes are incompatible with the operation.
    Shape(String),
    /// An argument is invalid for reasons other than shape.
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Device(m) => write!(f, "device error: {m}"),
            Error::Shape(m) => write!(f, "shape error: {m}"),
            Error::Invalid(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for Error {}
