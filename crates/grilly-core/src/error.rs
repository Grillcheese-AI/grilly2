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
    /// torch supports this; grilly does not yet.
    NotImplemented(String),
    /// An index is out of range (Python's IndexError, where torch raises it).
    Index(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Device(m) => write!(f, "device error: {m}"),
            Error::Shape(m) => write!(f, "shape error: {m}"),
            Error::Invalid(m) => write!(f, "invalid argument: {m}"),
            Error::NotImplemented(m) => write!(f, "grilly: {m} is not implemented yet"),
            Error::Index(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}
