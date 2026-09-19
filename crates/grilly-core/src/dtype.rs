//! Element types. grilly stores float32, int64 (needs the device's
//! shaderInt64), int32 and bool. bool is held as one 32-bit word per element
//! (0 or 1) so kernels need no 8-bit storage support; that is invisible
//! except in memory use.

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    I64,
    I32,
    Bool,
}

impl DType {
    /// Bytes per element as stored.
    pub fn size(self) -> usize {
        match self {
            DType::I64 => 8,
            DType::F32 | DType::I32 | DType::Bool => 4,
        }
    }

    /// 32-bit words per element as stored.
    pub fn words(self) -> usize {
        self.size() / 4
    }

    pub fn is_floating(self) -> bool {
        self == DType::F32
    }

    /// torch's name.
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::I64 => "int64",
            DType::I32 => "int32",
            DType::Bool => "bool",
        }
    }

    /// The code kernels use (g2-cast.glsl).
    pub fn code(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::I64 => 1,
            DType::I32 => 2,
            DType::Bool => 3,
        }
    }

    pub fn from_code(code: u32) -> Result<DType> {
        Ok(match code {
            0 => DType::F32,
            1 => DType::I64,
            2 => DType::I32,
            3 => DType::Bool,
            _ => return Err(Error::Invalid(format!("unknown dtype code {code}"))),
        })
    }

    /// The bytes of `value` as one element of this dtype (as stored).
    pub fn encode(self, value: f64) -> Vec<u8> {
        match self {
            DType::F32 => (value as f32).to_le_bytes().to_vec(),
            DType::I64 => (value as i64).to_le_bytes().to_vec(),
            DType::I32 => (value as i32).to_le_bytes().to_vec(),
            DType::Bool => u32::from(value != 0.0).to_le_bytes().to_vec(),
        }
    }
}

/// Refuse an op that needs float32.
pub fn require_float(op: &str, dtype: DType) -> Result<()> {
    if dtype.is_floating() {
        Ok(())
    } else {
        Err(Error::NotImplemented(format!("{op} for dtype {}", dtype.name())))
    }
}
