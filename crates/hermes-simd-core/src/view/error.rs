//! Error type shared by every SIMD view operation.

/// Error types for SIMD view operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdError {
    /// The lengths of the operand views do not match.
    LengthMismatch,
    /// A modular arithmetic parameter is outside its valid domain.
    InvalidModulus,
    /// The input slice is too small to load the requested vector.
    InsufficientInputLength,
    /// The output slice is too small to store the results.
    InsufficientOutputLength,
    /// The memory address is not aligned as required.
    UnalignedAddress,
    /// An index is out of bounds of the view.
    IndexOutOfBounds,
    /// The current host cannot execute the requested SIMD target safely.
    UnsupportedTarget,
}

impl core::fmt::Display for SimdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch => write!(f, "Operand views have mismatched lengths"),
            Self::InvalidModulus => write!(f, "Modular arithmetic requires a non-zero modulus"),
            Self::InsufficientInputLength => write!(f, "Input slice has insufficient length"),
            Self::InsufficientOutputLength => write!(f, "Output slice has insufficient length"),
            Self::UnalignedAddress => {
                write!(f, "Memory address does not satisfy alignment constraints")
            }
            Self::IndexOutOfBounds => write!(f, "Index is out of bounds of the view"),
            Self::UnsupportedTarget => {
                write!(f, "SIMD target is not supported or enabled on this host")
            }
        }
    }
}

// Implement standard Error trait if std is available.
#[cfg(feature = "std")]
impl std::error::Error for SimdError {}
