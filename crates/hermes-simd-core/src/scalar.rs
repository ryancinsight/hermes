//! Sealed element-type trait for SIMD-eligible scalars.

pub use hermes_numeric::{NumericElement, FloatElement, CastFrom, CastTo};
/// Backward-compatible alias for NumericElement.
pub use hermes_numeric::NumericElement as Scalar;
pub use hermes_numeric::{F16, F32, F64, Bf16, Bf8, Bf4, F8, F4, I8, I16, I32};
