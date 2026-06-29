//! Sealed element-type trait for SIMD-eligible scalars.

/// Backward-compatible alias for NumericElement.
pub use eunomia::NumericElement as Scalar;
pub use eunomia::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8, I16, I32, I8};
pub use eunomia::{CastFrom, CastTo, FloatElement, NumericElement};
