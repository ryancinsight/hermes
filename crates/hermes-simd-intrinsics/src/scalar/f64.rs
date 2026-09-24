//! Fallback scalar f64 kernel.
//!
//! All SIMD operations degenerate to element-wise scalar loops. This is the
//! universal fallback when no hardware SIMD feature is detected.

impl_scalar_float_kernel!(f64, 2, 63, 0xFFFF_FFFF_FFFF_FFFF_u64);
