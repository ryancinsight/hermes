//! Fallback scalar SIMD kernels module.

#[macro_use]
mod float_kernel;

mod emulated;
pub mod f16;
pub mod f32;
pub mod f64;
pub mod tiling;
