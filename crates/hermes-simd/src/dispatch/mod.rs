//! Runtime-dispatched SIMD operations.
//!
//! # Monomorphization chain
//!
//! `sum::<f32>(data)` -> `f32::sum(data)` -> `sum::dispatch_sum::<f32>(data)` -> avx2 kernel.

mod abs_reduce;
pub mod argmax;
pub mod argmin;
mod axpy;
pub mod binary;
pub mod complex;
pub mod dot;
pub mod gemm;
pub mod gemv;
pub mod gemv_strided;
pub mod gemv_transpose;
pub mod gemv_transpose_strided;
pub mod masked;
pub mod max;
pub mod min;
pub mod modular;
mod ops;
mod popcount;
mod real_interleave;
pub mod scale;
/// Sealed SIMD dispatch trait and blanket implementations.
pub mod simd_ops;
pub mod sparse;
pub mod sum;

pub use ops::*;
pub use popcount::{
    dispatch_reduce_popcount, dispatch_reduce_popcount_and, dispatch_reduce_popcount_or,
    dispatch_reduce_popcount_xor,
};
pub use real_interleave::{
    real_mul_to_interleaved_complex, real_mul_to_interleaved_complex_runtime,
};
pub use simd_ops::SimdOps;
