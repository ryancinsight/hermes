//! Safely typed views over slices with static alignment, architecture dispatch, reference typestates, and execution mode.
//!
//! This module root is a passthrough facade: it declares the operation-family
//! leaf modules and re-exports their public surface, so every path that existed
//! before the split (`view::SimdView`, `view::SimdError`, ...) still resolves.

mod capability;
mod casts;
mod chunk;
mod error;
mod simd_view;

/// Vectorized indirect load (gather) operations.
pub mod gather;
/// Lane-masked math and compaction/expansion operations on SIMD views.
pub mod masked;
mod masked_compaction;
/// Standard elementwise and accumulation operations on SIMD views.
pub mod ops;
/// Standard exclusive mutable elementwise operations on SIMD views.
pub mod ops_mut;
/// Unrolled generic horizontal reductions on SIMD views.
pub mod reduce;
/// Inclusive/exclusive prefix scans and running min/max.
pub mod scan;
/// Vectorized indirect store (scatter) operations.
pub mod scatter;
/// Lane-wise conditional select and masked-negate.
pub mod select;
/// 2D matrix tile views and operations.
pub mod tile;
/// Unary mapping operations on SIMD views.
pub mod unary;

pub use capability::Simd;
pub use chunk::SimdChunk;
pub use error::SimdError;
pub use simd_view::SimdView;
pub use tile::{TileMatrixMultiply, TileView};

/// Module containing the SIMD mask register wrappers.
pub mod complex_reg;
pub mod mask_reg;
/// Module containing operator overload implementations for SIMD vectors.
pub mod vector_ops;
/// Module containing the generic SIMD vector register wrappers.
pub mod vector_reg;

pub use complex_reg::ComplexReg;
pub use mask_reg::Mask;
pub use vector_reg::Vector;

pub(crate) use simd_view::{check_lengths_equal, check_output_length};
