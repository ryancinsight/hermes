//! Clone-on-write container for lazy SIMD memory layout allocation.
//!
//! `SimdCow` provides zero-copy read access to borrowed slices via `SimdView`,
//! deferring heap allocation until mutable or owned access is required.
//!
//! CoW combinators (`zip_cow`, `transform_in_place`) apply elementwise `ElementOp`
//! strategies without unnecessary intermediate allocations:
//! - `zip_cow`: exactly one `AlignedVec` allocation for the output.
//! - `transform_in_place`: promotes to owned (one allocation) if borrowed, then
//!   mutates in-place — subsequent calls are allocation-free.
//!
//! Scalar tail handling uses `ElementOp::apply_scalar` to avoid vector load/store
//! boundary UB at the end of non-multiple-of-LANE_COUNT slices.

/// CoW combinators: `zip_cow`, `transform_in_place`, `reduce`, arithmetic shorthands,
/// and ergonomic `From`/`Extend` conversions.
pub mod combinators;
/// Unary map, in-place scale, splat-fill, argmin/argmax, gather, and prefix-scan extensions.
pub mod extensions;
/// `SimdCow` enum definition, accessors, and trait implementations.
pub mod types;

/// Norm, normalize, and scalar-broadcast arithmetic on Clone-on-Write SIMD containers.
pub mod math;
/// Operator overloads for Clone-on-Write SIMD containers.
pub mod ops;
/// Zero-copy serialization support for Clone-on-Write SIMD containers using `rkyv`.
pub mod rkyv;
/// Unary op dispatch (map_cow), ternary FMA (fma_cow), and clamp_cow.
pub mod unary;

pub use hermes_numeric::{ArchivedPacked4Cow, Packed4CowResolver};
pub use rkyv::{ArchivedSimdCow, SimdCowResolver};
pub use types::SimdCow;
