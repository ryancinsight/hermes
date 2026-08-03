//! Aligned heap-allocated vector for SIMD workloads.

pub mod rkyv;

mod aligned;
mod tests;

pub use aligned::AlignedVec;