//! Aligned heap-allocated vector for SIMD workloads.

pub mod rkyv;

mod aligned;
#[cfg(test)]
mod tests;

pub use aligned::AlignedVec;
