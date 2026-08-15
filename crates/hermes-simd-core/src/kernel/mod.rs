//! SIMD kernel contracts and operation-family capability facets.

mod backend;
mod roles;

pub use backend::{BackendKernel, MAX_SIMD_LANES};
pub use roles::{
    SimdArith, SimdBitwise, SimdCompare, SimdGather, SimdLoadStore, SimdMask, SimdPermute,
    SimdReduce, SimdStorage,
};

/// Aggregate capability implemented by every sealed SIMD backend.
///
/// Consumers that use several operation families can retain this aggregate
/// bound. Consumers that use one family should prefer the corresponding role
/// facet, such as [`SimdReduce`] or [`SimdGather`]. Each facet owns its public
/// operation contract and forwards to the single [`BackendKernel`] seam, so
/// this decomposition adds no dynamic dispatch or data movement.
pub trait SimdKernel<T: crate::scalar::Scalar>:
    SimdLoadStore<T>
    + SimdArith<T>
    + SimdBitwise<T>
    + SimdCompare<T>
    + SimdReduce<T>
    + SimdMask<T>
    + SimdGather<T>
    + SimdPermute<T>
{
}

impl<T: crate::scalar::Scalar, A: BackendKernel<T>> SimdKernel<T> for A {}
