//! Shared register storage contract for operation-family facets.

use crate::scalar::Scalar;

use super::super::BackendKernel;

/// The associated register and lane contract shared by all SIMD facets.
pub trait SimdStorage<T: Scalar>: crate::private::Sealed {
    /// Backend-native vector register for `T`.
    type Vector: Copy + Send + Sync + 'static;

    /// Backend-native mask register for `T`.
    type Mask: Copy + Send + Sync + 'static;

    /// Backend-native integer index register for `T`.
    type IndexVector: Copy + Send + Sync + 'static;

    /// Number of `T` lanes in one vector register.
    const LANE_COUNT: usize;

    /// Compile-time proof that fixed fallback buffers can hold one register.
    const LANE_BOUND_CHECK: () = assert!(
        Self::LANE_COUNT <= super::super::MAX_SIMD_LANES,
        "SimdStorage::LANE_COUNT exceeds MAX_SIMD_LANES"
    );

    /// Register unrolling factor selected by the backend.
    const UNROLL_FACTOR: usize;

    /// Whether the backend provides a non-temporal store.
    const SUPPORTS_NT_STORE: bool;

    /// Whether this scalar/backend pair needs the x86 F16C target frame.
    #[doc(hidden)]
    const REQUIRES_F16C: bool = false;
}

impl<T: Scalar, A: BackendKernel<T>> SimdStorage<T> for A {
    type Vector = <A as BackendKernel<T>>::Vector;
    type Mask = <A as BackendKernel<T>>::Mask;
    type IndexVector = <A as BackendKernel<T>>::IndexVector;

    const LANE_COUNT: usize = <A as BackendKernel<T>>::LANE_COUNT;
    const UNROLL_FACTOR: usize = <A as BackendKernel<T>>::UNROLL_FACTOR;
    const SUPPORTS_NT_STORE: bool = <A as BackendKernel<T>>::SUPPORTS_NT_STORE;
    const REQUIRES_F16C: bool = <A as BackendKernel<T>>::REQUIRES_F16C;
}
