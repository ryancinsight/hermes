//! Reduction capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for horizontal and masked numeric reductions.
pub trait SimdReduce<T: Scalar>: SimdStorage<T> + Sealed {
    /// Sums all lanes.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn sum_reduce(v: Self::Vector) -> T;

    /// Sums active lanes and ignores inactive lanes.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> T;

    /// Reduces all lanes to their minimum.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn min_reduce(v: Self::Vector) -> T;

    /// Reduces all lanes to their maximum.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn max_reduce(v: Self::Vector) -> T;
}

impl<T: Scalar, A: BackendKernel<T>> SimdReduce<T> for A {
    unsafe fn sum_reduce(v: Self::Vector) -> T {
        <A as BackendKernel<T>>::sum_reduce(v)
    }

    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> T {
        <A as BackendKernel<T>>::masked_sum_reduce(v, mask)
    }

    unsafe fn min_reduce(v: Self::Vector) -> T {
        <A as BackendKernel<T>>::min_reduce(v)
    }

    unsafe fn max_reduce(v: Self::Vector) -> T {
        <A as BackendKernel<T>>::max_reduce(v)
    }
}
