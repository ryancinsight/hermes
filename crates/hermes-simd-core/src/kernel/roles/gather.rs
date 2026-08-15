//! Gather and scatter capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for indexed loads and stores.
pub trait SimdGather<T: Scalar>: SimdStorage<T> + Sealed {
    /// Gathers one value per lane from indexed memory.
    ///
    /// # Safety
    /// Every active address must be valid for a read.
    unsafe fn gather(base: *const T, indices: Self::IndexVector) -> Self::Vector;

    /// Gathers active lanes and preserves `src` in inactive lanes.
    ///
    /// # Safety
    /// Every active address must be valid for a read.
    unsafe fn gather_masked(
        base: *const T,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Scatters one value per lane to indexed memory.
    ///
    /// # Safety
    /// Every address must be valid for a write.
    unsafe fn scatter(base: *mut T, indices: Self::IndexVector, val: Self::Vector);

    /// Scatters active lanes to indexed memory.
    ///
    /// # Safety
    /// Every active address must be valid for a write.
    unsafe fn scatter_masked(
        base: *mut T,
        indices: Self::IndexVector,
        mask: Self::Mask,
        val: Self::Vector,
    );
}

impl<T: Scalar, A: BackendKernel<T>> SimdGather<T> for A {
    unsafe fn gather(base: *const T, indices: Self::IndexVector) -> Self::Vector {
        <A as BackendKernel<T>>::gather(base, indices)
    }

    unsafe fn gather_masked(
        base: *const T,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        <A as BackendKernel<T>>::gather_masked(base, indices, mask, src)
    }

    unsafe fn scatter(base: *mut T, indices: Self::IndexVector, val: Self::Vector) {
        <A as BackendKernel<T>>::scatter(base, indices, val);
    }

    unsafe fn scatter_masked(
        base: *mut T,
        indices: Self::IndexVector,
        mask: Self::Mask,
        val: Self::Vector,
    ) {
        <A as BackendKernel<T>>::scatter_masked(base, indices, mask, val);
    }
}
