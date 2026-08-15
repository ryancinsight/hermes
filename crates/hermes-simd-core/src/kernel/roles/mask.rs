//! Mask capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for mask construction, conversion, and compaction.
pub trait SimdMask<T: Scalar>: SimdStorage<T> + Sealed {
    /// Packs active lanes into the low lanes of a register.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector;

    /// Expands packed lanes into active positions and fills inactive positions.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector;

    /// Constructs a native mask from one boolean per lane.
    ///
    /// # Safety
    /// The backend's target features must be available and `bits` must have
    /// exactly [`SimdStorage::LANE_COUNT`] elements.
    ///
    /// # Panics
    /// Panics in debug builds if `bits.len()` differs from
    /// [`SimdStorage::LANE_COUNT`].
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask;

    /// Activates the first `k` lanes.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn leading_k_mask(k: usize) -> Self::Mask;

    /// Converts a raw bitmask to a native mask.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask;

    /// Converts a native mask to an all-ones/all-zeroes vector.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector;

    /// Converts a comparison-result vector to a native mask.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask;

    /// Converts a native mask to a raw bitmask.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64;
}

impl<T: Scalar, A: BackendKernel<T>> SimdMask<T> for A {
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        <A as BackendKernel<T>>::compress(src, mask)
    }

    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::expand(src, mask, fill)
    }

    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        <A as BackendKernel<T>>::mask_from_bools(bits)
    }

    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        <A as BackendKernel<T>>::leading_k_mask(k)
    }

    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask {
        <A as BackendKernel<T>>::mask_from_bitmask(bm)
    }

    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        <A as BackendKernel<T>>::mask_to_vector(mask)
    }

    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        <A as BackendKernel<T>>::vector_to_mask(v)
    }

    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        <A as BackendKernel<T>>::mask_to_bitmask(mask)
    }
}
