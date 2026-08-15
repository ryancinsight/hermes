//! Load and store capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for aligned, unaligned, masked, and streaming memory access.
pub trait SimdLoadStore<T: Scalar>: SimdStorage<T> + Sealed {
    /// Loads one register from an aligned pointer.
    ///
    /// # Safety
    /// The pointer must be valid and aligned for one register of `T`.
    unsafe fn load_aligned(ptr: *const T) -> Self::Vector;

    /// Loads one register from an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must be valid for one register of `T`.
    unsafe fn load_unaligned(ptr: *const T) -> Self::Vector;

    /// Stores one register to an aligned pointer.
    ///
    /// # Safety
    /// The pointer must be valid and aligned for one register of `T`.
    unsafe fn store_aligned(ptr: *mut T, val: Self::Vector);

    /// Stores one register to an unaligned pointer.
    ///
    /// # Safety
    /// The pointer must be valid for one register of `T`.
    unsafe fn store_unaligned(ptr: *mut T, val: Self::Vector);

    /// Stores one register using the backend's non-temporal path when supported.
    ///
    /// # Safety
    /// The pointer must be valid and aligned for one register of `T`.
    unsafe fn store_streaming(ptr: *mut T, val: Self::Vector);

    /// Orders preceding non-temporal stores before subsequent reads.
    fn stream_write_barrier();

    /// Loads active lanes and preserves `src` in inactive lanes.
    ///
    /// # Safety
    /// The pointer must be valid for `SimdStorage::LANE_COUNT` elements. The
    /// mask selects which lanes are loaded; inactive lanes come from `src`.
    unsafe fn masked_load_unaligned(
        ptr: *const T,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Stores active lanes and leaves inactive lanes unchanged.
    ///
    /// # Safety
    /// The pointer must be valid for `SimdStorage::LANE_COUNT` elements. Only
    /// active lanes selected by `mask` are written.
    unsafe fn masked_store_unaligned(ptr: *mut T, mask: Self::Mask, val: Self::Vector);
}

impl<T: Scalar, A: BackendKernel<T>> SimdLoadStore<T> for A {
    unsafe fn load_aligned(ptr: *const T) -> Self::Vector {
        <A as BackendKernel<T>>::load_aligned(ptr)
    }

    unsafe fn load_unaligned(ptr: *const T) -> Self::Vector {
        <A as BackendKernel<T>>::load_unaligned(ptr)
    }

    unsafe fn store_aligned(ptr: *mut T, val: Self::Vector) {
        <A as BackendKernel<T>>::store_aligned(ptr, val);
    }

    unsafe fn store_unaligned(ptr: *mut T, val: Self::Vector) {
        <A as BackendKernel<T>>::store_unaligned(ptr, val);
    }

    unsafe fn store_streaming(ptr: *mut T, val: Self::Vector) {
        <A as BackendKernel<T>>::store_streaming(ptr, val);
    }

    fn stream_write_barrier() {
        <A as BackendKernel<T>>::stream_write_barrier();
    }

    unsafe fn masked_load_unaligned(
        ptr: *const T,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        <A as BackendKernel<T>>::masked_load_unaligned(ptr, mask, src)
    }

    unsafe fn masked_store_unaligned(ptr: *mut T, mask: Self::Mask, val: Self::Vector) {
        <A as BackendKernel<T>>::masked_store_unaligned(ptr, mask, val);
    }
}
