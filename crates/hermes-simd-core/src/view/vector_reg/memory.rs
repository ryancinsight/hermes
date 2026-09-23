//! Loading from and storing to slices, aligned and unaligned, masked and plain.

use super::is_vector_aligned;
use super::runtime_support_result;
use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::view::mask_reg::Mask;
use crate::view::SimdError;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Load a Vector from an aligned pointer.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, and `ptr` must be valid
    /// for reads and aligned to `Arch::LANE_COUNT * size_of::<T>()` bytes.
    #[inline(always)]
    pub unsafe fn load_aligned(ptr: *const T) -> Self {
        Self::new(Arch::load_aligned(ptr))
    }

    /// Load a Vector from an unaligned pointer.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, and `ptr` must be valid
    /// for reads.
    #[inline(always)]
    pub unsafe fn load_unaligned(ptr: *const T) -> Self {
        Self::new(Arch::load_unaligned(ptr))
    }

    /// Store the Vector elements to an aligned pointer.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, and `ptr` must be valid
    /// for writes and aligned to `Arch::LANE_COUNT * size_of::<T>()` bytes.
    #[inline(always)]
    pub unsafe fn store_aligned(self, ptr: *mut T) {
        Arch::store_aligned(ptr, self.raw);
    }

    /// Store the Vector elements to an unaligned pointer.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, and `ptr` must be valid
    /// for writes.
    #[inline(always)]
    pub unsafe fn store_unaligned(self, ptr: *mut T) {
        Arch::store_unaligned(ptr, self.raw);
    }

    /// Masked load from an unaligned pointer: active lanes loaded from `ptr`, inactive lanes from `src`.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, and `ptr` must be valid
    /// for reads of `Arch::LANE_COUNT` elements.
    #[inline(always)]
    pub unsafe fn masked_load_unaligned(ptr: *const T, mask: Mask<T, Arch>, src: Self) -> Self {
        Self::new(Arch::masked_load_unaligned(ptr, mask.raw, src.raw))
    }

    /// Masked store to an unaligned pointer: active lanes written to `ptr`, inactive lanes left unchanged.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, and `ptr` must be valid
    /// for writes of `Arch::LANE_COUNT` elements.
    #[inline(always)]
    pub unsafe fn masked_store_unaligned(self, ptr: *mut T, mask: Mask<T, Arch>) {
        Arch::masked_store_unaligned(ptr, mask.raw, self.raw);
    }

    /// Loads active lanes from a pointer with a partial accessible prefix.
    ///
    /// Inactive lanes retain their values from `src` and are not read from
    /// memory.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, `valid_lanes` must not
    /// exceed `Arch::LANE_COUNT`, every active mask lane must be less than
    /// `valid_lanes`, and `ptr` must be valid for reading those active elements.
    #[inline(always)]
    pub unsafe fn masked_load_partial(
        ptr: *const T,
        valid_lanes: usize,
        mask: Mask<T, Arch>,
        src: Self,
    ) -> Self {
        // SAFETY: forwarded unchanged to the architecture contract.
        Self::new(unsafe { Arch::masked_load_partial(ptr, valid_lanes, mask.raw, src.raw) })
    }

    /// Stores active lanes to a pointer with a partial accessible prefix.
    ///
    /// Inactive lanes are not read or written.
    ///
    /// # Safety
    /// The host must support `Arch`'s target features, `valid_lanes` must not
    /// exceed `Arch::LANE_COUNT`, every active mask lane must be less than
    /// `valid_lanes`, and `ptr` must be valid for writing those active elements.
    #[inline(always)]
    pub unsafe fn masked_store_partial(self, ptr: *mut T, valid_lanes: usize, mask: Mask<T, Arch>) {
        // SAFETY: forwarded unchanged to the architecture contract.
        unsafe { Arch::masked_store_partial(ptr, valid_lanes, mask.raw, self.raw) };
    }

    /// Load one vector from the start of a slice using the unaligned kernel load.
    ///
    /// # Errors
    /// Returns [`SimdError::InsufficientInputLength`] when `data` has fewer
    /// elements than `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn load_unaligned_from_slice(data: &[T]) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        if data.len() < Arch::LANE_COUNT {
            return Err(SimdError::InsufficientInputLength);
        }
        // SAFETY: length was checked for one complete vector; unaligned load
        // has no alignment precondition.
        unsafe { Ok(Self::load_unaligned(data.as_ptr())) }
    }

    /// Load one vector from the start of a slice using the aligned kernel load.
    ///
    /// # Errors
    /// Returns [`SimdError::InsufficientInputLength`] when `data` has fewer
    /// elements than `Arch::LANE_COUNT`, and [`SimdError::UnalignedAddress`]
    /// when the slice start is not aligned to the vector byte width.
    #[inline(always)]
    pub fn load_aligned_from_slice(data: &[T]) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        if data.len() < Arch::LANE_COUNT {
            return Err(SimdError::InsufficientInputLength);
        }
        if !is_vector_aligned::<T, Arch>(data.as_ptr()) {
            return Err(SimdError::UnalignedAddress);
        }
        // SAFETY: length and vector-width alignment were checked above.
        unsafe { Ok(Self::load_aligned(data.as_ptr())) }
    }

    /// Store this vector to the start of a slice using the unaligned kernel store.
    ///
    /// # Errors
    /// Returns [`SimdError::InsufficientOutputLength`] when `out` has fewer
    /// elements than `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn store_unaligned_to_slice(self, out: &mut [T]) -> Result<(), SimdError> {
        if out.len() < Arch::LANE_COUNT {
            return Err(SimdError::InsufficientOutputLength);
        }
        // SAFETY: length was checked for one complete vector; unaligned store
        // has no alignment precondition.
        unsafe {
            self.store_unaligned(out.as_mut_ptr());
        }
        Ok(())
    }

    /// Store this vector to the start of a slice using the aligned kernel store.
    ///
    /// # Errors
    /// Returns [`SimdError::InsufficientOutputLength`] when `out` has fewer
    /// elements than `Arch::LANE_COUNT`, and [`SimdError::UnalignedAddress`]
    /// when the slice start is not aligned to the vector byte width.
    #[inline(always)]
    pub fn store_aligned_to_slice(self, out: &mut [T]) -> Result<(), SimdError> {
        if out.len() < Arch::LANE_COUNT {
            return Err(SimdError::InsufficientOutputLength);
        }
        if !is_vector_aligned::<T, Arch>(out.as_ptr()) {
            return Err(SimdError::UnalignedAddress);
        }
        // SAFETY: length and vector-width alignment were checked above.
        unsafe {
            self.store_aligned(out.as_mut_ptr());
        }
        Ok(())
    }

    /// Safe masked load from a slice.
    ///
    /// Active lanes (according to `mask`) must reside within the bounds of `data`.
    /// Inactive lanes are populated from the corresponding lanes of `src`.
    ///
    /// # Errors
    /// Returns [`SimdError::IndexOutOfBounds`] when an active mask lane is
    /// outside `data`.
    #[inline]
    pub fn masked_load_from_slice(
        data: &[T],
        mask: Mask<T, Arch>,
        src: Self,
    ) -> Result<Self, SimdError> {
        let len = data.len();
        let bm = mask.to_bitmask().0;
        let is_out_of_bounds = if len < u64::BITS as usize {
            (bm >> len) != 0
        } else {
            false
        };
        if is_out_of_bounds {
            return Err(SimdError::IndexOutOfBounds);
        }

        if len >= Arch::LANE_COUNT {
            // SAFETY: data has at least LANE_COUNT elements, and we verified that no active lane index
            // is beyond the slice bounds (since len >= LANE_COUNT).
            // Hence, it is safe to load directly.
            unsafe { Ok(Self::masked_load_unaligned(data.as_ptr(), mask, src)) }
        } else {
            // SAFETY: the mask was checked against the slice length, so every
            // active lane is inside the accessible prefix.
            unsafe { Ok(Self::masked_load_partial(data.as_ptr(), len, mask, src)) }
        }
    }

    /// Safe masked store to a slice.
    ///
    /// Active lanes (according to `mask`) must reside within the bounds of `data`.
    /// Inactive lanes in the slice are left unchanged.
    ///
    /// # Errors
    /// Returns [`SimdError::IndexOutOfBounds`] when an active mask lane is
    /// outside `data`.
    #[inline]
    pub fn masked_store_to_slice(
        self,
        data: &mut [T],
        mask: Mask<T, Arch>,
    ) -> Result<(), SimdError> {
        let len = data.len();
        let bm = mask.to_bitmask().0;
        let is_out_of_bounds = if len < u64::BITS as usize {
            (bm >> len) != 0
        } else {
            false
        };
        if is_out_of_bounds {
            return Err(SimdError::IndexOutOfBounds);
        }

        if len >= Arch::LANE_COUNT {
            // SAFETY: data has at least LANE_COUNT elements, and we verified that no active lane index
            // is beyond the slice bounds (since len >= LANE_COUNT).
            // Hence, it is safe to store directly.
            unsafe {
                self.masked_store_unaligned(data.as_mut_ptr(), mask);
            }
        } else {
            // SAFETY: the mask was checked against the slice length, so every
            // active lane is inside the accessible prefix.
            unsafe { self.masked_store_partial(data.as_mut_ptr(), len, mask) };
        }
        Ok(())
    }
}
