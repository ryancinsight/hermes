//! Monomorphized SIMD vector register wrapper.
//!
//! # Safety
//!
//! Every operation here ultimately calls a `#[target_feature]`-gated
//! [`SimdKernel`](crate::kernel::SimdKernel) method, sound only on a host that
//! implements `Arch`. Two disciplines discharge that obligation:
//!
//! - **Safe methods** call `assert_runtime_supported` (or
//!   `runtime_support_result` for the `try_*` forms) before their `unsafe`
//!   kernel call, so the check immediately above each block is its target-feature
//!   proof. Those blocks therefore carry a per-site `SAFETY` comment only when
//!   they add a further obligation — a raw-pointer bound, a lane-index range, or
//!   a `MaybeUninit` initialization.
//! - The `pub unsafe fn` register loads/stores (`load_aligned` and friends) push
//!   *both* the target-feature requirement and pointer validity to the caller;
//!   each states both in its `# Safety` section.
//!
//! Lane-count and lane-index preconditions (`from_array`, `extract`, `cast`, …)
//! are proven at compile time by the `AssertLaneCount`/`AssertLaneIndex` const
//! guards, so a mismatch fails the build rather than reading out of bounds.

use super::mask_reg::Mask;
use super::SimdError;
use crate::arch::SimdArch;
use crate::kernel::{SimdKernel, MAX_SIMD_LANES};
use crate::mask::BitMask;
use crate::scalar::{CastFrom, Scalar};
use core::marker::PhantomData;

/// A monomorphized vector register type wrapping the architecture-native raw register.
#[repr(transparent)]
pub struct Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// The underlying raw vector register.
    pub raw: Arch::Vector,
    _marker: PhantomData<T>,
}

impl<T, Arch> Clone for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, Arch> Copy for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
}

impl<T, Arch> core::fmt::Debug for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        assert_runtime_supported::<T, Arch>();
        const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
        let lane_count = Arch::LANE_COUNT;
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: target feature checked above. The store writes exactly
        // `lane_count` elements into the `MAX_SIMD_LANES`-slot buffer (bounded by
        // `LANE_BOUND_CHECK`), so the `lane_count`-length slice reads only
        // initialized elements.
        unsafe {
            Arch::store_unaligned(buf.as_mut_ptr().cast::<T>(), self.raw);
            let init_slice = core::slice::from_raw_parts(buf.as_ptr().cast::<T>(), lane_count);
            f.debug_list().entries(init_slice).finish()
        }
    }
}

impl<T, Arch> PartialEq for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        assert_runtime_supported::<T, Arch>();
        const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
        let lane_count = Arch::LANE_COUNT;
        let mut buf_self = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut buf_other = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: target feature checked above. Each store writes `lane_count`
        // elements into its buffer, so both `lane_count`-length slices read only
        // initialized elements.
        unsafe {
            Arch::store_unaligned(buf_self.as_mut_ptr().cast::<T>(), self.raw);
            Arch::store_unaligned(buf_other.as_mut_ptr().cast::<T>(), other.raw);
            let slice_self = core::slice::from_raw_parts(buf_self.as_ptr().cast::<T>(), lane_count);
            let slice_other =
                core::slice::from_raw_parts(buf_other.as_ptr().cast::<T>(), lane_count);
            slice_self == slice_other
        }
    }
}

impl<T, Arch> Eq for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar + Eq,
{
}

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Create a new Vector wrapping a raw vector register.
    #[inline(always)]
    pub const fn new(raw: Arch::Vector) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Construct a Vector with all lanes set to zero.
    ///
    /// # Panics
    ///
    /// Panics if the architecture is not supported or enabled on this host.
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self::try_zero().expect("SIMD target is not supported or enabled on this host")
    }

    /// Try to construct a Vector with all lanes set to zero.
    ///
    /// # Errors
    /// Returns [`SimdError::UnsupportedTarget`] when the architecture is not
    /// supported or enabled on this host.
    #[inline(always)]
    pub fn try_zero() -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        Ok(Self::new(unsafe { Arch::zero() }))
    }

    /// Construct a Vector by broadcasting a scalar value to all lanes.
    ///
    /// # Panics
    ///
    /// Panics if the architecture is not supported or enabled on this host.
    #[inline(always)]
    pub fn splat(val: T) -> Self {
        Self::try_splat(val).expect("SIMD target is not supported or enabled on this host")
    }

    /// Try to construct a Vector by broadcasting a scalar value to all lanes.
    ///
    /// # Errors
    /// Returns [`SimdError::UnsupportedTarget`] when the architecture is not
    /// supported or enabled on this host.
    #[inline(always)]
    pub fn try_splat(val: T) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        Ok(Self::new(unsafe { Arch::splat(val) }))
    }

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
        runtime_support_result::<T, Arch>()?;
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
        runtime_support_result::<T, Arch>()?;
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
    /// outside `data`, or [`SimdError::UnsupportedTarget`] when the
    /// architecture is not supported or enabled on this host.
    #[inline]
    pub fn masked_load_from_slice(
        data: &[T],
        mask: Mask<T, Arch>,
        src: Self,
    ) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        let len = data.len();
        let bm = unsafe { mask.to_bitmask().0 };
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
            // Short slice path to prevent page faults: copy to a stack-aligned
            // `MAX_SIMD_LANES`-lane buffer.
            // The buffer holds `LANE_COUNT` lanes; `LANE_BOUND_CHECK` proves
            // `LANE_COUNT <= MAX_SIMD_LANES` at compile time per backend.
            const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
            #[repr(C, align(64))]
            struct AlignedBuf<T>([core::mem::MaybeUninit<T>; MAX_SIMD_LANES]);

            let mut buf = AlignedBuf([core::mem::MaybeUninit::uninit(); MAX_SIMD_LANES]);
            for i in 0..len {
                buf.0[i].write(data[i]);
            }
            for i in len..Arch::LANE_COUNT {
                buf.0[i].write(T::ZERO);
            }

            unsafe {
                Ok(Self::masked_load_unaligned(
                    buf.0.as_ptr().cast::<T>(),
                    mask,
                    src,
                ))
            }
        }
    }

    /// Safe masked store to a slice.
    ///
    /// Active lanes (according to `mask`) must reside within the bounds of `data`.
    /// Inactive lanes in the slice are left unchanged.
    ///
    /// # Errors
    /// Returns [`SimdError::IndexOutOfBounds`] when an active mask lane is
    /// outside `data`, or [`SimdError::UnsupportedTarget`] when the
    /// architecture is not supported or enabled on this host.
    #[inline]
    pub fn masked_store_to_slice(
        self,
        data: &mut [T],
        mask: Mask<T, Arch>,
    ) -> Result<(), SimdError> {
        runtime_support_result::<T, Arch>()?;
        let len = data.len();
        let bm = unsafe { mask.to_bitmask().0 };
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
            // Short slice path to prevent page faults: copy to stack-aligned buffer, perform masked store,
            // then copy active elements back.
            // The buffer holds `LANE_COUNT` lanes; `LANE_BOUND_CHECK` proves
            // `LANE_COUNT <= MAX_SIMD_LANES` at compile time per backend.
            const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
            #[repr(C, align(64))]
            struct AlignedBuf<T>([core::mem::MaybeUninit<T>; MAX_SIMD_LANES]);

            let mut buf = AlignedBuf([core::mem::MaybeUninit::uninit(); MAX_SIMD_LANES]);
            for i in 0..len {
                buf.0[i].write(data[i]);
            }

            unsafe {
                self.masked_store_unaligned(buf.0.as_mut_ptr().cast::<T>(), mask);
            }

            unsafe {
                let init_slice = core::slice::from_raw_parts(buf.0.as_ptr().cast::<T>(), len);
                data.copy_from_slice(init_slice);
            }
        }
        Ok(())
    }

    /// Horizontal sum reduction of all lanes in the Vector.
    #[inline(always)]
    pub fn sum_reduce(self) -> T {
        assert_runtime_supported::<T, Arch>();
        unsafe { Arch::sum_reduce(self.raw) }
    }

    /// Elementwise population count (number of set bits).
    #[inline(always)]
    #[must_use]
    pub fn popcount(self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::popcount(self.raw) })
    }

    /// Horizontal bitwise AND reduction across all lanes.
    #[inline(always)]
    pub fn horizontal_bitwise_and(self) -> T {
        assert_runtime_supported::<T, Arch>();
        unsafe { Arch::horizontal_bitwise_and(self.raw) }
    }

    /// Horizontal bitwise OR reduction across all lanes.
    #[inline(always)]
    pub fn horizontal_bitwise_or(self) -> T {
        assert_runtime_supported::<T, Arch>();
        unsafe { Arch::horizontal_bitwise_or(self.raw) }
    }

    /// Horizontal bitwise XOR reduction across all lanes.
    #[inline(always)]
    pub fn horizontal_bitwise_xor(self) -> T {
        assert_runtime_supported::<T, Arch>();
        unsafe { Arch::horizontal_bitwise_xor(self.raw) }
    }

    /// Elementwise absolute value.
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::abs(self.raw) })
    }

    /// Elementwise minimum of `self` and `other`.
    #[inline(always)]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::min(self.raw, other.raw) })
    }

    /// Elementwise maximum of `self` and `other`.
    #[inline(always)]
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::max(self.raw, other.raw) })
    }

    /// Elementwise square root.
    #[inline(always)]
    #[must_use]
    pub fn sqrt(self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::sqrt(self.raw) })
    }

    /// Elementwise equal comparison (`self == other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_eq(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::cmp_eq(self.raw, other.raw) })
    }

    /// Elementwise not-equal comparison (`self != other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_ne(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::cmp_ne(self.raw, other.raw) })
    }

    /// Elementwise less-than comparison (`self < other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_lt(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::cmp_lt(self.raw, other.raw) })
    }

    /// Elementwise less-than-or-equal comparison (`self <= other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_le(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::cmp_le(self.raw, other.raw) })
    }

    /// Elementwise greater-than comparison (`self > other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_gt(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::cmp_gt(self.raw, other.raw) })
    }

    /// Elementwise greater-than-or-equal comparison (`self >= other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_ge(self, other: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::cmp_ge(self.raw, other.raw) })
    }

    /// Conditional blend: select lanes from `true_val` where the mask lane in `self` is active (sign bit set), and from `false_val` otherwise.
    #[inline(always)]
    #[must_use]
    pub fn blend(self, true_val: Self, false_val: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::blend(self.raw, true_val.raw, false_val.raw) })
    }

    /// Create a Vector from an array of size `N`, where `N` must equal `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn from_array<const N: usize>(arr: [T; N]) -> Self {
        assert_runtime_supported::<T, Arch>();
        let _ = AssertLaneCount::<T, Arch, N>::OK;
        // SAFETY: target feature checked above; `AssertLaneCount` proved
        // `N == LANE_COUNT`, so `arr` holds a full vector's worth of elements for
        // the unaligned load.
        unsafe { Self::load_unaligned(arr.as_ptr()) }
    }

    /// Try to create a Vector from an array of size `N`, where `N` must equal
    /// `Arch::LANE_COUNT`.
    ///
    /// # Errors
    /// Returns [`SimdError::UnsupportedTarget`] when the architecture is not
    /// supported or enabled on this host.
    #[inline(always)]
    pub fn try_from_array<const N: usize>(arr: [T; N]) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        let _ = AssertLaneCount::<T, Arch, N>::OK;
        // SAFETY: as `from_array` — `N == LANE_COUNT`, so `arr` covers the load.
        unsafe { Ok(Self::load_unaligned(arr.as_ptr())) }
    }

    /// Convert the vector to an array of size `N`, where `N` must equal `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn to_array<const N: usize>(self) -> [T; N] {
        assert_runtime_supported::<T, Arch>();
        let _ = AssertLaneCount::<T, Arch, N>::OK;
        let mut arr = [core::mem::MaybeUninit::<T>::uninit(); N];
        // SAFETY: target feature checked above; `AssertLaneCount` proved
        // `N == LANE_COUNT`, so the store initializes all `N` slots before the
        // `[T; N]` is read out.
        unsafe {
            self.store_unaligned(arr.as_mut_ptr().cast::<T>());
            core::ptr::read(arr.as_ptr().cast::<[T; N]>())
        }
    }

    /// Convert this vector mask representation (sign bits) into a portable `BitMask`.
    #[inline(always)]
    pub fn to_bitmask(self) -> BitMask<64> {
        assert_runtime_supported::<T, Arch>();
        const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let lanes = <Arch as SimdKernel<T>>::LANE_COUNT;
        // SAFETY: target feature checked above; the store writes `lanes` elements
        // into the `MAX_SIMD_LANES`-slot buffer (bounded by `LANE_BOUND_CHECK`),
        // so `assume_init` reads only those initialized lanes.
        unsafe {
            self.store_unaligned(buf.as_mut_ptr().cast::<T>());
            let mut m = 0u64;
            for i in 0..lanes {
                let val = buf[i].assume_init();
                if val.to_f64() != 0.0 || val.is_nan() {
                    m |= 1u64 << i;
                }
            }
            BitMask(m)
        }
    }

    /// Elementwise equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_eq_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: `from_bitmask` requires only that the host support `Arch`,
        // which the inner `cmp_eq`/`to_bitmask` calls already assert.
        unsafe { Mask::from_bitmask(self.cmp_eq(other).to_bitmask()) }
    }

    /// Elementwise not-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_ne_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: `from_bitmask` requires only that the host support `Arch`,
        // which the inner `cmp_ne`/`to_bitmask` calls already assert.
        unsafe { Mask::from_bitmask(self.cmp_ne(other).to_bitmask()) }
    }

    /// Elementwise less-than comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_lt_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: `from_bitmask` requires only that the host support `Arch`,
        // which the inner `cmp_lt`/`to_bitmask` calls already assert.
        unsafe { Mask::from_bitmask(self.cmp_lt(other).to_bitmask()) }
    }

    /// Elementwise less-than-or-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_le_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: `from_bitmask` requires only that the host support `Arch`,
        // which the inner `cmp_le`/`to_bitmask` calls already assert.
        unsafe { Mask::from_bitmask(self.cmp_le(other).to_bitmask()) }
    }

    /// Elementwise greater-than comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_gt_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: `from_bitmask` requires only that the host support `Arch`,
        // which the inner `cmp_gt`/`to_bitmask` calls already assert.
        unsafe { Mask::from_bitmask(self.cmp_gt(other).to_bitmask()) }
    }

    /// Elementwise greater-than-or-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_ge_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: `from_bitmask` requires only that the host support `Arch`,
        // which the inner `cmp_ge`/`to_bitmask` calls already assert.
        unsafe { Mask::from_bitmask(self.cmp_ge(other).to_bitmask()) }
    }

    /// Cast the vector elements to another scalar type `U` where the lane counts match.
    #[inline(always)]
    pub fn cast<U>(self) -> Vector<U, Arch>
    where
        Arch: SimdKernel<U>,
        U: Scalar,
        U: CastFrom<T>,
    {
        assert_runtime_supported::<T, Arch>();
        assert_runtime_supported::<U, Arch>();
        let _ = AssertLaneCountSame::<T, U, Arch>::OK;
        const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
        let mut buf_t = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut buf_u = [core::mem::MaybeUninit::<U>::uninit(); MAX_SIMD_LANES];
        let lanes = <Arch as SimdKernel<T>>::LANE_COUNT;
        // SAFETY: target features for both `T` and `U` checked above;
        // `AssertLaneCountSame` and `LANE_BOUND_CHECK` bound `lanes` within both
        // buffers. The `T` store initializes `buf_t[..lanes]` before `assume_init`
        // reads it, the loop initializes `buf_u[..lanes]`, and the `U` load reads
        // exactly those `lanes` lanes.
        unsafe {
            self.store_unaligned(buf_t.as_mut_ptr().cast::<T>());
            for i in 0..lanes {
                let val_t = buf_t[i].assume_init();
                buf_u[i].write(U::cast_from(val_t));
            }
            Vector::<U, Arch>::new(Arch::load_unaligned(buf_u.as_ptr().cast::<U>()))
        }
    }

    /// Extract a single lane element by index at compile-time.
    #[inline(always)]
    pub fn extract<const I: usize>(self) -> T {
        assert_runtime_supported::<T, Arch>();
        let _ = AssertLaneIndex::<T, Arch, I>::OK;
        const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: target feature checked above; `AssertLaneIndex` proved
        // `I < LANE_COUNT`, and the store initializes `buf[..LANE_COUNT]`, so
        // `buf[I]` is initialized.
        unsafe {
            self.store_unaligned(buf.as_mut_ptr().cast::<T>());
            buf[I].assume_init()
        }
    }

    /// Insert a value into a single lane by index at compile-time.
    #[inline(always)]
    #[must_use]
    pub fn insert<const I: usize>(self, val: T) -> Self {
        assert_runtime_supported::<T, Arch>();
        let _ = AssertLaneIndex::<T, Arch, I>::OK;
        const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        // SAFETY: target feature checked above; `AssertLaneIndex` proved
        // `I < LANE_COUNT`. The store initializes `buf[..LANE_COUNT]`, `buf[I]` is
        // then overwritten, and the reload reads all `LANE_COUNT` initialized
        // lanes.
        unsafe {
            self.store_unaligned(buf.as_mut_ptr().cast::<T>());
            buf[I].write(val);
            Self::load_unaligned(buf.as_ptr().cast::<T>())
        }
    }

    /// Load a Vector from a chunk index of a `SimdView`.
    ///
    /// # Panics
    ///
    /// Panics if the architecture is not supported or enabled on this host,
    /// or if `chunk_idx` does not identify a complete SIMD lane group.
    #[inline(always)]
    #[must_use]
    pub fn from_view_chunk<Align, Mode, Ref>(
        view: &super::SimdView<'_, T, Arch, Align, Mode, Ref>,
        chunk_idx: usize,
    ) -> Self
    where
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
        Ref: core::ops::Deref<Target = [T]>,
    {
        assert_runtime_supported::<T, Arch>();
        let offset = chunk_idx * Arch::LANE_COUNT;
        let slice = view.as_slice();
        assert!(
            offset + Arch::LANE_COUNT <= slice.len(),
            "Chunk index out of bounds"
        );
        // SAFETY: target feature checked above; the assert guarantees
        // `offset + LANE_COUNT <= slice.len()`, so the load reads a full vector in
        // bounds. The aligned variant is taken only when `Align` proves the base
        // pointer is arch-aligned and `offset` is a lane-count multiple.
        unsafe {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                Self::load_aligned(slice.as_ptr().add(offset))
            } else {
                Self::load_unaligned(slice.as_ptr().add(offset))
            }
        }
    }

    /// Store this Vector into a mutable chunk of a mutable `SimdView`.
    ///
    /// # Panics
    ///
    /// Panics if the architecture is not supported or enabled on this host,
    /// or if `chunk_idx` does not identify a complete SIMD lane group.
    #[inline(always)]
    pub fn store_to_view_chunk<'a, Align, Mode>(
        self,
        view: &mut super::SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>,
        chunk_idx: usize,
    ) where
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
    {
        assert_runtime_supported::<T, Arch>();
        let offset = chunk_idx * Arch::LANE_COUNT;
        let slice = view.as_slice_mut();
        assert!(
            offset + Arch::LANE_COUNT <= slice.len(),
            "Chunk index out of bounds"
        );
        // SAFETY: as `from_view_chunk` — the assert guarantees
        // `offset + LANE_COUNT <= slice.len()`, so the store writes a full vector
        // in bounds; the aligned variant is gated on `Align`.
        unsafe {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                self.store_aligned(slice.as_mut_ptr().add(offset));
            } else {
                self.store_unaligned(slice.as_mut_ptr().add(offset));
            }
        }
    }
}

#[inline(always)]
fn is_vector_aligned<T, Arch>(ptr: *const T) -> bool
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    let alignment = Arch::LANE_COUNT * core::mem::size_of::<T>();
    alignment != 0 && (ptr as usize).is_multiple_of(alignment)
}

#[inline(always)]
pub(crate) fn runtime_support_result<T, Arch>() -> Result<(), SimdError>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    if Arch::is_runtime_supported() {
        Ok(())
    } else {
        Err(SimdError::UnsupportedTarget)
    }
}

#[inline(always)]
pub(crate) fn assert_runtime_supported<T, Arch>()
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    assert!(
        Arch::is_runtime_supported(),
        "SIMD target is not supported or enabled on this host"
    );
}

struct AssertLaneIndex<T, Arch, const I: usize>(PhantomData<(T, Arch)>);
impl<T, Arch, const I: usize> AssertLaneIndex<T, Arch, I>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    const OK: () = {
        assert!(
            I < <Arch as SimdKernel<T>>::LANE_COUNT,
            "Lane index out of bounds"
        );
    };
}

struct AssertLaneCountSame<T, U, Arch>(PhantomData<(T, U, Arch)>);
impl<T, U, Arch> AssertLaneCountSame<T, U, Arch>
where
    Arch: SimdArch + SimdKernel<T> + SimdKernel<U>,
    T: Scalar,
    U: Scalar,
{
    const OK: () = {
        assert!(
            <Arch as SimdKernel<T>>::LANE_COUNT == <Arch as SimdKernel<U>>::LANE_COUNT,
            "Source and destination vectors must have the same lane count"
        );
    };
}

struct AssertLaneCount<T, Arch, const N: usize>(PhantomData<(T, Arch)>);
impl<T, Arch, const N: usize> AssertLaneCount<T, Arch, N>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    const OK: () = {
        assert!(
            N == Arch::LANE_COUNT,
            "Array size must match Vector lane count"
        );
    };
}
