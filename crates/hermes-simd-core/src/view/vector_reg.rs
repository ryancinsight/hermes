//! Monomorphized SIMD vector register wrapper.

use super::mask_reg::Mask;
use super::SimdError;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
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
        let lane_count = Arch::LANE_COUNT;
        assert!(
            lane_count <= 128,
            "LANE_COUNT exceeds maximum debug buffer size of 128"
        );
        let mut buf = [unsafe { core::mem::zeroed::<T>() }; 128];
        unsafe {
            Arch::store_unaligned(buf.as_mut_ptr(), self.raw);
        }
        f.debug_list().entries(&buf[..lane_count]).finish()
    }
}

impl<T, Arch> PartialEq for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar + PartialEq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let lane_count = Arch::LANE_COUNT;
        assert!(lane_count <= 128);
        let mut buf_self = [unsafe { core::mem::zeroed::<T>() }; 128];
        let mut buf_other = [unsafe { core::mem::zeroed::<T>() }; 128];
        unsafe {
            Arch::store_unaligned(buf_self.as_mut_ptr(), self.raw);
            Arch::store_unaligned(buf_other.as_mut_ptr(), other.raw);
        }
        buf_self[..lane_count] == buf_other[..lane_count]
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
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(unsafe { Arch::zero() })
    }

    /// Construct a Vector by broadcasting a scalar value to all lanes.
    #[inline(always)]
    pub fn splat(val: T) -> Self {
        Self::new(unsafe { Arch::splat(val) })
    }

    /// Load a Vector from an aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for reads and aligned to `Arch::LANE_COUNT * size_of::<T>()` bytes.
    #[inline(always)]
    pub unsafe fn load_aligned(ptr: *const T) -> Self {
        Self::new(Arch::load_aligned(ptr))
    }

    /// Load a Vector from an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for reads.
    #[inline(always)]
    pub unsafe fn load_unaligned(ptr: *const T) -> Self {
        Self::new(Arch::load_unaligned(ptr))
    }

    /// Store the Vector elements to an aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for writes and aligned to `Arch::LANE_COUNT * size_of::<T>()` bytes.
    #[inline(always)]
    pub unsafe fn store_aligned(self, ptr: *mut T) {
        Arch::store_aligned(ptr, self.raw);
    }

    /// Store the Vector elements to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for writes.
    #[inline(always)]
    pub unsafe fn store_unaligned(self, ptr: *mut T) {
        Arch::store_unaligned(ptr, self.raw);
    }

    /// Load one vector from the start of a slice using the unaligned kernel load.
    ///
    /// Returns [`SimdError::InsufficientInputLength`] when `data` has fewer
    /// elements than `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn load_unaligned_from_slice(data: &[T]) -> Result<Self, SimdError> {
        if data.len() < Arch::LANE_COUNT {
            return Err(SimdError::InsufficientInputLength);
        }
        // SAFETY: length was checked for one complete vector; unaligned load
        // has no alignment precondition.
        unsafe { Ok(Self::load_unaligned(data.as_ptr())) }
    }

    /// Load one vector from the start of a slice using the aligned kernel load.
    ///
    /// Returns [`SimdError::InsufficientInputLength`] when `data` has fewer
    /// elements than `Arch::LANE_COUNT`, and [`SimdError::UnalignedAddress`]
    /// when the slice start is not aligned to the vector byte width.
    #[inline(always)]
    pub fn load_aligned_from_slice(data: &[T]) -> Result<Self, SimdError> {
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

    /// Horizontal sum reduction of all lanes in the Vector.
    #[inline(always)]
    pub fn sum_reduce(self) -> T {
        unsafe { Arch::sum_reduce(self.raw) }
    }

    /// Elementwise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(unsafe { Arch::abs(self.raw) })
    }

    /// Elementwise minimum of `self` and `other`.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::new(unsafe { Arch::min(self.raw, other.raw) })
    }

    /// Elementwise maximum of `self` and `other`.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self::new(unsafe { Arch::max(self.raw, other.raw) })
    }

    /// Elementwise square root.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self::new(unsafe { Arch::sqrt(self.raw) })
    }

    /// Elementwise equal comparison (`self == other`).
    #[inline(always)]
    pub fn cmp_eq(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_eq(self.raw, other.raw) })
    }

    /// Elementwise not-equal comparison (`self != other`).
    #[inline(always)]
    pub fn cmp_ne(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_ne(self.raw, other.raw) })
    }

    /// Elementwise less-than comparison (`self < other`).
    #[inline(always)]
    pub fn cmp_lt(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_lt(self.raw, other.raw) })
    }

    /// Elementwise less-than-or-equal comparison (`self <= other`).
    #[inline(always)]
    pub fn cmp_le(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_le(self.raw, other.raw) })
    }

    /// Elementwise greater-than comparison (`self > other`).
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_gt(self.raw, other.raw) })
    }

    /// Elementwise greater-than-or-equal comparison (`self >= other`).
    #[inline(always)]
    pub fn cmp_ge(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_ge(self.raw, other.raw) })
    }

    /// Conditional blend: select lanes from `true_val` where the mask lane in `self` is active (sign bit set), and from `false_val` otherwise.
    #[inline(always)]
    pub fn blend(self, true_val: Self, false_val: Self) -> Self {
        Self::new(unsafe { Arch::blend(self.raw, true_val.raw, false_val.raw) })
    }

    /// Create a Vector from an array of size `N`, where `N` must equal `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn from_array<const N: usize>(arr: [T; N]) -> Self {
        let _ = AssertLaneCount::<T, Arch, N>::OK;
        unsafe { Self::load_unaligned(arr.as_ptr()) }
    }

    /// Convert the vector to an array of size `N`, where `N` must equal `Arch::LANE_COUNT`.
    #[inline(always)]
    pub fn to_array<const N: usize>(self) -> [T; N] {
        let _ = AssertLaneCount::<T, Arch, N>::OK;
        let mut arr = [T::ZERO; N];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        arr
    }

    /// Convert this vector mask representation (sign bits) into a portable `BitMask`.
    #[inline(always)]
    pub fn to_bitmask(self) -> BitMask<64> {
        let mut buf = [T::ZERO; 128];
        let lanes = <Arch as SimdKernel<T>>::LANE_COUNT;
        unsafe {
            self.store_unaligned(buf.as_mut_ptr());
        }
        let mut m = 0u64;
        for i in 0..lanes {
            if buf[i].to_f64() != 0.0 || buf[i].is_nan() {
                m |= 1u64 << i;
            }
        }
        BitMask(m)
    }

    /// Elementwise equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_eq_mask(self, other: Self) -> Mask<T, Arch> {
        unsafe { Mask::from_bitmask(self.cmp_eq(other).to_bitmask()) }
    }

    /// Elementwise not-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_ne_mask(self, other: Self) -> Mask<T, Arch> {
        unsafe { Mask::from_bitmask(self.cmp_ne(other).to_bitmask()) }
    }

    /// Elementwise less-than comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_lt_mask(self, other: Self) -> Mask<T, Arch> {
        unsafe { Mask::from_bitmask(self.cmp_lt(other).to_bitmask()) }
    }

    /// Elementwise less-than-or-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_le_mask(self, other: Self) -> Mask<T, Arch> {
        unsafe { Mask::from_bitmask(self.cmp_le(other).to_bitmask()) }
    }

    /// Elementwise greater-than comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_gt_mask(self, other: Self) -> Mask<T, Arch> {
        unsafe { Mask::from_bitmask(self.cmp_gt(other).to_bitmask()) }
    }

    /// Elementwise greater-than-or-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_ge_mask(self, other: Self) -> Mask<T, Arch> {
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
        let _ = AssertLaneCountSame::<T, U, Arch>::OK;
        let mut buf_t = [T::ZERO; 128];
        let mut buf_u = [U::ZERO; 128];
        let lanes = <Arch as SimdKernel<T>>::LANE_COUNT;
        unsafe {
            self.store_unaligned(buf_t.as_mut_ptr());
            for i in 0..lanes {
                buf_u[i] = U::cast_from(buf_t[i]);
            }
            Vector::<U, Arch>::new(Arch::load_unaligned(buf_u.as_ptr()))
        }
    }

    /// Extract a single lane element by index at compile-time.
    #[inline(always)]
    pub fn extract<const I: usize>(self) -> T {
        let _ = AssertLaneIndex::<T, Arch, I>::OK;
        let mut buf = [T::ZERO; 128];
        unsafe {
            self.store_unaligned(buf.as_mut_ptr());
        }
        buf[I]
    }

    /// Insert a value into a single lane by index at compile-time.
    #[inline(always)]
    pub fn insert<const I: usize>(self, val: T) -> Self {
        let _ = AssertLaneIndex::<T, Arch, I>::OK;
        let mut buf = [T::ZERO; 128];
        unsafe {
            self.store_unaligned(buf.as_mut_ptr());
            buf[I] = val;
            Self::load_unaligned(buf.as_ptr())
        }
    }

    /// Load a Vector from a chunk index of a `SimdView`.
    #[inline(always)]
    pub fn from_view_chunk<Align, Mode, Ref>(
        view: &super::SimdView<'_, T, Arch, Align, Mode, Ref>,
        chunk_idx: usize,
    ) -> Self
    where
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
        Ref: core::ops::Deref<Target = [T]>,
    {
        let offset = chunk_idx * Arch::LANE_COUNT;
        let slice = view.as_slice();
        assert!(
            offset + Arch::LANE_COUNT <= slice.len(),
            "Chunk index out of bounds"
        );
        unsafe {
            if Align::IS_ALIGNED {
                Self::load_aligned(slice.as_ptr().add(offset))
            } else {
                Self::load_unaligned(slice.as_ptr().add(offset))
            }
        }
    }

    /// Store this Vector into a mutable chunk of a mutable `SimdView`.
    #[inline(always)]
    pub fn store_to_view_chunk<'a, Align, Mode>(
        self,
        view: &mut super::SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>,
        chunk_idx: usize,
    ) where
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
    {
        let offset = chunk_idx * Arch::LANE_COUNT;
        let slice = view.as_slice_mut();
        assert!(
            offset + Arch::LANE_COUNT <= slice.len(),
            "Chunk index out of bounds"
        );
        unsafe {
            if Align::IS_ALIGNED {
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
