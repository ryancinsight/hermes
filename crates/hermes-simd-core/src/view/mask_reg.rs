//! Monomorphized SIMD mask register wrapper.

use super::vector_reg::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::mask::BitMask;
use crate::scalar::Scalar;
use core::marker::PhantomData;

/// A type-safe, architecture-native SIMD mask type.
#[repr(transparent)]
pub struct Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// The underlying raw mask register or representation.
    pub raw: Arch::Mask,
    _marker: PhantomData<T>,
}

impl<T, Arch> Clone for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, Arch> Copy for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
}

impl<T, Arch> core::fmt::Debug for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bm = unsafe { self.to_bitmask() };
        f.debug_tuple("Mask").field(&bm.to_bools()).finish()
    }
}

impl<T, Arch> PartialEq for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.to_bitmask() == other.to_bitmask() }
    }
}

impl<T, Arch> Eq for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
}

impl<T, Arch> Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Create a new Mask wrapping a raw mask register.
    #[inline(always)]
    pub const fn new(raw: Arch::Mask) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Construct a Mask from a portable `BitMask<64>`.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    pub unsafe fn from_bitmask(bm: BitMask<64>) -> Self {
        Self::new(Arch::mask_from_bitmask(bm.0))
    }

    /// Convert the Mask to a portable `BitMask<64>`.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    pub unsafe fn to_bitmask(self) -> BitMask<64> {
        BitMask(Arch::mask_to_bitmask(self.raw))
    }

    /// Returns `true` if any lanes of the mask are active.
    #[inline(always)]
    pub fn any(self) -> bool {
        unsafe { !self.to_bitmask().is_none_active() }
    }

    /// Returns `true` if all lanes of the mask are active.
    #[inline(always)]
    pub fn all(self) -> bool {
        let lanes = <Arch as SimdKernel<T>>::LANE_COUNT;
        let expected = if lanes >= 64 {
            u64::MAX
        } else {
            (1u64 << lanes) - 1
        };
        unsafe { (self.to_bitmask().0 & expected) == expected }
    }

    /// Returns `true` if no lanes of the mask are active.
    #[inline(always)]
    pub fn none(self) -> bool {
        unsafe { self.to_bitmask().is_none_active() }
    }

    /// Select elements from `true_val` where the mask is active, and from `false_val` otherwise.
    #[inline(always)]
    pub fn select(self, true_val: Vector<T, Arch>, false_val: Vector<T, Arch>) -> Vector<T, Arch> {
        let zero = Vector::<T, Arch>::zero();
        Vector::new(unsafe { Arch::masked_add(true_val.raw, zero.raw, self.raw, false_val.raw) })
    }
}

// Bitwise operations on Mask
impl<T, Arch> core::ops::BitAnd for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        unsafe {
            let bm_self = self.to_bitmask();
            let bm_rhs = rhs.to_bitmask();
            Self::from_bitmask(bm_self & bm_rhs)
        }
    }
}

impl<T, Arch> core::ops::BitOr for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        unsafe {
            let bm_self = self.to_bitmask();
            let bm_rhs = rhs.to_bitmask();
            Self::from_bitmask(bm_self | bm_rhs)
        }
    }
}

impl<T, Arch> core::ops::BitXor for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        unsafe {
            let bm_self = self.to_bitmask();
            let bm_rhs = rhs.to_bitmask();
            Self::from_bitmask(BitMask(bm_self.0 ^ bm_rhs.0))
        }
    }
}

impl<T, Arch> core::ops::Not for Mask<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        unsafe {
            let bm = self.to_bitmask();
            let lanes = <Arch as SimdKernel<T>>::LANE_COUNT;
            let active_mask = if lanes >= 64 {
                u64::MAX
            } else {
                (1u64 << lanes) - 1
            };
            Self::from_bitmask(BitMask((!bm.0) & active_mask))
        }
    }
}
