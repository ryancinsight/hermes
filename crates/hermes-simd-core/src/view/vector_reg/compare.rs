//! Lane-wise comparison, blending and bitmask extraction.

use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::mask::BitMask;
use crate::scalar::Scalar;
use crate::view::mask_reg::Mask;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Elementwise equal comparison (`self == other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_eq(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_eq(self.raw, other.raw) })
    }

    /// Elementwise not-equal comparison (`self != other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_ne(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_ne(self.raw, other.raw) })
    }

    /// Elementwise less-than comparison (`self < other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_lt(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_lt(self.raw, other.raw) })
    }

    /// Elementwise less-than-or-equal comparison (`self <= other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_le(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_le(self.raw, other.raw) })
    }

    /// Elementwise greater-than comparison (`self > other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_gt(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_gt(self.raw, other.raw) })
    }

    /// Elementwise greater-than-or-equal comparison (`self >= other`).
    #[inline(always)]
    #[must_use]
    pub fn cmp_ge(self, other: Self) -> Self {
        Self::new(unsafe { Arch::cmp_ge(self.raw, other.raw) })
    }

    /// Conditional blend: select lanes from `true_val` where the mask lane in `self` is active (sign bit set), and from `false_val` otherwise.
    #[inline(always)]
    #[must_use]
    pub fn blend(self, true_val: Self, false_val: Self) -> Self {
        Self::new(unsafe { Arch::blend(self.raw, true_val.raw, false_val.raw) })
    }

    /// Convert this vector mask representation (sign bits) into a portable `BitMask`.
    #[inline(always)]
    pub fn to_bitmask(self) -> BitMask<64> {
        // SAFETY: constructing `self` proved host support for `Arch`. The
        // `vector_to_mask` contract defines a lane as active iff its sign bit
        // is set, for arbitrary (non-canonical) lanes — tested bit-level, so
        // `+2.0` stays inactive and `-0.0` stays active, matching hardware
        // movemask semantics — and `mask_to_bitmask` emits one bit per lane.
        // The composition therefore reports exactly the documented sign bits
        // while staying in registers (one `vmovmskps` on AVX2) instead of
        // spilling to a stack buffer for scalar extraction.
        BitMask(unsafe { Arch::mask_to_bitmask(Arch::vector_to_mask(self.raw)) })
    }

    /// Elementwise equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_eq_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: constructing `self` proved that the host supports `Arch`.
        Mask::new(unsafe { Arch::vector_to_mask(Arch::cmp_eq(self.raw, other.raw)) })
    }

    /// Elementwise not-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_ne_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: constructing `self` proved that the host supports `Arch`.
        Mask::new(unsafe { Arch::vector_to_mask(Arch::cmp_ne(self.raw, other.raw)) })
    }

    /// Elementwise less-than comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_lt_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: constructing `self` proved that the host supports `Arch`.
        Mask::new(unsafe { Arch::vector_to_mask(Arch::cmp_lt(self.raw, other.raw)) })
    }

    /// Elementwise less-than-or-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_le_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: constructing `self` proved that the host supports `Arch`.
        Mask::new(unsafe { Arch::vector_to_mask(Arch::cmp_le(self.raw, other.raw)) })
    }

    /// Elementwise greater-than comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_gt_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: constructing `self` proved that the host supports `Arch`.
        Mask::new(unsafe { Arch::vector_to_mask(Arch::cmp_gt(self.raw, other.raw)) })
    }

    /// Elementwise greater-than-or-equal comparison returning a native `Mask`.
    #[inline(always)]
    pub fn cmp_ge_mask(self, other: Self) -> Mask<T, Arch> {
        // SAFETY: constructing `self` proved that the host supports `Arch`.
        Mask::new(unsafe { Arch::vector_to_mask(Arch::cmp_ge(self.raw, other.raw)) })
    }
}
