//! Elementwise arithmetic and fused multiply-add variants.

use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Elementwise absolute value.
    #[inline(always)]
    #[must_use]
    pub fn abs(self) -> Self {
        Self::new(unsafe { Arch::abs(self.raw) })
    }

    /// Elementwise minimum of `self` and `other`.
    #[inline(always)]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self::new(unsafe { Arch::min(self.raw, other.raw) })
    }

    /// Elementwise maximum of `self` and `other`.
    #[inline(always)]
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self::new(unsafe { Arch::max(self.raw, other.raw) })
    }

    /// Elementwise square root.
    #[inline(always)]
    #[must_use]
    pub fn sqrt(self) -> Self {
        Self::new(unsafe { Arch::sqrt(self.raw) })
    }

    /// Fused multiply-add: `self * b + c`, with one rounding.
    ///
    /// The single-rounding contract is the point: it is both faster and more
    /// accurate than a separate multiply and add, and an error analysis that
    /// assumes it is not satisfied by the two-operation form. Rust never
    /// contracts `a * b + c` on its own, so the fusion must be written.
    ///
    /// This completes the safe surface for the multiply-accumulate kernels that
    /// dominate transforms, stencils, and dot products; before it, a consumer
    /// needing FMA had to drop to the `unsafe` `SimdArith` facet.
    #[inline(always)]
    #[must_use]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        Self::new(unsafe { Arch::fmadd(self.raw, b.raw, c.raw) })
    }

    /// Fused multiply-subtract: `self * b - c`, with one rounding.
    ///
    /// This is the subtracting counterpart of [`Vector::mul_add`]. Backends
    /// use a native fused instruction where available and otherwise negate the
    /// addend exactly before one fused multiply-add.
    #[inline(always)]
    #[must_use]
    pub fn mul_sub(self, b: Self, c: Self) -> Self {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        Self::new(unsafe { Arch::fmsub(self.raw, b.raw, c.raw) })
    }

    /// Fused multiply then alternately subtract and add: `self * b -+ c`,
    /// subtracting on even lanes and adding on odd lanes.
    ///
    /// With [`Vector::dup_even`], [`Vector::dup_odd`], and
    /// [`Vector::swap_adjacent`], this is the complete interleaved-complex
    /// multiply: one `fmaddsub` finishes the operation that would otherwise
    /// take a separate multiply, shuffle, and sign correction.
    #[inline(always)]
    #[must_use]
    pub fn fmaddsub(self, b: Self, c: Self) -> Self {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        Self::new(unsafe { Arch::fmaddsub(self.raw, b.raw, c.raw) })
    }

    /// Fused multiply then alternately add and subtract: `self * b +- c`,
    /// adding on even lanes and subtracting on odd lanes.
    ///
    /// The conjugated counterpart of [`Vector::fmaddsub`].
    #[inline(always)]
    #[must_use]
    pub fn fmsubadd(self, b: Self, c: Self) -> Self {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        Self::new(unsafe { Arch::fmsubadd(self.raw, b.raw, c.raw) })
    }
}
