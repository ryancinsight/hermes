//! Single-operand SIMD elementwise operation strategies.
//!
//! `UnaryOp<T>` is a sealed ZST trait for single-operand transforms. Implementors
//! define how a single vector is transformed (`apply`) and how a single scalar element
//! is transformed (`apply_scalar`). Both paths are `#[inline(always)]`.

use crate::kernel::SimdKernel;
use crate::ops::elementwise::Clamp;
use crate::scalar::{NumericElement, Scalar};

// ---------------------------------------------------------------------------
// UnaryOp — single-operand elementwise strategy
// ---------------------------------------------------------------------------

/// Sealed ZST trait for single-operand SIMD elementwise operations.
///
/// Implementors define how a single vector is transformed (`apply`) and how
/// a single scalar element is transformed (`apply_scalar`). Both paths are
/// `#[inline(always)]` — DCE eliminates unused strategies entirely.
///
/// # Zero-Cost Guarantee
///
/// Every `impl UnaryOp<T>` passes through to an `#[inline(always)]
/// SimdKernel<T>` method. The ZST strategy parameter is erased at every
/// monomorphization site: `size_of::<Abs>() == 0`.
pub trait UnaryOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Apply the operation to a vector: `self.apply::<Arch>(v) -> result`.
    ///
    /// Takes `self` by value so `Clamp<T>` can access its bounds; for true ZST
    /// strategies (`Abs`, `Neg`, `Sqrt`), `self` has size zero and the compiler
    /// removes it entirely from the generated code.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector;

    /// Apply the operation to a single scalar element.
    ///
    /// Used for the SIMD tail (elements that do not fill a complete vector).
    /// Requires only `T: Scalar` — no unsafe, no vector loads or stores.
    fn apply_scalar(self, a: T) -> T;
}

// ---------------------------------------------------------------------------
// Concrete unary ZSTs
// ---------------------------------------------------------------------------

/// Elementwise absolute value: `|a[i]|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Abs;

/// Elementwise negation: `-a[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neg;

/// Elementwise square root: `sqrt(a[i])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sqrt;

// ---------------------------------------------------------------------------
// Sealing impls
// ---------------------------------------------------------------------------

impl crate::private::Sealed for Abs {}
impl crate::private::Sealed for Neg {}
impl crate::private::Sealed for Sqrt {}

// ---------------------------------------------------------------------------
// UnaryOp impls
// ---------------------------------------------------------------------------

impl<T: Scalar> UnaryOp<T> for Abs {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        Arch::abs(v)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        a.abs()
    }
}

impl<T: Scalar> UnaryOp<T> for Neg {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        Arch::neg(v)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        T::ZERO - a
    }
}

impl<T: Scalar> UnaryOp<T> for Sqrt {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        Arch::sqrt(v)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        a.sqrt()
    }
}

impl<T: Scalar + PartialOrd + NumericElement> UnaryOp<T> for Clamp<T> {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        // clamp(v, lo, hi) = max(lo, min(v, hi))
        let lo_vec = Arch::splat(self.lo);
        let hi_vec = Arch::splat(self.hi);
        let clamped_hi = Arch::min(v, hi_vec);
        Arch::max(clamped_hi, lo_vec)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        a.min_scalar(self.hi).max_scalar(self.lo)
    }
}
