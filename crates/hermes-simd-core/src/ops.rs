//! Zero-cost operation strategy markers for SIMD reductions and elementwise transforms.
//!
//! `ReductionOp<T>` and `ElementOp<T>` are sealed ZST traits parameterized by the scalar
//! type `T: Scalar`. Concrete strategies (`Sum`, `Dot`, `Mul`, `Add`, `Sub`) implement
//! these traits and are passed as ZST values — they carry no runtime data and the compiler
//! eliminates all abstraction overhead via monomorphization.
//!
//! # Usage
//!
//! ```rust,ignore
//! let total: f32 = view.reduce(ops::Sum);
//! let dot: f32 = view.zip_reduce(&other, ops::Dot)?;
//! ```
//!
//! # Zero-Cost Guarantee
//!
//! Each `unsafe fn accumulate` / `unsafe fn apply` call site is a direct call to
//! an `#[inline(always)]` function that the compiler inlines into the surrounding loop.
//! The ZST parameter is erased entirely — `size_of::<Sum>() == 0`.
//!
//! # Scalar Tail Handling
//!
//! `ElementOp<T>` provides `apply_scalar(a, b) -> T` for processing tail elements that
//! do not fill a complete SIMD vector. This is a pure scalar operation using `T: Scalar`
//! arithmetic operators, eliminating all boundary-condition UB from vector load/store.

use crate::{
    kernel::SimdKernel,
    scalar::Scalar,
};

// ---------------------------------------------------------------------------
// Sealing
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ReductionOp — single-operand fold across lanes
// ---------------------------------------------------------------------------

/// Sealed ZST trait for SIMD horizontal reduction strategies.
///
/// Implementors define how a vector accumulator is updated (`accumulate`) and how
/// the final scalar result is extracted (`finalize`). Both methods are `#[inline(always)]`
/// and carry no branching — DCE eliminates unused strategies entirely.
pub trait ReductionOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Merge a new data vector `v` into accumulator `acc`.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector;

    /// Reduce the final accumulator to a scalar.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T;
}

// ---------------------------------------------------------------------------
// ElementOp — pairwise lane-wise operation
// ---------------------------------------------------------------------------

/// Sealed ZST trait for pairwise SIMD elementwise operations.
///
/// Used by `zip_reduce`, `zip_cow`, and `transform_in_place` to parameterize binary
/// vector operations without code duplication.
///
/// # Scalar Tail
///
/// `apply_scalar(a, b)` handles elements that do not fill a complete SIMD vector.
/// Implementations use direct `T: Scalar` arithmetic so no vector load/store
/// boundary conditions apply. The default does NOT exist — every impl must provide
/// both `apply` (vector) and `apply_scalar` (scalar element).
pub trait ElementOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Apply the operation to two vectors lane-wise: `apply(a, b) -> result`.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector;

    /// Apply the operation to two individual scalar elements: `apply_scalar(a, b) -> result`.
    ///
    /// Used for the SIMD tail (elements that do not fill a complete vector).
    /// Requires only `T: Scalar` — no unsafe, no vector loads or stores.
    fn apply_scalar(a: T, b: T) -> T;
}

// ---------------------------------------------------------------------------
// Concrete strategy ZSTs
// ---------------------------------------------------------------------------

/// Sum reduction: accumulate by adding vectors, finalize with `sum_reduce`.
///
/// `view.reduce(Sum)` is equivalent to `view.sum()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sum;

/// Dot-product pairwise operation: multiply two vectors lane-wise.
///
/// Use with `zip_reduce`: `a.zip_reduce(&b, Dot)` equals `a.dot(&b)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dot;

/// Elementwise multiplication: `a[i] * b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mul;

/// Elementwise addition: `a[i] + b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Add;

/// Elementwise subtraction: `a[i] - b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sub;

/// Elementwise division: `a[i] / b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Div;

/// Elementwise bitwise AND: `a[i] & b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitAnd;

/// Elementwise bitwise OR: `a[i] | b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitOr;

/// Elementwise bitwise XOR: `a[i] ^ b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitXor;

// ---------------------------------------------------------------------------
// Sealing impls
// ---------------------------------------------------------------------------

impl crate::private::Sealed for Sum {}
impl crate::private::Sealed for Dot {}
impl crate::private::Sealed for Mul {}
impl crate::private::Sealed for Add {}
impl crate::private::Sealed for Sub {}
impl crate::private::Sealed for Div {}
impl crate::private::Sealed for BitAnd {}
impl crate::private::Sealed for BitOr {}
impl crate::private::Sealed for BitXor {}

// ---------------------------------------------------------------------------
// ReductionOp impls
// ---------------------------------------------------------------------------

impl<T: Scalar> ReductionOp<T> for Sum {
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        Arch::add(acc, v)
    }

    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::sum_reduce(acc)
    }
}

impl<T: Scalar> ReductionOp<T> for Dot {
    /// Dot accumulation: `acc = fmadd(a, b, acc)` — called with the pairwise product vector.
    ///
    /// The `zip_reduce` loop computes `v = mul(a_chunk, b_chunk)` then calls `accumulate(acc, v)`.
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        // v already holds a[i]*b[i] product from the zip loop; just add to accumulator.
        Arch::add(acc, v)
    }

    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::sum_reduce(acc)
    }
}

// ---------------------------------------------------------------------------
// ElementOp impls
// ---------------------------------------------------------------------------

impl<T: Scalar> ElementOp<T> for Mul {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::mul(a, b)
    }

    #[inline(always)]
    fn apply_scalar(a: T, b: T) -> T {
        a * b
    }
}

impl<T: Scalar> ElementOp<T> for Add {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::add(a, b)
    }

    #[inline(always)]
    fn apply_scalar(a: T, b: T) -> T {
        a + b
    }
}

impl<T: Scalar> ElementOp<T> for Sub {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::sub(a, b)
    }

    #[inline(always)]
    fn apply_scalar(a: T, b: T) -> T {
        a - b
    }
}

impl<T: Scalar> ElementOp<T> for Div {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::div(a, b)
    }

    #[inline(always)]
    fn apply_scalar(a: T, b: T) -> T {
        a / b
    }
}

impl<T: Scalar> ElementOp<T> for BitAnd {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::bitand(a, b)
    }

    #[inline(always)]
    fn apply_scalar(a: T, b: T) -> T {
        a.bitand(b)
    }
}

impl<T: Scalar> ElementOp<T> for BitOr {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::bitor(a, b)
    }

    #[inline(always)]
    fn apply_scalar(a: T, b: T) -> T {
        a.bitor(b)
    }
}

impl<T: Scalar> ElementOp<T> for BitXor {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::bitxor(a, b)
    }

    #[inline(always)]
    fn apply_scalar(a: T, b: T) -> T {
        a.bitxor(b)
    }
}