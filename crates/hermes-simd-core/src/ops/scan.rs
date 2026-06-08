//! Associative prefix-scan operation strategies and inclusion-mode ZSTs.
//!
//! `ScanOp<T>` is a sealed ZST trait for prefix scan operations (cumulative sum,
//! cumulative product, running min/max). `ScanMode` is a sealed ZST trait for
//! inclusive vs exclusive scan variants.

use crate::scalar::Scalar;

// ---------------------------------------------------------------------------
// ScanOp — associative binary prefix-scan operation
// ---------------------------------------------------------------------------

/// Sealed ZST trait for prefix scan operations.
pub trait ScanOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Returns the identity element of the operation.
    fn identity() -> T;
    /// Combine two values using the operation: `a op b`.
    fn combine(a: T, b: T) -> T;
}

/// Addition scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanAdd;

/// Multiplication scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanMul;

/// Minimum scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanMin;

/// Maximum scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanMax;

impl crate::private::Sealed for ScanAdd {}
impl crate::private::Sealed for ScanMul {}
impl crate::private::Sealed for ScanMin {}
impl crate::private::Sealed for ScanMax {}

impl<T: Scalar> ScanOp<T> for ScanAdd {
    #[inline(always)]
    fn identity() -> T {
        T::ZERO
    }
    #[inline(always)]
    fn combine(a: T, b: T) -> T {
        a + b
    }
}

impl<T: Scalar> ScanOp<T> for ScanMul {
    #[inline(always)]
    fn identity() -> T {
        T::ONE
    }
    #[inline(always)]
    fn combine(a: T, b: T) -> T {
        a * b
    }
}

impl<T: Scalar> ScanOp<T> for ScanMin {
    #[inline(always)]
    fn identity() -> T {
        T::MAX_VALUE
    }
    #[inline(always)]
    fn combine(a: T, b: T) -> T {
        a.min_scalar(b)
    }
}

impl<T: Scalar> ScanOp<T> for ScanMax {
    #[inline(always)]
    fn identity() -> T {
        T::MIN_VALUE
    }
    #[inline(always)]
    fn combine(a: T, b: T) -> T {
        a.max_scalar(b)
    }
}

// ---------------------------------------------------------------------------
// ScanMode — prefix-scan inclusion mode
// ---------------------------------------------------------------------------

/// Sealed ZST trait for prefix scan inclusion modes.
pub trait ScanMode: crate::private::Sealed + Copy + 'static {
    /// Whether the scan is inclusive of the current element.
    const IS_INCLUSIVE: bool;
}

/// Inclusive scan ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inclusive;

/// Exclusive scan ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exclusive;

impl crate::private::Sealed for Inclusive {}
impl crate::private::Sealed for Exclusive {}

impl ScanMode for Inclusive {
    const IS_INCLUSIVE: bool = true;
}

impl ScanMode for Exclusive {
    const IS_INCLUSIVE: bool = false;
}
