//! Comparison capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for lane comparisons and blends.
pub trait SimdCompare<T: Scalar>: SimdStorage<T> + Sealed {
    /// Compares lanes for equality.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Compares lanes for inequality.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Compares lanes for less-than ordering.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Compares lanes for less-than-or-equal ordering.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Compares lanes for greater-than ordering.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Compares lanes for greater-than-or-equal ordering.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Selects `true_val` or `false_val` using the sign bit of each mask lane.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector;
}

impl<T: Scalar, A: BackendKernel<T>> SimdCompare<T> for A {
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::cmp_eq(a, b)
    }

    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::cmp_ne(a, b)
    }

    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::cmp_lt(a, b)
    }

    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::cmp_le(a, b)
    }

    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::cmp_gt(a, b)
    }

    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::cmp_ge(a, b)
    }

    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        <A as BackendKernel<T>>::blend(mask, true_val, false_val)
    }
}
