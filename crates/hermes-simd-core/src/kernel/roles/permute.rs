//! Cross-lane permutation capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::ops::{ScanMode, ScanOp};
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for scans, lane permutations, and adjacent shuffles.
pub trait SimdPermute<T: Scalar>: SimdStorage<T> + Sealed {
    /// Performs an inclusive or exclusive intra-register scan.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn scan_vector<Op: ScanOp<T>, SMode: ScanMode>(
        v: Self::Vector,
        carry: T,
    ) -> (Self::Vector, T);

    /// Reverses the lane order.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn reverse(v: Self::Vector) -> Self::Vector;

    /// Interleaves two registers into low and high result registers.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector);

    /// Deinterleaves two registers into even and odd result registers.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector);

    /// Swaps each adjacent lane pair.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector;

    /// Duplicates each even lane into its adjacent odd lane.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector;

    /// Duplicates each odd lane into its adjacent even lane.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector;

    /// Computes alternating fused multiply-add/subtract lanes.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;

    /// Computes alternating fused multiply-subtract/add lanes.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
}

impl<T: Scalar, A: BackendKernel<T>> SimdPermute<T> for A {
    unsafe fn scan_vector<Op: ScanOp<T>, SMode: ScanMode>(
        v: Self::Vector,
        carry: T,
    ) -> (Self::Vector, T) {
        <A as BackendKernel<T>>::scan_vector::<Op, SMode>(v, carry)
    }

    unsafe fn reverse(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::reverse(v)
    }

    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::interleave(a, b)
    }

    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::deinterleave(a, b)
    }

    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::swap_adjacent(v)
    }

    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::dup_even(v)
    }

    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::dup_odd(v)
    }

    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::fmaddsub(a, b, c)
    }

    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::fmsubadd(a, b, c)
    }
}
