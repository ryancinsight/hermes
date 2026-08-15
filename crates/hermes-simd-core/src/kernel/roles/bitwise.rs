//! Bitwise capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for lane-wise and horizontal bitwise operations.
pub trait SimdBitwise<T: Scalar>: SimdStorage<T> + Sealed {
    /// Computes lane-wise bitwise AND.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise bitwise OR.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise bitwise XOR.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise bitwise NOT.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn bitnot(a: Self::Vector) -> Self::Vector;

    /// Counts set bits in each lane.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn popcount(a: Self::Vector) -> Self::Vector;

    /// Reduces all lanes with bitwise AND.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn horizontal_bitwise_and(v: Self::Vector) -> T;

    /// Reduces all lanes with bitwise OR.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn horizontal_bitwise_or(v: Self::Vector) -> T;

    /// Reduces all lanes with bitwise XOR.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn horizontal_bitwise_xor(v: Self::Vector) -> T;
}

impl<T: Scalar, A: BackendKernel<T>> SimdBitwise<T> for A {
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::bitand(a, b)
    }

    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::bitor(a, b)
    }

    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::bitxor(a, b)
    }

    unsafe fn bitnot(a: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::bitnot(a)
    }

    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::popcount(a)
    }

    unsafe fn horizontal_bitwise_and(v: Self::Vector) -> T {
        <A as BackendKernel<T>>::horizontal_bitwise_and(v)
    }

    unsafe fn horizontal_bitwise_or(v: Self::Vector) -> T {
        <A as BackendKernel<T>>::horizontal_bitwise_or(v)
    }

    unsafe fn horizontal_bitwise_xor(v: Self::Vector) -> T {
        <A as BackendKernel<T>>::horizontal_bitwise_xor(v)
    }
}
