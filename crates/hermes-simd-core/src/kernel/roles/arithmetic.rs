//! Dense arithmetic capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::private::Sealed;
use crate::scalar::{FloatElement, RoundTiesEven, Scalar};

use super::storage::SimdStorage;

/// Backend capability for dense elementwise arithmetic and scalar broadcast.
pub trait SimdArith<T: Scalar>: SimdStorage<T> + Sealed {
    /// Computes lane-wise addition.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise multiplication.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise subtraction.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise fused multiply-add.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;

    /// Computes masked lane-wise addition with merge semantics.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Computes masked lane-wise multiplication with merge semantics.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Computes masked fused multiply-add with merge semantics.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector;

    /// Returns an all-zero register.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn zero() -> Self::Vector;

    /// Broadcasts one scalar to every lane.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn splat(val: T) -> Self::Vector;

    /// Computes lane-wise division.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise absolute value.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn abs(a: Self::Vector) -> Self::Vector;

    /// Computes lane-wise minimum.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise maximum.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Computes lane-wise square root.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector;

    /// Computes lane-wise reciprocal square root.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector;

    /// Computes lane-wise floor.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn floor(a: Self::Vector) -> Self::Vector
    where
        T: FloatElement;

    /// Computes lane-wise ceiling.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn ceil(a: Self::Vector) -> Self::Vector
    where
        T: FloatElement;

    /// Computes lane-wise round-to-nearest-even.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn round(a: Self::Vector) -> Self::Vector
    where
        T: RoundTiesEven;

    /// Computes lane-wise truncation toward zero.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn trunc(a: Self::Vector) -> Self::Vector
    where
        T: FloatElement;

    /// Computes lane-wise negation.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn neg(a: Self::Vector) -> Self::Vector;
}

impl<T: Scalar, A: BackendKernel<T>> SimdArith<T> for A {
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::add(a, b)
    }

    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::mul(a, b)
    }

    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::sub(a, b)
    }

    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::fmadd(a, b, c)
    }

    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        <A as BackendKernel<T>>::masked_add(a, b, mask, src)
    }

    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        <A as BackendKernel<T>>::masked_mul(a, b, mask, src)
    }

    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        <A as BackendKernel<T>>::masked_fmadd(a, b, c, mask)
    }

    unsafe fn zero() -> Self::Vector {
        <A as BackendKernel<T>>::zero()
    }

    unsafe fn splat(val: T) -> Self::Vector {
        <A as BackendKernel<T>>::splat(val)
    }

    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::div(a, b)
    }

    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::abs(a)
    }

    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::min(a, b)
    }

    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::max(a, b)
    }

    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::sqrt(a)
    }

    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::recip_sqrt(a)
    }

    unsafe fn floor(a: Self::Vector) -> Self::Vector
    where
        T: FloatElement,
    {
        <A as BackendKernel<T>>::floor(a)
    }

    unsafe fn ceil(a: Self::Vector) -> Self::Vector
    where
        T: FloatElement,
    {
        <A as BackendKernel<T>>::ceil(a)
    }

    unsafe fn round(a: Self::Vector) -> Self::Vector
    where
        T: RoundTiesEven,
    {
        <A as BackendKernel<T>>::round(a)
    }

    unsafe fn trunc(a: Self::Vector) -> Self::Vector
    where
        T: FloatElement,
    {
        <A as BackendKernel<T>>::trunc(a)
    }

    unsafe fn neg(a: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::neg(a)
    }
}
