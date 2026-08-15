//! Multiplicative reduction strategy.

use crate::kernel::{SimdArith, SimdCompare, SimdLoadStore, SimdMask, SimdReduce, SimdStorage};
use crate::scalar::Scalar;

use super::ReductionOp;

/// Multiplicative reduction: computes `∏ data[i]`.
///
/// Identity element is `T::ONE`. Uses SIMD `mul` to accumulate lane products, then
/// reduces horizontally via a scalar lane-extraction loop because the operation
/// family has no universally available `prod_reduce` primitive.
///
/// # Zero-Cost Guarantee
///
/// `size_of::<Product>() == 0`. The strategy is selected through monomorphization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Product;

impl crate::private::Sealed for Product {}

impl<T: Scalar> ReductionOp<T> for Product {
    /// Accumulate: `acc = acc * v` (lane-wise multiply).
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn accumulate<
        Arch: SimdLoadStore<T> + SimdArith<T> + SimdCompare<T> + SimdMask<T> + SimdReduce<T>,
    >(
        acc: Arch::Vector,
        v: Arch::Vector,
    ) -> Arch::Vector {
        Arch::mul(acc, v)
    }

    /// Finalize the accumulated product vector through initialized scalar storage.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn finalize<
        Arch: SimdLoadStore<T> + SimdArith<T> + SimdCompare<T> + SimdMask<T> + SimdReduce<T>,
    >(
        acc: Arch::Vector,
    ) -> T {
        // Compile-time bound against the shared scalar-fallback buffer SSOT.
        const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
        let mut buf = [T::ZERO; crate::kernel::MAX_SIMD_LANES];
        Arch::store_unaligned(buf.as_mut_ptr(), acc);
        let mut result = T::ONE;
        for i in 0..Arch::LANE_COUNT {
            result = result * buf[i];
        }
        result
    }

    #[inline(always)]
    fn identity_scalar() -> T {
        T::ONE
    }

    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a * b
    }
}
