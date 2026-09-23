//! Horizontal reductions over a single register.

use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Horizontal sum reduction of all lanes in the Vector.
    #[inline(always)]
    pub fn sum_reduce(self) -> T {
        unsafe { Arch::sum_reduce(self.raw) }
    }

    /// Elementwise population count (number of set bits).
    #[inline(always)]
    #[must_use]
    pub fn popcount(self) -> Self {
        Self::new(unsafe { Arch::popcount(self.raw) })
    }

    /// Horizontal bitwise AND reduction across all lanes.
    #[inline(always)]
    pub fn horizontal_bitwise_and(self) -> T {
        unsafe { Arch::horizontal_bitwise_and(self.raw) }
    }

    /// Horizontal bitwise OR reduction across all lanes.
    #[inline(always)]
    pub fn horizontal_bitwise_or(self) -> T {
        unsafe { Arch::horizontal_bitwise_or(self.raw) }
    }

    /// Horizontal bitwise XOR reduction across all lanes.
    #[inline(always)]
    pub fn horizontal_bitwise_xor(self) -> T {
        unsafe { Arch::horizontal_bitwise_xor(self.raw) }
    }
}
