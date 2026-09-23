//! Construction: zero/splat and their fallible counterparts.

use super::runtime_support_result;
use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::view::SimdError;
use core::marker::PhantomData;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Wrap a raw register after the caller has established host support.
    #[inline(always)]
    pub(crate) const fn new(raw: Arch::Vector) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Construct a Vector with all lanes set to zero.
    ///
    /// # Panics
    ///
    /// Panics if the architecture is not supported or enabled on this host.
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self::try_zero().expect("SIMD target is not supported or enabled on this host")
    }

    /// Try to construct a Vector with all lanes set to zero.
    ///
    /// # Errors
    /// Returns [`SimdError::UnsupportedTarget`] when the architecture is not
    /// supported or enabled on this host.
    #[inline(always)]
    pub fn try_zero() -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        Ok(Self::new(unsafe { Arch::zero() }))
    }

    /// Construct a Vector by broadcasting a scalar value to all lanes.
    ///
    /// # Panics
    ///
    /// Panics if the architecture is not supported or enabled on this host.
    #[inline(always)]
    pub fn splat(val: T) -> Self {
        Self::try_splat(val).expect("SIMD target is not supported or enabled on this host")
    }

    /// Try to construct a Vector by broadcasting a scalar value to all lanes.
    ///
    /// # Errors
    /// Returns [`SimdError::UnsupportedTarget`] when the architecture is not
    /// supported or enabled on this host.
    #[inline(always)]
    pub fn try_splat(val: T) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        Ok(Self::new(unsafe { Arch::splat(val) }))
    }

    /// Broadcasts one lane pair across the register: `[lo, hi, lo, hi, ...]`.
    ///
    /// On interleaved complex data this is one sample filling the register —
    /// the twiddle-factor shape a per-register complex multiply consumes. Each
    /// ISA has a single pair-broadcast instruction, so this costs one op where
    /// interleaving two scalar broadcasts costs a shuffle pair.
    /// # Panics
    ///
    /// Panics if the architecture is not supported or enabled on this host.
    #[inline(always)]
    #[must_use]
    pub fn splat_pair(lo: T, hi: T) -> Self {
        Self::try_splat_pair(lo, hi).expect("SIMD target is not supported or enabled on this host")
    }

    /// Try to broadcast one lane pair across the register.
    ///
    /// The fallible half of [`Self::splat_pair`], as [`Self::try_splat`] is of
    /// [`Self::splat`]: both are constructors, so neither has an existing
    /// register to prove host support, and the check is theirs to make.
    ///
    /// # Errors
    /// Returns [`SimdError::UnsupportedTarget`] when the architecture is not
    /// supported or enabled on this host.
    #[inline(always)]
    pub fn try_splat_pair(lo: T, hi: T) -> Result<Self, SimdError> {
        runtime_support_result::<T, Arch>()?;
        // SAFETY: the check above proved the host supports `Arch`.
        Ok(Self::new(unsafe { Arch::splat_pair(lo, hi) }))
    }
}
