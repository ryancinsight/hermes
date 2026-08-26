//! Capability for constructing SIMD values after one runtime support check.

use super::{Mask, SimdView, Vector};
use crate::align::Unaligned;
use crate::arch::SimdArch;
use crate::execution::Unmasked;
use crate::iter::SimdIoChunks;
use crate::kernel::SimdKernel;
use crate::mask::BitMask;
use crate::scalar::Scalar;
use core::marker::PhantomData;

/// Proof that the current host supports `Arch` for scalar `T`.
///
/// [`crate::Vector`], [`crate::Mask`], and [`SimdView`] values derived from this
/// capability can invoke architecture operations without repeating the runtime
/// feature probe. Consumer kernels receive a `Simd` from
/// `hermes_simd::vectorize`; direct construction is unsafe because a freely
/// nameable architecture marker is not itself a runtime capability.
pub struct Simd<T, Arch> {
    _marker: PhantomData<(T, Arch)>,
}

impl<T, Arch> Simd<T, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    /// Construct a capability without probing the host.
    ///
    /// # Safety
    ///
    /// The current host must support every target feature required by `Arch`.
    /// Prefer `hermes_simd::vectorize`, which performs the probe and constructs
    /// this capability inside the selected target-feature scope.
    #[inline(always)]
    #[must_use]
    pub const unsafe fn assume_supported() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Construct a vector whose lanes are all zero without another host probe.
    #[inline(always)]
    #[must_use]
    pub fn zero(&self) -> Vector<T, Arch> {
        // SAFETY: possession of `self` proves host support for `Arch`.
        Vector::new(unsafe { Arch::zero() })
    }

    /// Broadcast `value` to every lane without another host probe.
    #[inline(always)]
    #[must_use]
    pub fn splat(&self, value: T) -> Vector<T, Arch> {
        // SAFETY: possession of `self` proves host support for `Arch`.
        Vector::new(unsafe { Arch::splat(value) })
    }

    /// Construct a mask from portable lane bits without another host probe.
    #[inline(always)]
    #[must_use]
    pub fn mask_from_bitmask(&self, bits: BitMask<64>) -> Mask<T, Arch> {
        // SAFETY: possession of `self` proves host support for `Arch`.
        Mask::new(unsafe { Arch::mask_from_bitmask(bits.0) })
    }

    /// Construct a read-only unaligned view without another host probe.
    #[inline(always)]
    #[must_use]
    pub fn view<'a>(&self, data: &'a [T]) -> SimdView<'a, T, Arch, Unaligned> {
        // SAFETY: possession of `self` proves host support, `data` carries its
        // own validity lifetime, and the result claims no alignment.
        unsafe { SimdView::from_supported_slice(data) }
    }

    /// Construct a mutable unaligned view without another host probe.
    #[inline(always)]
    #[must_use]
    pub fn view_mut<'a>(
        &self,
        data: &'a mut [T],
    ) -> SimdView<'a, T, Arch, Unaligned, Unmasked, &'a mut [T]> {
        // SAFETY: possession of `self` proves host support, the exclusive borrow
        // proves non-aliasing for `'a`, and the result claims no alignment.
        unsafe { SimdView::from_supported_slice_mut(data) }
    }

    /// Iterate groups of planar inputs and outputs under one shared loop limit.
    ///
    /// Every yielded input or output contains exactly one architecture register.
    /// Consume the returned iterator with [`SimdIoChunks::into_remainders`] to
    /// handle every source suffix that the iterator has not yielded.
    #[inline(always)]
    #[must_use]
    pub fn io_chunks<'input, 'output, const INPUTS: usize, const OUTPUTS: usize>(
        &self,
        inputs: [&'input [T]; INPUTS],
        outputs: [&'output mut [T]; OUTPUTS],
    ) -> SimdIoChunks<'input, 'output, T, Arch, Unmasked, INPUTS, OUTPUTS> {
        SimdIoChunks::from_supported_slices(inputs, outputs)
    }
}
