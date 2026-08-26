//! Exact-width SIMD lane-group views.

use super::Vector;
use crate::arch::SimdArch;
use crate::execution::{ExecutionMode, Unmasked};
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

/// A non-empty view containing exactly one architecture register of elements.
///
/// [`crate::SimdChunks`] and [`crate::ZipChunks`] construct `SimdChunk` values
/// only after validating the parent view and partitioning its complete-lane
/// prefix. Consequently [`SimdChunk::load`] and [`SimdChunk::store`] need no
/// runtime host-support, length, or alignment check.
#[repr(transparent)]
pub struct SimdChunk<'a, T, Arch, Mode: ExecutionMode = Unmasked, Ref: 'a = &'a [T]> {
    ptr: *mut T,
    _marker: PhantomData<(&'a T, Arch, Mode, Ref)>,
}

// SAFETY: immutable chunks expose shared access only; `Ref: Send` carries the
// source borrow's thread-transfer contract.
unsafe impl<T, Arch, Mode: ExecutionMode, Ref: Send> Send for SimdChunk<'_, T, Arch, Mode, Ref> {}

// SAFETY: chunk access is shared unless the reference typestate is mutable;
// `Ref: Sync` carries the source borrow's sharing contract.
unsafe impl<T, Arch, Mode: ExecutionMode, Ref: Sync> Sync for SimdChunk<'_, T, Arch, Mode, Ref> {}

impl<'a, T, Arch, Mode: ExecutionMode> Clone for SimdChunk<'a, T, Arch, Mode, &'a [T]> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, Arch, Mode: ExecutionMode> Copy for SimdChunk<'a, T, Arch, Mode, &'a [T]> {}

impl<T, Arch, Mode: ExecutionMode, Ref> SimdChunk<'_, T, Arch, Mode, Ref>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    /// Construct a complete immutable lane group from a validated parent view.
    ///
    /// # Safety
    ///
    /// The host must support `Arch`, and `ptr` must remain valid for reads of
    /// exactly `Arch::LANE_COUNT` elements for `'a`.
    #[inline(always)]
    pub(crate) const unsafe fn from_supported_ptr(ptr: *const T) -> Self {
        Self {
            ptr: ptr.cast_mut(),
            _marker: PhantomData,
        }
    }

    /// Pointer to the first element in this complete lane group.
    #[inline(always)]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Number of elements in this complete lane group.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        Arch::LANE_COUNT
    }

    /// Returns `false`; a complete architecture lane group is never empty.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

impl<T, Arch, Mode, Ref> SimdChunk<'_, T, Arch, Mode, Ref>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
    Ref: Deref<Target = [T]>,
{
    /// Access the complete lane group as a slice.
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: construction proved validity for exactly `LANE_COUNT` elements.
        unsafe { core::slice::from_raw_parts(self.ptr, Arch::LANE_COUNT) }
    }

    /// Load this complete lane group into one vector register.
    #[inline(always)]
    #[must_use]
    pub fn load(&self) -> Vector<T, Arch> {
        // SAFETY: construction proved host support and validity for one full
        // register; the chunk carries no alignment claim, so use unaligned load.
        unsafe { Vector::load_unaligned(self.ptr) }
    }
}

impl<'a, T, Arch, Mode> SimdChunk<'a, T, Arch, Mode, &'a mut [T]>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
    /// Construct a complete mutable lane group from a validated parent view.
    ///
    /// # Safety
    ///
    /// The host must support `Arch`; `ptr` must remain valid for reads and
    /// writes of exactly `Arch::LANE_COUNT` elements for `'a`; and no live value
    /// may alias those elements mutably.
    #[inline(always)]
    pub(crate) const unsafe fn from_supported_ptr_mut(ptr: *mut T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Mutable pointer to the first element in this complete lane group.
    #[inline(always)]
    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Access the complete lane group as a mutable slice.
    #[inline(always)]
    #[must_use]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        // SAFETY: construction proved exclusive validity for one complete group.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, Arch::LANE_COUNT) }
    }

    /// Store one vector register into this complete lane group.
    #[inline(always)]
    pub fn store(&mut self, vector: Vector<T, Arch>) {
        // SAFETY: both values prove host support, and construction proved an
        // exclusively writable complete group. No alignment is assumed.
        unsafe { vector.store_unaligned(self.ptr) }
    }
}

impl<T, Arch, Mode, Ref> Deref for SimdChunk<'_, T, Arch, Mode, Ref>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
    Ref: Deref<Target = [T]>,
{
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T, Arch, Mode> DerefMut for SimdChunk<'a, T, Arch, Mode, &'a mut [T]>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Mode: ExecutionMode,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}
