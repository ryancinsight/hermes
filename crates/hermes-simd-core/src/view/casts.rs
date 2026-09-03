//! Zero-copy casts between SIMD views over compatible POD element types.

use super::SimdView;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use eunomia::layout::{try_cast_slice, try_cast_slice_mut, Pod};

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a [T]>
where
    T: Pod,
{
    /// Safe cast of the underlying data slice to a slice of another Pod type, returning a new `SimdView`.
    #[inline]
    #[must_use]
    pub fn cast<U: Pod>(self) -> Option<SimdView<'a, U, Arch, Align, Mode, &'a [U]>> {
        // SAFETY: `ptr` was created from a shared slice that remains valid for
        // `'a`; consuming `self` prevents the source view from being used while
        // the returned view carries that same borrow.
        let casted = try_cast_slice(unsafe { &*self.ptr }).ok()?;
        SimdView::new(casted)
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
where
    T: Pod,
{
    /// Safe cast of the underlying mutable data slice to a mutable slice of another Pod type, returning a new mutable `SimdView`.
    #[inline]
    #[must_use]
    pub fn cast_mut<U: Pod>(self) -> Option<SimdView<'a, U, Arch, Align, Mode, &'a mut [U]>> {
        // SAFETY: `ptr` was created from the exclusive slice borrow carried by
        // `self`; consuming the view transfers that exclusivity to the result.
        let casted = try_cast_slice_mut(unsafe { &mut *self.ptr }).ok()?;
        SimdView::new_mut(casted)
    }
}
