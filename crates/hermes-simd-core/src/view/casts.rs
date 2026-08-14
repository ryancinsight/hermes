//! Zero-copy casts between SIMD views over compatible POD element types.

use super::SimdView;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a [T]>
where
    T: bytemuck::Pod,
{
    /// Safe cast of the underlying data slice to a slice of another Pod type, returning a new `SimdView`.
    #[inline]
    #[must_use]
    pub fn cast<U: bytemuck::Pod>(self) -> Option<SimdView<'a, U, Arch, Align, Mode, &'a [U]>> {
        let casted = bytemuck::try_cast_slice(unsafe { &*self.ptr }).ok()?;
        SimdView::new(casted)
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
where
    T: bytemuck::Pod,
{
    /// Safe cast of the underlying mutable data slice to a mutable slice of another Pod type, returning a new mutable `SimdView`.
    #[inline]
    #[must_use]
    pub fn cast_mut<U: bytemuck::Pod>(
        self,
    ) -> Option<SimdView<'a, U, Arch, Align, Mode, &'a mut [U]>> {
        let casted = bytemuck::try_cast_slice_mut(unsafe { &mut *self.ptr }).ok()?;
        SimdView::new_mut(casted)
    }
}
