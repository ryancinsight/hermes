//! Zero-copy bridge from a rank-1 [`TensorView`] to a typed
//! [`SimdView`](crate::view::SimdView).
//!
//! This is the seam where the tensor substrate hands a contiguous 1-D view to
//! the SIMD execution layer; it is kept separate so the dependency on the
//! `view`/`arch`/`kernel` stack does not leak into the rank-agnostic core.

use super::TensorView;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::Unmasked;
use crate::kernel::SimdKernel;
use crate::tensor::layout::RowMajor;
use crate::view::SimdView;

impl<'a, T> TensorView<'a, T, 1, RowMajor, &'a [T]>
where
    T: crate::scalar::Scalar,
{
    /// Promote this contiguous rank-1 view into a typed [`SimdView`].
    ///
    /// Zero-copy: shares the same underlying slice. Returns `None` if the slice is empty
    /// or if the alignment check fails for `Align`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use hermes_simd_core::tensor::TensorView;
    /// use hermes_simd_intrinsics::Scalar;
    /// use hermes_simd_core::align::Unaligned;
    ///
    /// let data = [1.0f32, 2.0, 3.0, 4.0];
    /// let t = TensorView::<f32, 1>::new(&data, [4]).unwrap();
    /// let view = t.into_simd_view::<Scalar, Unaligned>().unwrap();
    /// ```
    #[inline]
    #[must_use]
    pub fn into_simd_view<Arch, Align>(
        &self,
    ) -> Option<SimdView<'a, T, Arch, Align, Unmasked, &'a [T]>>
    where
        Arch: SimdArch + SimdKernel<T>,
        Align: Alignment,
    {
        // SAFETY: this impl is bound to `Ref = &'a [T]` (an immutable shared
        // borrow constructed for `'a`), so the `*mut [T]` was originally a
        // `&'a [T]`; reconstructing a `&'a [T]` from it is sound, and
        // `SimdView::new` reborrows it for the same `'a`.
        let slice: &'a [T] = unsafe { &*self.ptr };
        SimdView::new(slice)
    }
}
