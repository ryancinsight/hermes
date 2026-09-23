//! Bridges between a register and a `SimdView` chunk.

use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Load a Vector from a chunk index of a `SimdView`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_idx` does not identify a complete SIMD lane group.
    #[inline(always)]
    #[must_use]
    pub fn from_view_chunk<Align, Mode, Ref>(
        view: &crate::view::SimdView<'_, T, Arch, Align, Mode, Ref>,
        chunk_idx: usize,
    ) -> Self
    where
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
        Ref: core::ops::Deref<Target = [T]>,
    {
        let offset = chunk_idx * Arch::LANE_COUNT;
        let slice = view.as_slice();
        assert!(
            offset + Arch::LANE_COUNT <= slice.len(),
            "Chunk index out of bounds"
        );
        // SAFETY: constructing `view` proved host support; the assert guarantees
        // `offset + LANE_COUNT <= slice.len()`, so the load reads a full vector in
        // bounds. The aligned variant is taken only when `Align` proves the base
        // pointer is arch-aligned and `offset` is a lane-count multiple.
        unsafe {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                Self::load_aligned(slice.as_ptr().add(offset))
            } else {
                Self::load_unaligned(slice.as_ptr().add(offset))
            }
        }
    }

    /// Store this Vector into a mutable chunk of a mutable `SimdView`.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_idx` does not identify a complete SIMD lane group.
    #[inline(always)]
    pub fn store_to_view_chunk<'a, Align, Mode>(
        self,
        view: &mut crate::view::SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>,
        chunk_idx: usize,
    ) where
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
    {
        let offset = chunk_idx * Arch::LANE_COUNT;
        let slice = view.as_slice_mut();
        assert!(
            offset + Arch::LANE_COUNT <= slice.len(),
            "Chunk index out of bounds"
        );
        // SAFETY: as `from_view_chunk` — the assert guarantees
        // `offset + LANE_COUNT <= slice.len()`, so the store writes a full vector
        // in bounds; the aligned variant is gated on `Align`.
        unsafe {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                self.store_aligned(slice.as_mut_ptr().add(offset));
            } else {
                self.store_unaligned(slice.as_mut_ptr().add(offset));
            }
        }
    }
}
