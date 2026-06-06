//! Zero-copy SIMD chunk iterators over [`SimdView`].
//!
//! # Design
//!
//! `SimdChunks` iterates non-overlapping sub-views of exactly `LANE_COUNT` elements
//! from a `SimdView`, leaving the remainder (the scalar tail) accessible via
//! [`SimdChunks::remainder`]. This is the canonical pattern for SIMD loop bodies:
//!
//! ```rust,ignore
//! let mut chunks = view.simd_chunks();
//! for chunk in &mut chunks {
//!     // chunk: SimdView<'_, T, Arch, Align>
//!     // process full LANE_COUNT-wide chunk
//! }
//! let tail = chunks.remainder();
//! // process tail[i] in scalar loop
//! ```
//!
//! # Zero-Cost Guarantee
//!
//! `SimdChunks` stores only a raw pointer, a current position, and a total length —
//! 3 words (24 bytes on 64-bit). No heap allocation. The iterator itself is zero-sized
//! relative to the underlying slice; all bounds checks are eliminated for the SIMD
//! chunks because `pos + LANE_COUNT <= simd_end` is verified before each yield.
//!
//! `LANE_COUNT` is read from `Arch::LANE_COUNT` (via `SimdKernel<T>`), a compile-time
//! constant, so the compiler unrolls or vectorizes the advancement step without a runtime
//! division.

use core::marker::PhantomData;
use crate::arch::SimdArch;
use crate::align::Alignment;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::view::SimdView;

/// Iterator over non-overlapping `LANE_COUNT`-wide sub-views of a `SimdView`.
///
/// Created by [`SimdView::simd_chunks`]. The final partial chunk (length `< LANE_COUNT`)
/// is NOT yielded as an `Item`; access it via [`SimdChunks::remainder`] after the loop.
///
/// # Type Parameters
/// Mirrors the parent [`SimdView`] — `T`, `Arch`, `Align`, `Mode` are all preserved so
/// the yielded sub-views carry identical type-level guarantees.
pub struct SimdChunks<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode> {
    /// Base pointer of the original slice.
    base: *const T,
    /// Current element offset (advances by `LANE_COUNT` per step).
    pos: usize,
    /// Total number of elements in the original slice.
    total: usize,
    /// `floor(total / LANE_COUNT) * LANE_COUNT` — the SIMD-processable prefix.
    simd_end: usize,
    _marker: PhantomData<(&'a T, Arch, Align, Mode)>,
}

// SAFETY: SimdChunks borrows `'a` data immutably; forwarding Send/Sync is sound
// when `T: Send` / `T: Sync`.
unsafe impl<'a, T: Send, Arch: SimdArch + crate::kernel::SimdKernel<T>, Align: Alignment, Mode: ExecutionMode> Send
    for SimdChunks<'a, T, Arch, Align, Mode>
where
    T: crate::scalar::Scalar,
{}
unsafe impl<'a, T: Sync, Arch: SimdArch + crate::kernel::SimdKernel<T>, Align: Alignment, Mode: ExecutionMode> Sync
    for SimdChunks<'a, T, Arch, Align, Mode>
where
    T: crate::scalar::Scalar,
{}

impl<'a, T: 'a, Arch: SimdArch + crate::kernel::SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdChunks<'a, T, Arch, Align, Mode>
where
    T: crate::scalar::Scalar,
{
    /// Create a new `SimdChunks` iterator from raw parts.
    ///
    /// # Safety
    /// `base` must be valid for reads of `total` elements for the lifetime `'a`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(base: *const T, total: usize, lane_count: usize) -> Self {
        let simd_end = (total / lane_count) * lane_count;
        Self {
            base,
            pos: 0,
            total,
            simd_end,
            _marker: PhantomData,
        }
    }

    /// Returns the scalar tail — elements that did not fill a complete SIMD vector.
    ///
    /// Equivalent to `&original_slice[simd_end..]`. Length is `total % LANE_COUNT`.
    /// May be empty if `total` is a multiple of `LANE_COUNT`.
    ///
    /// # Usage
    /// Call this after exhausting the iterator (or at any time) to access the tail:
    /// ```rust,ignore
    /// let mut chunks = view.simd_chunks();
    /// for chunk in &mut chunks { /* SIMD body */ }
    /// for &x in chunks.remainder() { /* scalar tail */ }
    /// ```
    #[inline(always)]
    pub fn remainder(&self) -> &'a [T] {
        // SAFETY: base + simd_end is within the original slice of length `total`.
        // simd_end <= total by construction.
        unsafe {
            core::slice::from_raw_parts(
                self.base.add(self.simd_end),
                self.total - self.simd_end,
            )
        }
    }
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdChunks<'a, T, Arch, Align, Mode>
{
    /// Returns the number of complete SIMD chunks remaining.
    ///
    /// After `n` calls to `next`, this returns `original_chunks - n`.
    #[inline(always)]
    pub fn chunks_remaining(&self) -> usize {
        if self.simd_end > self.pos {
            (self.simd_end - self.pos) / Arch::LANE_COUNT
        } else {
            0
        }
    }
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    Iterator for SimdChunks<'a, T, Arch, Align, Mode>
{
    type Item = SimdView<'a, T, Arch, Align, Mode, &'a [T]>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.simd_end {
            return None;
        }
        // SAFETY:
        // - `base + pos` is within the original slice (pos < simd_end <= total).
        // - `pos + LANE_COUNT <= simd_end <= total`, so the sub-slice is within bounds.
        // - `Align::ALIGNMENT` contract is preserved: the base pointer satisfies it,
        //    and `pos * size_of::<T>()` is a multiple of LANE_COUNT*size_of::<T>(),
        //    which is >= ALIGNMENT for all known backends.
        let chunk_slice = unsafe {
            core::slice::from_raw_parts(self.base.add(self.pos), Arch::LANE_COUNT)
        };
        self.pos += Arch::LANE_COUNT;
        // SAFETY: alignment is guaranteed by AlignedVec contract on the parent.
        // For Unaligned parents, `SimdView::new` with `Unaligned` never fails.
        Some(SimdView::new(chunk_slice).expect("chunk alignment invariant violated"))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.chunks_remaining();
        (remaining, Some(remaining))
    }
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    ExactSizeIterator for SimdChunks<'a, T, Arch, Align, Mode>
{}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    DoubleEndedIterator for SimdChunks<'a, T, Arch, Align, Mode>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.pos >= self.simd_end {
            return None;
        }
        self.simd_end -= Arch::LANE_COUNT;
        // SAFETY: same as `next` — `simd_end` is still within original slice bounds.
        let chunk_slice = unsafe {
            core::slice::from_raw_parts(self.base.add(self.simd_end), Arch::LANE_COUNT)
        };
        Some(SimdView::new(chunk_slice).expect("chunk alignment invariant violated"))
    }
}

/// Iterator over non-overlapping mutable `LANE_COUNT`-wide sub-views of a `SimdView`.
///
/// Created by [`SimdView::simd_chunks_mut`]. The final partial chunk (length `< LANE_COUNT`)
/// is NOT yielded as an `Item`; access it via [`SimdChunksMut::into_remainder`] after the loop.
pub struct SimdChunksMut<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode> {
    /// Base pointer of the original slice.
    base: *mut T,
    /// Current element offset (advances by `LANE_COUNT` per step).
    pos: usize,
    /// Total number of elements in the original slice.
    total: usize,
    /// `floor(total / LANE_COUNT) * LANE_COUNT` — the SIMD-processable prefix.
    simd_end: usize,
    _marker: PhantomData<(&'a mut T, Arch, Align, Mode)>,
}

// SAFETY: SimdChunksMut borrows `'a` data mutably; forwarding Send/Sync is sound
// when `T: Send` / `T: Sync`.
unsafe impl<'a, T: Send, Arch: SimdArch + crate::kernel::SimdKernel<T>, Align: Alignment, Mode: ExecutionMode> Send
    for SimdChunksMut<'a, T, Arch, Align, Mode>
where
    T: crate::scalar::Scalar,
{}
unsafe impl<'a, T: Sync, Arch: SimdArch + crate::kernel::SimdKernel<T>, Align: Alignment, Mode: ExecutionMode> Sync
    for SimdChunksMut<'a, T, Arch, Align, Mode>
where
    T: crate::scalar::Scalar,
{}

impl<'a, T: 'a, Arch: SimdArch + crate::kernel::SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdChunksMut<'a, T, Arch, Align, Mode>
where
    T: crate::scalar::Scalar,
{
    /// Create a new `SimdChunksMut` iterator from raw parts.
    ///
    /// # Safety
    /// `base` must be valid for reads and writes of `total` elements for the lifetime `'a`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(base: *mut T, total: usize, lane_count: usize) -> Self {
        let simd_end = (total / lane_count) * lane_count;
        Self {
            base,
            pos: 0,
            total,
            simd_end,
            _marker: PhantomData,
        }
    }

    /// Returns the mutable scalar tail — elements that did not fill a complete SIMD vector.
    ///
    /// Consumes the iterator to return a mutable slice with lifetime `'a`.
    #[inline(always)]
    pub fn into_remainder(self) -> &'a mut [T] {
        // SAFETY: base + simd_end is within the original slice of length `total`.
        // simd_end <= total by construction.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.base.add(self.simd_end),
                self.total - self.simd_end,
            )
        }
    }
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdChunksMut<'a, T, Arch, Align, Mode>
{
    /// Returns the number of complete SIMD chunks remaining.
    ///
    /// After `n` calls to `next`, this returns `original_chunks - n`.
    #[inline(always)]
    pub fn chunks_remaining(&self) -> usize {
        if self.simd_end > self.pos {
            (self.simd_end - self.pos) / Arch::LANE_COUNT
        } else {
            0
        }
    }
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    Iterator for SimdChunksMut<'a, T, Arch, Align, Mode>
{
    type Item = SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.simd_end {
            return None;
        }
        // SAFETY:
        // - `base + pos` is within the original slice (pos < simd_end <= total).
        // - `pos + LANE_COUNT <= simd_end <= total`, so the sub-slice is within bounds.
        // - `Align::ALIGNMENT` contract is preserved: the base pointer satisfies it,
        //    and `pos * size_of::<T>()` is a multiple of LANE_COUNT*size_of::<T>(),
        //    which is >= ALIGNMENT for all known backends.
        let chunk_slice = unsafe {
            core::slice::from_raw_parts_mut(self.base.add(self.pos), Arch::LANE_COUNT)
        };
        self.pos += Arch::LANE_COUNT;
        // SAFETY: alignment is guaranteed by AlignedVec contract on the parent.
        // For Unaligned parents, `SimdView::new_mut` with `Unaligned` never fails.
        Some(SimdView::new_mut(chunk_slice).expect("chunk alignment invariant violated"))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.chunks_remaining();
        (remaining, Some(remaining))
    }
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    ExactSizeIterator for SimdChunksMut<'a, T, Arch, Align, Mode>
{}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    DoubleEndedIterator for SimdChunksMut<'a, T, Arch, Align, Mode>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.pos >= self.simd_end {
            return None;
        }
        self.simd_end -= Arch::LANE_COUNT;
        // SAFETY: same as `next` — `simd_end` is still within original slice bounds.
        let chunk_slice = unsafe {
            core::slice::from_raw_parts_mut(self.base.add(self.simd_end), Arch::LANE_COUNT)
        };
        Some(SimdView::new_mut(chunk_slice).expect("chunk alignment invariant violated"))
    }
}