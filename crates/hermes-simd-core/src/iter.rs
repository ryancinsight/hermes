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

// ---------------------------------------------------------------------------
// FusedIterator — optimizer hint: once exhausted, next() always returns None.
// ---------------------------------------------------------------------------

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    core::iter::FusedIterator for SimdChunks<'a, T, Arch, Align, Mode>
{}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    core::iter::FusedIterator for SimdChunksMut<'a, T, Arch, Align, Mode>
{}

// ---------------------------------------------------------------------------
// ZipChunks — paired chunk iterator for two immutable SimdViews.
// ---------------------------------------------------------------------------

/// Iterator over non-overlapping paired `LANE_COUNT`-wide sub-views of two `SimdView`s.
///
/// Advances both views in lockstep, yielding pairs of chunks until the shorter
/// view's SIMD prefix is exhausted.  Access the tails via [`ZipChunks::remainder`].
///
/// # Type Parameters
/// Mirrors both parent `SimdView`s — element type `T`, architecture `Arch`,
/// alignment `Align`, and execution mode `Mode` are shared between the two views.
///
/// # Zero-Cost Guarantee
/// Stores two base pointers, two positions, and one shared `simd_end` — 5 words
/// on 64-bit targets. No heap allocation. `LANE_COUNT` is a compile-time constant so
/// the advancement step optimizes identically to the single-view `SimdChunks` case.
pub struct ZipChunks<'a, 'b, T, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode> {
    base_a: *const T,
    base_b: *const T,
    pos: usize,
    simd_end: usize,
    total_a: usize,
    total_b: usize,
    _marker: core::marker::PhantomData<(&'a T, &'b T, Arch, Align, Mode)>,
}

// SAFETY: ZipChunks borrows two `'a`/`'b` immutable slices; forwarding Send/Sync is sound.
unsafe impl<'a, 'b, T: Send, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    Send for ZipChunks<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar,
{}
unsafe impl<'a, 'b, T: Sync, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    Sync for ZipChunks<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar,
{}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    ZipChunks<'a, 'b, T, Arch, Align, Mode>
{
    /// Construct a `ZipChunks` from two slice pointers.
    ///
    /// # Safety
    /// `base_a` must be valid for reads of `total_a` elements for `'a`.
    /// `base_b` must be valid for reads of `total_b` elements for `'b`.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        base_a: *const T,
        total_a: usize,
        base_b: *const T,
        total_b: usize,
    ) -> Self {
        let lane_count = Arch::LANE_COUNT;
        let min_total = total_a.min(total_b);
        let simd_end = (min_total / lane_count) * lane_count;
        Self {
            base_a,
            base_b,
            pos: 0,
            simd_end,
            total_a,
            total_b,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the scalar tails for both views.
    ///
    /// Each tail contains elements from `simd_end` to `total` for the respective slice.
    #[inline(always)]
    pub fn remainder(&self) -> (&'a [T], &'b [T]) {
        // SAFETY: base + simd_end is within bounds by construction.
        unsafe {
            (
                core::slice::from_raw_parts(self.base_a.add(self.simd_end), self.total_a - self.simd_end),
                core::slice::from_raw_parts(self.base_b.add(self.simd_end), self.total_b - self.simd_end),
            )
        }
    }

    /// Returns the number of complete paired SIMD chunks remaining.
    #[inline(always)]
    pub fn chunks_remaining(&self) -> usize {
        if self.simd_end > self.pos {
            (self.simd_end - self.pos) / Arch::LANE_COUNT
        } else {
            0
        }
    }
}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    Iterator for ZipChunks<'a, 'b, T, Arch, Align, Mode>
{
    type Item = (
        SimdView<'a, T, Arch, Align, Mode, &'a [T]>,
        SimdView<'b, T, Arch, Align, Mode, &'b [T]>,
    );

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.simd_end {
            return None;
        }
        // SAFETY: pos < simd_end <= min(total_a, total_b) <= total_a and total_b.
        let (chunk_a, chunk_b) = unsafe {
            (
                core::slice::from_raw_parts(self.base_a.add(self.pos), Arch::LANE_COUNT),
                core::slice::from_raw_parts(self.base_b.add(self.pos), Arch::LANE_COUNT),
            )
        };
        self.pos += Arch::LANE_COUNT;
        Some((
            SimdView::new(chunk_a).expect("zip chunk_a alignment invariant violated"),
            SimdView::new(chunk_b).expect("zip chunk_b alignment invariant violated"),
        ))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.chunks_remaining();
        (r, Some(r))
    }
}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    ExactSizeIterator for ZipChunks<'a, 'b, T, Arch, Align, Mode>
{}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    core::iter::FusedIterator for ZipChunks<'a, 'b, T, Arch, Align, Mode>
{}

// ---------------------------------------------------------------------------
// ZipChunksMut — paired mutable/immutable SIMD chunk iterator
// ---------------------------------------------------------------------------

/// Paired iterator over non-overlapping `LANE_COUNT`-wide sub-views where the first
/// operand is **mutable** and the second is **immutable**.
///
/// Enables zero-copy 2-operand in-place transforms without unsafe pointer arithmetic
/// at call sites. Canonical usage (SAXPY: `a[i] += s * b[i]`):
///
/// ```rust,ignore
/// let mut chunks = view_a.zip_chunks_mut(&view_b);
/// for (mut chunk_a, chunk_b) in &mut chunks {
///     // chunk_a: SimdView<'a, ..., &'a mut [T]>
///     // chunk_b: SimdView<'b, ..., &'b [T]>
///     chunk_a.transform_in_place(&chunk_b, FmaAdd);
/// }
/// let (tail_a, tail_b) = chunks.into_remainder();
/// for (a, &b) in tail_a.iter_mut().zip(tail_b) {
///     *a = (*a).scalar_fmadd(b, T::ZERO);
/// }
/// ```
///
/// # Zero-Cost Guarantee
///
/// `ZipChunksMut` stores one `*mut T`, one `*const T`, and three `usize` — 5 words total.
/// No heap allocation. `LANE_COUNT` is a compile-time constant, so pointer advancement is
/// a single `add` instruction.
pub struct ZipChunksMut<'a, 'b, T: 'a + 'b, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode> {
    /// Base mutable pointer for the first (output) operand.
    ptr_a: *mut T,
    /// Base immutable pointer for the second (input) operand.
    ptr_b: *const T,
    /// Current element offset.
    pos: usize,
    /// Total elements in the shorter of the two slices.
    total: usize,
    /// `floor(total / LANE_COUNT) * LANE_COUNT` — the SIMD-processable prefix.
    simd_end: usize,
    _marker: PhantomData<(&'a mut T, &'b T, Arch, Align, Mode)>,
}

// SAFETY: `ZipChunksMut` holds exclusive (`*mut T`) access to `'a` data and shared
// (`*const T`) access to `'b` data. Forwarding Send/Sync is sound when `T: Send + Sync`.
unsafe impl<'a, 'b, T, Arch, Align, Mode> Send for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar + Send + Sync,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{}
unsafe impl<'a, 'b, T, Arch, Align, Mode> Sync for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar + Send + Sync,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{}

impl<'a, 'b, T: 'a + 'b, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar,
{
    /// Create a new `ZipChunksMut` from raw pointer parts.
    ///
    /// # Safety
    /// - `ptr_a` must be valid for **exclusive** reads and writes of `total_a` elements for `'a`.
    /// - `ptr_b` must be valid for reads of `total_b` elements for `'b`.
    /// - The two ranges must not overlap.
    #[inline]
    pub(crate) unsafe fn from_raw_parts(
        ptr_a: *mut T,
        total_a: usize,
        ptr_b: *const T,
        total_b: usize,
    ) -> Self {
        let lane_count = Arch::LANE_COUNT;
        let total = total_a.min(total_b);
        let simd_end = (total / lane_count) * lane_count;
        Self {
            ptr_a,
            ptr_b,
            pos: 0,
            total,
            simd_end,
            _marker: PhantomData,
        }
    }

    /// Consume the iterator and return the scalar tail slices.
    ///
    /// Elements `[simd_end..total]` for each operand. Call this **after** the loop.
    #[inline(always)]
    pub fn into_remainder(self) -> (&'a mut [T], &'b [T]) {
        let len = self.total - self.simd_end;
        // SAFETY: ptr_a/ptr_b + simd_end is within bounds by construction; both
        // lifetimes are preserved by the return type.
        unsafe {
            (
                core::slice::from_raw_parts_mut(self.ptr_a.add(self.simd_end), len),
                core::slice::from_raw_parts(self.ptr_b.add(self.simd_end), len),
            )
        }
    }

    /// Number of complete SIMD chunks remaining.
    #[inline(always)]
    pub fn chunks_remaining(&self) -> usize {
        if self.simd_end > self.pos {
            (self.simd_end - self.pos) / Arch::LANE_COUNT
        } else {
            0
        }
    }
}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    Iterator for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
{
    /// Yields `(mutable chunk of A, immutable chunk of B)`.
    type Item = (
        SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>,
        SimdView<'b, T, Arch, Align, Mode, &'b [T]>,
    );

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.simd_end {
            return None;
        }
        let lane = Arch::LANE_COUNT;
        // SAFETY: pos < simd_end <= total <= total_a and total_b; exclusive access
        // is guaranteed by the `&'a mut [T]` lifetime held by the caller.
        let (chunk_a, chunk_b) = unsafe {
            (
                core::slice::from_raw_parts_mut(self.ptr_a.add(self.pos), lane),
                core::slice::from_raw_parts(self.ptr_b.add(self.pos), lane),
            )
        };
        self.pos += lane;
        Some((
            SimdView::new_mut(chunk_a).expect("ZipChunksMut chunk_a alignment violated"),
            SimdView::new(chunk_b).expect("ZipChunksMut chunk_b alignment violated"),
        ))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.chunks_remaining();
        (r, Some(r))
    }
}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    ExactSizeIterator for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
{}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    core::iter::FusedIterator for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
{}