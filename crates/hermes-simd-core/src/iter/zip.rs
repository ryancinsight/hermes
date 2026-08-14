//! `ZipChunks` and `ZipChunksMut` — paired SIMD chunk iterators for two views.
//!
//! `ZipChunks` advances two immutable views in lockstep; `ZipChunksMut` pairs a
//! mutable first operand with an immutable second operand for in-place transforms.
//!
//! # Safety
//!
//! Every kernel call below is `#[target_feature]`-gated and is therefore sound
//! only on a host implementing `Arch`. That holds by construction rather than by
//! inspection: [`SimdView::new`](crate::view::SimdView::new) returns `None` for
//! an architecture the host cannot execute, and the sparse and copy-on-write
//! constructors assert the same condition, so possessing one of these
//! arch-parameterized values *is* the proof. Per-site `SAFETY` comments record
//! only the obligations that go beyond it — pointer provenance, bounds, and
//! alignment.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::view::SimdView;

// ---------------------------------------------------------------------------
// ZipChunks — paired immutable/immutable
// ---------------------------------------------------------------------------

/// Iterator over non-overlapping paired `LANE_COUNT`-wide sub-views of two `SimdView`s.
///
/// Advances both views in lockstep, yielding pairs of chunks until the shorter
/// view's SIMD prefix is exhausted.  Access the tails via [`ZipChunks::remainder`].
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
{
}
unsafe impl<'a, 'b, T: Sync, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    Sync for ZipChunks<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar,
{
}

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
    #[inline(always)]
    #[must_use]
    pub fn remainder(&self) -> (&'a [T], &'b [T]) {
        // SAFETY: base + simd_end is within bounds by construction.
        unsafe {
            (
                core::slice::from_raw_parts(
                    self.base_a.add(self.simd_end),
                    self.total_a - self.simd_end,
                ),
                core::slice::from_raw_parts(
                    self.base_b.add(self.simd_end),
                    self.total_b - self.simd_end,
                ),
            )
        }
    }

    /// Returns the number of complete paired SIMD chunks remaining.
    #[inline(always)]
    #[must_use]
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
{
}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    core::iter::FusedIterator for ZipChunks<'a, 'b, T, Arch, Align, Mode>
{
}

// ---------------------------------------------------------------------------
// ZipChunksMut — paired mutable/immutable
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
///     chunk_a.transform_in_place(&chunk_b, FmaAdd);
/// }
/// let (tail_a, tail_b) = chunks.into_remainder();
/// ```
///
/// # Zero-Cost Guarantee
///
/// `ZipChunksMut` stores one `*mut T`, one `*const T`, and three `usize` — 5 words total.
/// No heap allocation. `LANE_COUNT` is a compile-time constant.
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
    _marker: core::marker::PhantomData<(&'a mut T, &'b T, Arch, Align, Mode)>,
}

// SAFETY: `ZipChunksMut` holds exclusive (`*mut T`) access to `'a` data and shared
// (`*const T`) access to `'b` data. Forwarding Send/Sync is sound when `T: Send + Sync`.
unsafe impl<'a, 'b, T, Arch, Align, Mode> Send for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar + Send + Sync,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
}
unsafe impl<'a, 'b, T, Arch, Align, Mode> Sync for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
where
    T: Scalar + Send + Sync,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
{
}

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
            _marker: core::marker::PhantomData,
        }
    }

    /// Consume the iterator and return the scalar tail slices.
    ///
    /// Elements `[simd_end..total]` for each operand. Call this **after** the loop.
    #[inline(always)]
    #[must_use]
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
    #[must_use]
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
{
}

impl<'a, 'b, T: Scalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    core::iter::FusedIterator for ZipChunksMut<'a, 'b, T, Arch, Align, Mode>
{
}
