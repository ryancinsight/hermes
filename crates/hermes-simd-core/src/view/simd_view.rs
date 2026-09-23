//! The zero-copy [`SimdView`] typed slice view and its typestate operations.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::{ExecutionMode, Unmasked};
use crate::iter;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use core::marker::PhantomData;

use super::SimdError;

/// A zero-copy, typed slice view parameterized by architecture, alignment, execution mode, and reference typestates.
///
/// # Type Parameters
/// - `T`: scalar element type
/// - `Arch`: SIMD architecture ZST marker
/// - `Align`: alignment typestate
/// - `Mode`: execution mode (`Unmasked` or `Masked`); defaults to `Unmasked`
/// - `Ref`: reference typestate; defaults to `&'a [T]`
///
/// Guaranteed to have zero runtime overhead and remains `#[repr(transparent)]`.
#[repr(transparent)]
pub struct SimdView<
    'a,
    T: 'a,
    Arch: SimdArch,
    Align: Alignment,
    Mode: ExecutionMode = Unmasked,
    Ref: 'a = &'a [T],
> {
    // `pub(super)` reproduces this struct's pre-split effective visibility
    // exactly: these fields were private to `view` when the type was declared
    // there, so they stay reachable from `view` and its descendants (only
    // `view::casts` reads them) and nowhere else.
    pub(super) ptr: *mut [T],
    pub(super) _marker: PhantomData<(&'a T, Arch, Align, Mode, Ref)>,
}

unsafe impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode, Ref: 'a> Send
    for SimdView<'a, T, Arch, Align, Mode, Ref>
where
    Ref: Send,
{
}

unsafe impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode, Ref: 'a> Sync
    for SimdView<'a, T, Arch, Align, Mode, Ref>
where
    Ref: Sync,
{
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode> Clone
    for SimdView<'a, T, Arch, Align, Mode, &'a [T]>
{
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode> Copy
    for SimdView<'a, T, Arch, Align, Mode, &'a [T]>
{
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a [T]>
{
    /// Create a new read-only `SimdView` after verifying that `Arch` runs on
    /// this host and that the alignment invariants hold.
    ///
    /// Returns `None` when the host cannot execute `Arch` — naming a marker the
    /// CPU does not implement, such as `Avx512` on a machine without it — or
    /// when the alignment requirements are not met. Every operation on the view
    /// calls `#[target_feature]`-gated kernels, so a view that existed without
    /// that guarantee would let safe code execute unsupported instructions.
    #[inline]
    pub fn new(data: &'a [T]) -> Option<Self> {
        if !Arch::is_runtime_supported() {
            return None;
        }
        if Align::IS_ALIGNED {
            let req_align = Arch::REGISTER_WIDTH_BITS as usize / 8;
            if req_align > 0 && Align::ALIGN_BYTES < req_align {
                return None;
            }
            let addr = data.as_ptr() as usize;
            if addr % Align::ALIGN_BYTES != 0 {
                return None;
            }
        }
        Some(Self {
            ptr: core::ptr::from_ref(data).cast_mut(),
            _marker: PhantomData,
        })
    }
}

impl<'a, T: 'a, Arch: SimdArch, Mode: ExecutionMode>
    SimdView<'a, T, Arch, crate::align::Unaligned, Mode, &'a [T]>
{
    /// Construct an unaligned child view from an existing capability-bearing view.
    ///
    /// # Safety
    /// The host must support `Arch`, and `data` must remain valid for `'a`.
    #[inline(always)]
    pub(crate) unsafe fn from_supported_slice(data: &'a [T]) -> Self {
        Self {
            ptr: core::ptr::from_ref(data).cast_mut(),
            _marker: PhantomData,
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
{
    /// Create a new mutable `SimdView` after verifying that `Arch` runs on this
    /// host and that the alignment invariants hold.
    ///
    /// Returns `None` under the same conditions as [`SimdView::new`].
    #[inline]
    pub fn new_mut(data: &'a mut [T]) -> Option<Self> {
        if !Arch::is_runtime_supported() {
            return None;
        }
        if Align::IS_ALIGNED {
            let req_align = Arch::REGISTER_WIDTH_BITS as usize / 8;
            if req_align > 0 && Align::ALIGN_BYTES < req_align {
                return None;
            }
            let addr = data.as_ptr() as usize;
            if addr % Align::ALIGN_BYTES != 0 {
                return None;
            }
        }
        Some(Self {
            ptr: core::ptr::from_mut(data),
            _marker: PhantomData,
        })
    }

    /// Access the underlying raw mutable slice.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { &mut *self.ptr }
    }

    /// Downgrade the exclusive mutable view to a shared read-only view.
    #[inline(always)]
    #[must_use]
    pub fn downgrade(self) -> SimdView<'a, T, Arch, Align, Mode, &'a [T]> {
        SimdView {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch, Mode: ExecutionMode>
    SimdView<'a, T, Arch, crate::align::Unaligned, Mode, &'a mut [T]>
{
    /// Construct an unaligned mutable child view from an existing capability-bearing view.
    ///
    /// # Safety
    /// The host must support `Arch`, `data` must remain exclusively borrowed for
    /// `'a`, and no other live view may alias its elements mutably.
    #[inline(always)]
    pub(crate) unsafe fn from_supported_slice_mut(data: &'a mut [T]) -> Self {
        Self {
            ptr: core::ptr::from_mut(data),
            _marker: PhantomData,
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
{
    /// Access the underlying raw slice.
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        unsafe { &*self.ptr }
    }

    /// Returns the length of the slice.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns true if the slice is empty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Strips the static alignment guarantee of this view, returning an unaligned view zero-cost.
    #[inline(always)]
    #[must_use]
    pub fn into_unaligned(self) -> SimdView<'a, T, Arch, crate::align::Unaligned, Mode, Ref> {
        SimdView {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Attempts to promote the alignment of this view to boundary `A` bytes.
    /// Returns `Some(SimdView)` if the start pointer is aligned to `A` bytes, otherwise `None`.
    #[inline]
    #[must_use]
    pub fn try_into_aligned<const A: usize>(
        self,
    ) -> Option<SimdView<'a, T, Arch, crate::align::Aligned<A>, Mode, Ref>> {
        let req_align = Arch::REGISTER_WIDTH_BITS as usize / 8;
        if req_align > 0 && A < req_align {
            return None;
        }
        let addr = self.as_slice().as_ptr() as usize;
        if addr % A == 0 {
            Some(SimdView {
                ptr: self.ptr,
                _marker: PhantomData,
            })
        } else {
            None
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a [T]>
{
    /// Zero-copy sub-slice over a range of indices, returning an unaligned view.
    #[inline]
    #[must_use]
    pub fn slice_unaligned(
        self,
        range: core::ops::Range<usize>,
    ) -> SimdView<'a, T, Arch, crate::align::Unaligned, Mode, &'a [T]> {
        let sub = &self.as_slice()[range];
        SimdView {
            ptr: core::ptr::from_ref(sub).cast_mut(),
            _marker: PhantomData,
        }
    }

    /// Zero-copy sub-slice over a range of indices, returning an aligned view with boundary `A` bytes.
    /// Returns `Some(SimdView)` if the sub-slice satisfies the alignment, otherwise `None`.
    #[inline]
    #[must_use]
    pub fn slice_aligned<const A: usize>(
        self,
        range: core::ops::Range<usize>,
    ) -> Option<SimdView<'a, T, Arch, crate::align::Aligned<A>, Mode, &'a [T]>> {
        let req_align = Arch::REGISTER_WIDTH_BITS as usize / 8;
        if req_align > 0 && A < req_align {
            return None;
        }
        let sub = &self.as_slice()[range];
        let addr = sub.as_ptr() as usize;
        if addr % A == 0 {
            Some(SimdView {
                ptr: core::ptr::from_ref(sub).cast_mut(),
                _marker: PhantomData,
            })
        } else {
            None
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
{
    /// Zero-copy mutable sub-slice over a range of indices, returning an unaligned view.
    #[inline]
    #[must_use]
    pub fn slice_unaligned_mut(
        mut self,
        range: core::ops::Range<usize>,
    ) -> SimdView<'a, T, Arch, crate::align::Unaligned, Mode, &'a mut [T]> {
        let sub = &mut self.as_slice_mut()[range];
        SimdView {
            ptr: core::ptr::from_mut(sub),
            _marker: PhantomData,
        }
    }

    /// Zero-copy mutable sub-slice over a range of indices, returning an aligned view with boundary `A` bytes.
    /// Returns `Some(SimdView)` if the sub-slice satisfies the alignment, otherwise `None`.
    #[inline]
    #[must_use]
    pub fn slice_aligned_mut<const A: usize>(
        mut self,
        range: core::ops::Range<usize>,
    ) -> Option<SimdView<'a, T, Arch, crate::align::Aligned<A>, Mode, &'a mut [T]>> {
        let req_align = Arch::REGISTER_WIDTH_BITS as usize / 8;
        if req_align > 0 && A < req_align {
            return None;
        }
        let sub = &mut self.as_slice_mut()[range];
        let addr = sub.as_ptr() as usize;
        if addr % A == 0 {
            Some(SimdView {
                ptr: core::ptr::from_mut(sub),
                _marker: PhantomData,
            })
        } else {
            None
        }
    }
}

impl<
        'a,
        T: Scalar + 'a,
        Arch: SimdArch + SimdKernel<T>,
        Align: Alignment,
        Mode: ExecutionMode,
        Ref: 'a,
    > SimdView<'a, T, Arch, Align, Mode, Ref>
{
    /// Return a zero-copy iterator over non-overlapping `LANE_COUNT`-wide chunks.
    ///
    /// Each yielded item is a `SimdView<'a, T, Arch, Align, Mode, &'a [T]>` covering
    /// exactly `Arch::LANE_COUNT` elements. The scalar tail (elements that do not fill
    /// a complete vector) is accessible via [`iter::SimdChunks::remainder`].
    #[inline(always)]
    #[must_use]
    pub fn simd_chunks(&self) -> iter::SimdChunks<'a, T, Arch, Align, Mode> {
        // SAFETY: self.as_slice() is valid for the lifetime 'a (it derives from our ptr).
        unsafe {
            iter::SimdChunks::from_raw_parts(self.as_slice().as_ptr(), self.len(), Arch::LANE_COUNT)
        }
    }

    /// Return a zero-copy iterator that advances two views in lockstep.
    ///
    /// Iterates non-overlapping `LANE_COUNT`-wide chunk pairs from `self` and `other`
    /// until the shorter SIMD prefix is exhausted. Access the tails via
    /// [`iter::ZipChunks::remainder`].
    #[inline(always)]
    #[must_use]
    pub fn zip_chunks<'b>(
        &self,
        other: &'b SimdView<'b, T, Arch, Align, Mode, &'b [T]>,
    ) -> iter::ZipChunks<'a, 'b, T, Arch, Align, Mode> {
        // SAFETY: both slice pointers are valid for their respective lifetimes.
        unsafe {
            iter::ZipChunks::from_raw_parts(
                self.as_slice().as_ptr(),
                self.len(),
                other.as_slice().as_ptr(),
                other.len(),
            )
        }
    }
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
{
    /// Return a zero-copy mutable iterator over non-overlapping `LANE_COUNT`-wide chunks.
    ///
    /// Each yielded item is a `SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>` covering
    /// exactly `Arch::LANE_COUNT` elements. The scalar tail (elements that do not fill
    /// a complete vector) is accessible via [`iter::SimdChunksMut::into_remainder`].
    #[inline(always)]
    #[must_use]
    pub fn simd_chunks_mut(self) -> iter::SimdChunksMut<'a, T, Arch, Align, Mode> {
        // SAFETY: self.ptr is valid for writes of total elements for lifetime 'a.
        unsafe {
            iter::SimdChunksMut::from_raw_parts(self.ptr.cast::<T>(), self.len(), Arch::LANE_COUNT)
        }
    }

    /// Return a paired mutable/immutable chunk iterator (the SAXPY pattern).
    ///
    /// Advances `self` (mutable) and `other` (immutable) in lockstep by `LANE_COUNT` per step.
    /// The scalar tails are returned by [`iter::ZipChunksMut::into_remainder`].
    ///
    /// ```rust,ignore
    /// let mut chunks = view_a.zip_chunks_mut(&view_b);
    /// for (mut a_chunk, b_chunk) in &mut chunks {
    ///     a_chunk.transform_in_place(&b_chunk, Add);
    /// }
    /// let (tail_a, tail_b) = chunks.into_remainder();
    /// for (a, &b) in tail_a.iter_mut().zip(tail_b) { *a = *a + b; }
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn zip_chunks_mut<'b>(
        self,
        other: &'b SimdView<'b, T, Arch, Align, Mode, &'b [T]>,
    ) -> iter::ZipChunksMut<'a, 'b, T, Arch, Align, Mode> {
        // SAFETY: self is an exclusive mutable view for 'a; other is a shared view for 'b.
        // Non-overlap is a caller invariant (enforced by the borrow checker: `self` is `&'a mut`).
        unsafe {
            iter::ZipChunksMut::from_raw_parts(
                self.ptr.cast::<T>(),
                self.len(),
                other.as_slice().as_ptr(),
                other.len(),
            )
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode, Ref: 'a> core::ops::Deref
    for SimdView<'a, T, Arch, Align, Mode, Ref>
{
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode> core::ops::DerefMut
    for SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

#[inline(never)]
pub(crate) fn check_lengths_equal(len1: usize, len2: usize) -> Result<(), SimdError> {
    if len1 != len2 {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

#[inline(never)]
pub(crate) fn check_output_length(input_len: usize, output_len: usize) -> Result<(), SimdError> {
    if output_len < input_len {
        return Err(SimdError::InsufficientOutputLength);
    }
    Ok(())
}
