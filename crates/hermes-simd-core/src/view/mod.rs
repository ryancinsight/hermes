//! Safely typed views over slices with static alignment, architecture dispatch, reference typestates, and execution mode.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::{ExecutionMode, Unmasked};
use crate::iter;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use core::marker::PhantomData;

/// Vectorized indirect load (gather) operations.
pub mod gather;
/// Lane-masked math and compaction/expansion operations on SIMD views.
pub mod masked;
/// Standard elementwise and accumulation operations on SIMD views.
pub mod ops;
/// Standard exclusive mutable elementwise operations on SIMD views.
pub mod ops_mut;
/// Unrolled generic horizontal reductions on SIMD views.
pub mod reduce;
/// Inclusive/exclusive prefix scans and running min/max.
pub mod scan;
/// Vectorized indirect store (scatter) operations.
pub mod scatter;
/// Lane-wise conditional select and masked-negate.
pub mod select;
/// 2D matrix tile views and operations.
pub mod tile;
/// Unary mapping operations on SIMD views.
pub mod unary;

pub use tile::{TileMatrixMultiply, TileView};

/// Module containing the SIMD mask register wrappers.
pub mod mask_reg;
/// Module containing operator overload implementations for SIMD vectors.
pub mod vector_ops;
/// Module containing the generic SIMD vector register wrappers.
pub mod vector_reg;

pub use mask_reg::Mask;
pub use vector_reg::Vector;

/// Error types for SIMD view operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdError {
    /// The lengths of the operand views do not match.
    LengthMismatch,
    /// The input slice is too small to load the requested vector.
    InsufficientInputLength,
    /// The output slice is too small to store the results.
    InsufficientOutputLength,
    /// The memory address is not aligned as required.
    UnalignedAddress,
    /// An index is out of bounds of the view.
    IndexOutOfBounds,
    /// The current host cannot execute the requested SIMD target safely.
    UnsupportedTarget,
}

impl core::fmt::Display for SimdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch => write!(f, "Operand views have mismatched lengths"),
            Self::InsufficientInputLength => write!(f, "Input slice has insufficient length"),
            Self::InsufficientOutputLength => write!(f, "Output slice has insufficient length"),
            Self::UnalignedAddress => {
                write!(f, "Memory address does not satisfy alignment constraints")
            }
            Self::IndexOutOfBounds => write!(f, "Index is out of bounds of the view"),
            Self::UnsupportedTarget => {
                write!(f, "SIMD target is not supported or enabled on this host")
            }
        }
    }
}

// Implement standard Error trait if std is available.
#[cfg(feature = "std")]
impl std::error::Error for SimdError {}

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
    ptr: *mut [T],
    _marker: PhantomData<(&'a T, Arch, Align, Mode, Ref)>,
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
            ptr: data as *const [T] as *mut [T],
            _marker: PhantomData,
        })
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
            ptr: data as *mut [T],
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
    pub fn downgrade(self) -> SimdView<'a, T, Arch, Align, Mode, &'a [T]> {
        SimdView {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
{
    /// Access the underlying raw slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { &*self.ptr }
    }

    /// Returns the length of the slice.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns true if the slice is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Strips the static alignment guarantee of this view, returning an unaligned view zero-cost.
    #[inline(always)]
    pub fn into_unaligned(self) -> SimdView<'a, T, Arch, crate::align::Unaligned, Mode, Ref> {
        SimdView {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Attempts to promote the alignment of this view to boundary `A` bytes.
    /// Returns `Some(SimdView)` if the start pointer is aligned to `A` bytes, otherwise `None`.
    #[inline]
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
    pub fn slice_unaligned(
        self,
        range: core::ops::Range<usize>,
    ) -> SimdView<'a, T, Arch, crate::align::Unaligned, Mode, &'a [T]> {
        let sub = &self.as_slice()[range];
        SimdView {
            ptr: sub as *const [T] as *mut [T],
            _marker: PhantomData,
        }
    }

    /// Zero-copy sub-slice over a range of indices, returning an aligned view with boundary `A` bytes.
    /// Returns `Some(SimdView)` if the sub-slice satisfies the alignment, otherwise `None`.
    #[inline]
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
                ptr: sub as *const [T] as *mut [T],
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
    pub fn slice_unaligned_mut(
        mut self,
        range: core::ops::Range<usize>,
    ) -> SimdView<'a, T, Arch, crate::align::Unaligned, Mode, &'a mut [T]> {
        let sub = &mut self.as_slice_mut()[range];
        SimdView {
            ptr: sub as *mut [T],
            _marker: PhantomData,
        }
    }

    /// Zero-copy mutable sub-slice over a range of indices, returning an aligned view with boundary `A` bytes.
    /// Returns `Some(SimdView)` if the sub-slice satisfies the alignment, otherwise `None`.
    #[inline]
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
                ptr: sub as *mut [T],
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
    /// Return a zero-copy iterator over non-overlapping `LANE_COUNT`-wide sub-views.
    ///
    /// Each yielded item is a `SimdView<'a, T, Arch, Align, Mode, &'a [T]>` covering
    /// exactly `Arch::LANE_COUNT` elements. The scalar tail (elements that do not fill
    /// a complete vector) is accessible via [`iter::SimdChunks::remainder`].
    #[inline(always)]
    pub fn simd_chunks(&self) -> iter::SimdChunks<'a, T, Arch, Align, Mode> {
        // SAFETY: self.as_slice() is valid for the lifetime 'a (it derives from our ptr).
        unsafe {
            iter::SimdChunks::from_raw_parts(self.as_slice().as_ptr(), self.len(), Arch::LANE_COUNT)
        }
    }

    /// Return a zero-copy iterator that advances two views in lockstep.
    ///
    /// Iterates non-overlapping `LANE_COUNT`-wide pairs of sub-views from `self` and `other`
    /// until the shorter SIMD prefix is exhausted. Access the tails via
    /// [`iter::ZipChunks::remainder`].
    #[inline(always)]
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
    /// Return a zero-copy mutable iterator over non-overlapping `LANE_COUNT`-wide sub-views.
    ///
    /// Each yielded item is a `SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>` covering
    /// exactly `Arch::LANE_COUNT` elements. The scalar tail (elements that do not fill
    /// a complete vector) is accessible via [`iter::SimdChunksMut::into_remainder`].
    #[inline(always)]
    pub fn simd_chunks_mut(self) -> iter::SimdChunksMut<'a, T, Arch, Align, Mode> {
        // SAFETY: self.ptr is valid for writes of total elements for lifetime 'a.
        unsafe {
            iter::SimdChunksMut::from_raw_parts(self.ptr as *mut T, self.len(), Arch::LANE_COUNT)
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
    pub fn zip_chunks_mut<'b>(
        self,
        other: &'b SimdView<'b, T, Arch, Align, Mode, &'b [T]>,
    ) -> iter::ZipChunksMut<'a, 'b, T, Arch, Align, Mode> {
        // SAFETY: self is an exclusive mutable view for 'a; other is a shared view for 'b.
        // Non-overlap is a caller invariant (enforced by the borrow checker: `self` is `&'a mut`).
        unsafe {
            iter::ZipChunksMut::from_raw_parts(
                self.ptr as *mut T,
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

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a [T]>
where
    T: bytemuck::Pod,
{
    /// Safe cast of the underlying data slice to a slice of another Pod type, returning a new `SimdView`.
    #[inline]
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
    pub fn cast_mut<U: bytemuck::Pod>(
        self,
    ) -> Option<SimdView<'a, U, Arch, Align, Mode, &'a mut [U]>> {
        let casted = bytemuck::try_cast_slice_mut(unsafe { &mut *self.ptr }).ok()?;
        SimdView::new_mut(casted)
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
