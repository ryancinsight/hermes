//! Safely typed views over slices with static alignment, architecture dispatch, reference typestates, and execution mode.

use core::marker::PhantomData;
use crate::arch::SimdArch;
use crate::align::Alignment;
use crate::kernel::SimdKernel;
use crate::execution::{ExecutionMode, Unmasked};
use crate::scalar::Scalar;
use crate::iter;

/// Standard elementwise and accumulation operations on SIMD views.
pub mod ops;
/// Lane-masked math and compaction/expansion operations on SIMD views.
pub mod masked;
/// Unrolled generic horizontal reductions on SIMD views.
pub mod reduce;
/// 2D matrix tile views and operations.
pub mod tile;

pub use tile::{TileView, TileMatrixMultiply};

/// Module containing the generic SIMD vector register wrappers.
pub mod vector_reg;
/// Module containing operator overload implementations for SIMD vectors.
pub mod vector_ops;
/// Module containing the SIMD mask register wrappers.
pub mod mask_reg;

pub use vector_reg::Vector;
pub use mask_reg::Mask;

/// Error types for SIMD view operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdError {
    /// The lengths of the operand views do not match.
    LengthMismatch,
    /// The output slice is too small to store the results.
    InsufficientOutputLength,
    /// The memory address is not aligned as required.
    UnalignedAddress,
}

impl core::fmt::Display for SimdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch => write!(f, "Operand views have mismatched lengths"),
            Self::InsufficientOutputLength => write!(f, "Output slice has insufficient length"),
            Self::UnalignedAddress => write!(f, "Memory address does not satisfy alignment constraints"),
        }
    }
}

// Implement standard Error trait if std is available.
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
pub struct SimdView<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode = Unmasked, Ref: 'a = &'a [T]> {
    ptr: *mut [T],
    _marker: PhantomData<(&'a T, Arch, Align, Mode, Ref)>,
}

unsafe impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode, Ref: 'a> Send
    for SimdView<'a, T, Arch, Align, Mode, Ref>
where
    Ref: Send,
{}

unsafe impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode, Ref: 'a> Sync
    for SimdView<'a, T, Arch, Align, Mode, Ref>
where
    Ref: Sync,
{}

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
{}

impl<'a, T: 'a, Arch: SimdArch, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a [T]>
{
    /// Create a new read-only `SimdView` after verifying alignment invariants.
    /// Returns `None` if the alignment requirements are not met.
    #[inline]
    pub fn new(data: &'a [T]) -> Option<Self> {
        if Align::IS_ALIGNED {
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
    /// Create a new mutable `SimdView` after verifying alignment invariants.
    /// Returns `None` if the alignment requirements are not met.
    #[inline]
    pub fn new_mut(data: &'a mut [T]) -> Option<Self> {
        if Align::IS_ALIGNED {
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
}

impl<'a, T: Scalar + 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
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
            iter::SimdChunks::from_raw_parts(
                self.as_slice().as_ptr(),
                self.len(),
                Arch::LANE_COUNT,
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
            iter::SimdChunksMut::from_raw_parts(
                self.ptr as *mut T,
                self.len(),
                Arch::LANE_COUNT,
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
    pub fn cast_mut<U: bytemuck::Pod>(self) -> Option<SimdView<'a, U, Arch, Align, Mode, &'a mut [U]>> {
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
