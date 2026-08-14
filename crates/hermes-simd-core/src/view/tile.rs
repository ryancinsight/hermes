//! 2D matrix tile views for high-performance tiled and register-blocked matrix multiply kernels.

use crate::align::{Alignment, Unaligned};
use crate::scalar::NumericElement;
use core::marker::PhantomData;

/// A 2D matrix tile view, parameterized by dimensions, alignment, execution mode, and backing reference type.
///
/// Under the hood, this represents a 2D tile of shape `ROWS x COLS` with a specified row `stride`.
/// It is represented as `#[repr(C)]` to guarantee layout compatibility.
#[repr(C)]
pub struct TileView<
    'a,
    T: NumericElement,
    Backend,
    Arch,
    const ROWS: usize,
    const COLS: usize,
    Align: Alignment = Unaligned,
    Ref: 'a = &'a [T],
> {
    ptr: *mut T,
    stride: usize,
    _marker: PhantomData<(&'a T, Backend, Arch, Align, Ref)>,
}

unsafe impl<
        'a,
        T: NumericElement,
        Backend,
        Arch,
        const ROWS: usize,
        const COLS: usize,
        Align: Alignment,
        Ref: 'a,
    > Send for TileView<'a, T, Backend, Arch, ROWS, COLS, Align, Ref>
where
    Ref: Send,
{
}

unsafe impl<
        'a,
        T: NumericElement,
        Backend,
        Arch,
        const ROWS: usize,
        const COLS: usize,
        Align: Alignment,
        Ref: 'a,
    > Sync for TileView<'a, T, Backend, Arch, ROWS, COLS, Align, Ref>
where
    Ref: Sync,
{
}

impl<
        'a,
        T: NumericElement,
        Backend,
        Arch,
        const ROWS: usize,
        const COLS: usize,
        Align: Alignment,
        Ref: 'a,
    > Clone for TileView<'a, T, Backend, Arch, ROWS, COLS, Align, Ref>
where
    Ref: Clone,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            stride: self.stride,
            _marker: PhantomData,
        }
    }
}

impl<
        'a,
        T: NumericElement,
        Backend,
        Arch,
        const ROWS: usize,
        const COLS: usize,
        Align: Alignment,
        Ref: 'a,
    > Copy for TileView<'a, T, Backend, Arch, ROWS, COLS, Align, Ref>
where
    Ref: Copy,
{
}

impl<
        'a,
        T: NumericElement,
        Backend,
        Arch: crate::arch::SimdArch,
        const ROWS: usize,
        const COLS: usize,
        Align: Alignment,
    > TileView<'a, T, Backend, Arch, ROWS, COLS, Align, &'a [T]>
{
    /// Create a new read-only `TileView` after verifying bounds and alignment invariants.
    /// Returns `None` if the input slice is too small or if alignment constraints are not met.
    #[inline]
    pub fn new(data: &'a [T], stride: usize) -> Option<Self> {
        if data.len() < ROWS * stride {
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
            ptr: data.as_ptr().cast_mut(),
            stride,
            _marker: PhantomData,
        })
    }
}

impl<
        'a,
        T: NumericElement,
        Backend,
        Arch: crate::arch::SimdArch,
        const ROWS: usize,
        const COLS: usize,
        Align: Alignment,
    > TileView<'a, T, Backend, Arch, ROWS, COLS, Align, &'a mut [T]>
{
    /// Create a new mutable `TileView` after verifying bounds and alignment invariants.
    /// Returns `None` if the input slice is too small or if alignment constraints are not met.
    #[inline]
    pub fn new_mut(data: &'a mut [T], stride: usize) -> Option<Self> {
        if data.len() < ROWS * stride {
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
            ptr: data.as_mut_ptr(),
            stride,
            _marker: PhantomData,
        })
    }

    /// Access the underlying raw mutable pointer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<
        'a,
        T: NumericElement,
        Backend,
        Arch,
        const ROWS: usize,
        const COLS: usize,
        Align: Alignment,
        Ref: 'a,
    > TileView<'a, T, Backend, Arch, ROWS, COLS, Align, Ref>
{
    /// Access the underlying raw pointer.
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns the row stride of this tile view.
    #[inline(always)]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the number of rows.
    #[inline(always)]
    #[must_use]
    pub fn rows(&self) -> usize {
        ROWS
    }

    /// Returns the number of columns.
    #[inline(always)]
    #[must_use]
    pub fn cols(&self) -> usize {
        COLS
    }
}

/// Trait mediating zero-overhead matrix multiplication on 2D tiles.
///
/// Implementations are fully monomorphized to optimize layout, vectorization, and register pressure.
pub trait TileMatrixMultiply<
    TA,
    TB,
    TC,
    Backend,
    Arch,
    const M: usize,
    const N: usize,
    const K: usize,
>
{
    /// Performs tile matrix multiplication: C += A * B
    ///
    /// # Safety
    /// - Pointers `a`, `b`, and `c` must be valid for reads/writes of size M*`a_stride`, K*`b_stride`, M*`c_stride`.
    unsafe fn tile_matmul(
        c: *mut TC,
        c_stride: usize,
        a: *const TA,
        a_stride: usize,
        b: *const TB,
        b_stride: usize,
    );
}
