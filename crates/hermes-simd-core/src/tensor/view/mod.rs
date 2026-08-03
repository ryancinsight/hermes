//! Zero-copy N-dimensional strided tensor view.
//!
//! [`TensorView`] is the core rank-`N` view over a borrowed slice.  Shape and strides are
//! `[usize; N]` arrays resolved at compile time — the const generic `N` is erased after
//! monomorphization, leaving no runtime overhead vs. a hand-written 2-D or 3-D struct.
//!
//! This module is split by concern:
//! - `mod.rs` (here): the core struct, marker impls, constructors, and the
//!   rank-agnostic shape/stride/element accessors.
//! - `rank_ops` (private): rank-2 and rank-3 specialized views (rows, columns,
//!   diagonal, `matrix_at`, transpose).
//! - `simd_bridge` (private): the zero-copy rank-1 → `SimdView` seam.

use core::marker::PhantomData;

use super::error::TensorError;
use super::layout::{ColMajor, Layout, RowMajor};
use super::strides::{compute_offset, row_major_strides};

mod rank_ops;
mod simd_bridge;

// ---------------------------------------------------------------------------
// Core struct
// ---------------------------------------------------------------------------

/// Zero-copy N-dimensional strided view over a borrowed slice.
///
/// # Type Parameters
/// - `'a` — lifetime of the underlying data slice.
/// - `T` — element type.
/// - `N` — tensor rank (number of dimensions). Const generic; resolved at compile time.
/// - `Layout` — layout marker ZST ([`RowMajor`] or [`ColMajor`]). `PhantomData`; zero size.
/// - `Ref` — reference type-state (`&'a [T]` or `&'a mut [T]`).
///
/// # Invariants
/// - `strides[i] * shape[i]` must not overflow `usize`.
/// - `data.len() >= ∑ (shape[i]-1) * strides[i] + 1` for any valid element access.
/// - `new` and `with_strides` both verify `product(shape) <= data.len()`.
pub struct TensorView<'a, T: 'a, const N: usize, Layout = RowMajor, Ref = &'a [T]> {
    pub(super) ptr: *mut [T],
    pub(super) shape: [usize; N],
    pub(super) strides: [usize; N],
    pub(super) _layout: PhantomData<(&'a T, Layout, Ref)>,
}

// ---------------------------------------------------------------------------
// Send / Sync / Clone / Copy
// ---------------------------------------------------------------------------

unsafe impl<'a, T, const N: usize, Layout, Ref> Send for TensorView<'a, T, N, Layout, Ref> where
    Ref: Send
{
}

unsafe impl<'a, T, const N: usize, Layout, Ref> Sync for TensorView<'a, T, N, Layout, Ref> where
    Ref: Sync
{
}

impl<'a, T, const N: usize, Layout> Clone for TensorView<'a, T, N, Layout, &'a [T]> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, const N: usize, Layout> Copy for TensorView<'a, T, N, Layout, &'a [T]> {}

// ---------------------------------------------------------------------------
// Row-major immutable constructor
// ---------------------------------------------------------------------------

impl<'a, 'b, T, const N: usize> TensorView<'a, T, N, RowMajor, &'b [T]> {
    /// Create a row-major tensor view over `data` with the given `shape`.
    ///
    /// Strides are computed as `strides[i] = ∏_{j=i+1..N} shape[j]` (C-order).
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `∏ shape > data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hermes_simd_core::TensorView;
    ///
    /// let data = [0, 1, 2, 3, 4, 5];
    /// let view = TensorView::<i32, 2>::new(&data, [2, 3]).unwrap();
    ///
    /// assert_eq!(view.shape(), [2, 3]);
    /// assert_eq!(view.strides(), [3, 1]);
    /// assert_eq!(view.get([1, 2]).unwrap(), 5);
    /// ```
    #[inline]
    pub fn new(data: &'b [T], shape: [usize; N]) -> Result<Self, TensorError> {
        let elem_count = shape.iter().product::<usize>();
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        let strides = row_major_strides(shape);
        Ok(Self {
            ptr: data as *const [T] as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Row-major mutable constructor
// ---------------------------------------------------------------------------

impl<'a, 'b, T, const N: usize> TensorView<'a, T, N, RowMajor, &'b mut [T]> {
    /// Create a mutable row-major tensor view over `data` with the given `shape`.
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `∏ shape > data.len()`.
    #[inline]
    pub fn new_mut(data: &'b mut [T], shape: [usize; N]) -> Result<Self, TensorError> {
        let elem_count = shape.iter().product::<usize>();
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        let strides = row_major_strides(shape);
        Ok(Self {
            ptr: data as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Explicit-stride constructors (immutable + mutable)
// ---------------------------------------------------------------------------

impl<'a, 'b, T, const N: usize, L: Layout> TensorView<'a, T, N, L, &'b [T]> {
    /// Create a tensor view with explicit strides.
    ///
    /// Allows column-major, blocked, or any custom layout.
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `∏ shape > data.len()`.
    #[inline]
    pub fn with_strides(
        data: &'b [T],
        shape: [usize; N],
        strides: [usize; N],
    ) -> Result<Self, TensorError> {
        let elem_count = shape.iter().product::<usize>();
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        Ok(Self {
            ptr: data as *const [T] as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

impl<'a, 'b, T, const N: usize, L: Layout> TensorView<'a, T, N, L, &'b mut [T]> {
    /// Create a mutable tensor view with explicit strides.
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `∏ shape > data.len()`.
    #[inline]
    pub fn with_strides_mut(
        data: &'b mut [T],
        shape: [usize; N],
        strides: [usize; N],
    ) -> Result<Self, TensorError> {
        let elem_count = shape.iter().product::<usize>();
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        Ok(Self {
            ptr: data as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }

    /// Downgrade the exclusive mutable view to a shared read-only view.
    #[inline(always)]
    pub fn downgrade(self) -> TensorView<'a, T, N, L, &'b [T]> {
        TensorView {
            ptr: self.ptr,
            shape: self.shape,
            strides: self.strides,
            _layout: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// ColMajor ergonomic constructor
// ---------------------------------------------------------------------------

impl<'a, 'b, T> TensorView<'a, T, 2, ColMajor, &'b [T]> {
    /// Create a column-major (Fortran-order) 2-D tensor view.
    ///
    /// Fortran strides: `strides[0] = 1`, `strides[1] = shape[0]`.
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `shape[0] * shape[1] > data.len()`.
    #[inline]
    pub fn new_col_major(data: &'b [T], shape: [usize; 2]) -> Result<Self, TensorError> {
        let elem_count = shape[0] * shape[1];
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        // Fortran strides: strides[0] = 1 (column-stride), strides[1] = nrows (row-stride).
        let strides = [1, shape[0]];
        Ok(Self {
            ptr: data as *const [T] as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Shape / stride / element accessors (all Ref variants)
// ---------------------------------------------------------------------------

impl<'a, T, const N: usize, L, Ref> TensorView<'a, T, N, L, Ref> {
    /// The logical shape of this tensor: number of elements per dimension.
    #[inline(always)]
    pub fn shape(&self) -> [usize; N] {
        self.shape
    }

    /// The strides of this tensor in element units.
    #[inline(always)]
    pub fn strides(&self) -> [usize; N] {
        self.strides
    }

    /// Number of elements in this tensor: `∏ shape[i]`.
    #[inline]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns `true` if the tensor is empty (one of its dimensions is 0).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_elements() == 0
    }

    /// Whether this view is contiguous in row-major order.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        let expected = row_major_strides(self.shape);
        self.strides == expected
    }

    /// View the underlying flat slice (in storage order).
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { &*self.ptr }
    }

    /// Bounds-checked element access.
    #[inline]
    pub fn get(&self, idx: [usize; N]) -> Result<T, TensorError>
    where
        T: Copy,
    {
        for i in 0..N {
            if idx[i] >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds);
            }
        }
        let offset = compute_offset(&idx, &self.strides);
        Ok(self.as_slice()[offset])
    }

    /// Unchecked element access.
    ///
    /// # Safety
    /// `idx[i] < shape[i]` for all `i`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, idx: [usize; N]) -> T
    where
        T: Copy,
    {
        let offset = compute_offset(&idx, &self.strides);
        *self.as_slice().get_unchecked(offset)
    }

    /// Reshape this view to a different rank `M`, reusing the same flat slice.
    #[inline]
    pub fn reshape<const M: usize>(
        self,
        new_shape: [usize; M],
    ) -> Result<TensorView<'a, T, M, RowMajor, Ref>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        let old_count: usize = self.shape.iter().product();
        let new_count: usize = new_shape.iter().product();
        if old_count != new_count {
            return Err(TensorError::ShapeMismatch);
        }
        let strides = row_major_strides(new_shape);
        Ok(TensorView {
            ptr: self.ptr,
            shape: new_shape,
            strides,
            _layout: PhantomData,
        })
    }
}

// ---------------------------------------------------------------------------
// Mutable element access
// ---------------------------------------------------------------------------

impl<'a, 'b, T, const N: usize, L> TensorView<'a, T, N, L, &'b mut [T]> {
    /// Access the underlying flat mutable slice.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { &mut *self.ptr }
    }

    /// Bounds-checked element write access.
    #[inline]
    pub fn set(&mut self, idx: [usize; N], val: T) -> Result<(), TensorError> {
        for i in 0..N {
            if idx[i] >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds);
            }
        }
        let offset = compute_offset(&idx, &self.strides);
        self.as_slice_mut()[offset] = val;
        Ok(())
    }

    /// Unchecked element write access.
    ///
    /// # Safety
    /// `idx[i] < shape[i]` for all `i`.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, idx: [usize; N], val: T) {
        let offset = compute_offset(&idx, &self.strides);
        *self.as_slice_mut().get_unchecked_mut(offset) = val;
    }
}
