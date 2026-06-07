//! Zero-copy, const-generic N-dimensional strided tensor view.
//!
//! # Design
//!
//! `TensorView<'a, T, const N: usize>` is a rank-`N` view over a borrowed slice.
//! Shape and strides are `[usize; N]` arrays resolved at compile time — the const
//! generic `N` is erased after monomorphization, leaving no runtime overhead vs.
//! a hand-written 2-D or 3-D struct.
//!
//! # Layout Markers
//!
//! Two zero-sized layout markers tag contiguous storage assumptions:
//! - [`RowMajor`] — row-major (C-order) storage; `strides[i] = ∏_{j>i} shape[j]`.
//! - [`ColMajor`] — column-major (Fortran-order) storage.
//!
//! # Zero-Copy Contract
//!
//! - `new(data, shape)` — zero allocation; computes row-major strides from shape.
//! - `with_strides(data, shape, strides)` — zero allocation; caller supplies strides.
//! - `row_view(i)` — returns a `TensorView<'_, T, {N-1}>` sharing the same slice.
//! - `reshape(new_shape)` — returns a new view if the layout is contiguous; no copy.
//! - All `get` / `iter_rows` operations are also zero-copy.

use core::marker::PhantomData;
use crate::align::{Alignment, Unaligned};
use crate::vec::AlignedVec;

/// Row-major (C-order) layout marker ZST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMajor;

/// Column-major (Fortran-order) layout marker ZST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColMajor;

/// Sealed marker trait for tensor layout ZSTs.
///
/// External crates cannot implement this trait — the `crate::private::Sealed` supertrait
/// is `pub(crate)` only. Only `RowMajor` and `ColMajor` satisfy `Layout`.
/// `TensorView<_, _, _, L>` requires `L: Layout`, preventing accidental use of
/// arbitrary unit types as layout parameters.
pub trait Layout: crate::private::Sealed + Copy + 'static {}

impl crate::private::Sealed for RowMajor {}
impl crate::private::Sealed for ColMajor {}
impl Layout for RowMajor {}
impl Layout for ColMajor {}

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
    ptr:     *mut [T],
    shape:   [usize; N],
    strides: [usize; N],
    _layout: PhantomData<(&'a T, Layout, Ref)>,
}

unsafe impl<'a, T, const N: usize, Layout, Ref> Send for TensorView<'a, T, N, Layout, Ref>
where
    Ref: Send,
{}

unsafe impl<'a, T, const N: usize, Layout, Ref> Sync for TensorView<'a, T, N, Layout, Ref>
where
    Ref: Sync,
{}

impl<'a, T, const N: usize, Layout> Clone for TensorView<'a, T, N, Layout, &'a [T]> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, const N: usize, Layout> Copy for TensorView<'a, T, N, Layout, &'a [T]> {}

/// Error type for tensor construction and indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// The element count derived from `shape` does not fit in `data.len()`.
    ShapeMismatch,
    /// A row or slice index is out of bounds.
    IndexOutOfBounds,
    /// The view is not contiguous and cannot be reshaped without copying.
    NotContiguous,
}

impl<'a, 'b, T, const N: usize> TensorView<'a, T, N, RowMajor, &'b [T]> {
    /// Create a row-major tensor view over `data` with the given `shape`.
    ///
    /// Strides are computed as `strides[i] = ∏_{j=i+1..N} shape[j]` (C-order).
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `∏ shape > data.len()`.
    #[inline]
    pub fn new(data: &'b [T], shape: [usize; N]) -> Result<Self, TensorError> {
        let elem_count = shape.iter().product::<usize>();
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        let strides = row_major_strides(shape);
        Ok(Self {
            ptr:     data as *const [T] as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

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
            ptr:     data as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

impl<'a, 'b, T, const N: usize, Layout: self::Layout> TensorView<'a, T, N, Layout, &'b [T]> {
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
            ptr:     data as *const [T] as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

impl<'a, 'b, T, const N: usize, Layout: self::Layout> TensorView<'a, T, N, Layout, &'b mut [T]> {
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
            ptr:     data as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }

    /// Downgrade the exclusive mutable view to a shared read-only view.
    #[inline(always)]
    pub fn downgrade(self) -> TensorView<'a, T, N, Layout, &'b [T]> {
        TensorView {
            ptr:     self.ptr,
            shape:   self.shape,
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
            ptr:     data as *const [T] as *mut [T],
            shape,
            strides,
            _layout: PhantomData,
        })
    }
}

impl<'a, T, const N: usize, Layout, Ref> TensorView<'a, T, N, Layout, Ref> {
    /// The logical shape of this tensor: number of elements per dimension.
    #[inline(always)]
    pub fn shape(&self) -> [usize; N] { self.shape }

    /// The strides of this tensor in element units.
    #[inline(always)]
    pub fn strides(&self) -> [usize; N] { self.strides }

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
            ptr:     self.ptr,
            shape:   new_shape,
            strides,
            _layout: PhantomData,
        })
    }
}

impl<'a, 'b, T, const N: usize, Layout> TensorView<'a, T, N, Layout, &'b mut [T]> {
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

// ---------------------------------------------------------------------------
// 2-D specific methods — read-only
// ---------------------------------------------------------------------------

impl<'a, 'b, T, Layout> TensorView<'a, T, 2, Layout, &'b [T]> {
    /// Return a zero-copy 1-D view of row `i`.
    ///
    /// # Errors
    /// Returns [`TensorError::IndexOutOfBounds`] if `i >= shape[0]`.
    #[inline]
    pub fn row_view(&self, i: usize) -> Result<TensorView<'a, T, 1, RowMajor, &'b [T]>, TensorError> {
        if i >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let row_start = i * self.strides[0];
        let row_len   = self.shape[1];
        // SAFETY: ptr lives for 'b (it was constructed from &'b [T])
        let full_slice: &'b [T] = unsafe { &*self.ptr };
        let row_data = &full_slice[row_start..row_start + row_len];
        TensorView::new(row_data, [row_len])
    }

    /// Iterate over row slices of a contiguous 2-D tensor.
    ///
    /// # Errors
    /// Returns [`TensorError::NotContiguous`] if the view is not row-major contiguous.
    #[inline]
    pub fn iter_rows(&self) -> Result<impl Iterator<Item = &'b [T]>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        let ncols = self.shape[1];
        // SAFETY: ptr lives for 'b (it was constructed from &'b [T])
        let full_slice: &'b [T] = unsafe { &*self.ptr };
        Ok((0..self.shape[0]).map(move |i| {
            let start = i * ncols;
            &full_slice[start..start + ncols]
        }))
    }

    /// Return a zero-copy transposed view (swaps shape and strides).
    ///
    /// The result is `ColMajor`-tagged. No allocation.
    #[inline]
    pub fn transpose_view(&self) -> TensorView<'a, T, 2, ColMajor, &'b [T]> {
        TensorView {
            ptr:     self.ptr,
            shape:   [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
            _layout: PhantomData,
        }
    }

    /// Iterate over elements of column `j`, one element per row.
    ///
    /// Zero-copy: index arithmetic over the underlying slice.
    ///
    /// # Errors
    /// Returns [`TensorError::IndexOutOfBounds`] if `j >= shape[1]`.
    #[inline]
    pub fn col_iter(&self, j: usize) -> Result<impl Iterator<Item = T> + '_, TensorError>
    where
        T: Copy,
    {
        if j >= self.shape[1] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let nrows   = self.shape[0];
        let row_str = self.strides[0];
        let slice   = self.as_slice();
        Ok((0..nrows).map(move |i| slice[j + i * row_str]))
    }

    /// Iterate over main diagonal elements (min(rows, cols) elements).
    ///
    /// Zero-copy: index arithmetic, no allocation.
    #[inline]
    pub fn diag_iter(&self) -> impl Iterator<Item = T> + '_
    where
        T: Copy,
    {
        let diag_len = self.shape[0].min(self.shape[1]);
        let diag_str = self.strides[0] + self.strides[1];
        let slice    = self.as_slice();
        (0..diag_len).map(move |i| slice[i * diag_str])
    }
}

// ---------------------------------------------------------------------------
// 2-D specific methods — mutable
// ---------------------------------------------------------------------------

impl<'a, 'b, T, Layout> TensorView<'a, T, 2, Layout, &'b mut [T]> {
    /// Return a mutable 1-D view of row `i`.
    ///
    /// # Errors
    /// Returns [`TensorError::IndexOutOfBounds`] if `i >= shape[0]`.
    #[inline]
    pub fn row_view_mut(&mut self, i: usize) -> Result<TensorView<'a, T, 1, RowMajor, &mut [T]>, TensorError> {
        if i >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let row_start = i * self.strides[0];
        let row_len   = self.shape[1];
        let slice = unsafe { &mut *self.ptr };
        let row_data  = &mut slice[row_start..row_start + row_len];
        TensorView::new_mut(row_data, [row_len])
    }

    /// Iterate over mutable row slices of a contiguous 2-D tensor.
    ///
    /// # Errors
    /// Returns [`TensorError::NotContiguous`] if not row-major contiguous.
    #[inline]
    pub fn iter_rows_mut(&mut self) -> Result<impl Iterator<Item = &mut [T]>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        let ncols = self.shape[1];
        let slice = unsafe { &mut *self.ptr };
        Ok(slice.chunks_exact_mut(ncols))
    }
}

// ---------------------------------------------------------------------------
// 3-D specific methods — matrix_at (batch slice)
// ---------------------------------------------------------------------------

impl<'a, T, Layout> TensorView<'a, T, 3, Layout, &'a [T]> {
    /// Return a 2-D view of the `b`-th matrix in a batched tensor.
    #[inline]
    pub fn matrix_at(&self, b: usize) -> Result<TensorView<'_, T, 2, RowMajor, &'_ [T]>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        if b >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let rows  = self.shape[1];
        let cols  = self.shape[2];
        let start = b * rows * cols;
        let full_slice: &[T] = self.as_slice();
        let slice = &full_slice[start..start + rows * cols];
        TensorView::<'_, T, 2, RowMajor, &'_ [T]>::new(slice, [rows, cols])
    }
}

impl<'a, T, Layout> TensorView<'a, T, 3, Layout, &'a mut [T]> {
    /// Return a mutable 2-D view of the `b`-th matrix in a batched tensor.
    #[inline]
    pub fn matrix_at_mut(&mut self, b: usize) -> Result<TensorView<'_, T, 2, RowMajor, &'_ mut [T]>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        if b >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let rows  = self.shape[1];
        let cols  = self.shape[2];
        let start = b * rows * cols;
        let slice = unsafe { &mut *self.ptr };
        let sub_slice = &mut slice[start..start + rows * cols];
        TensorView::<'_, T, 2, RowMajor, &'_ mut [T]>::new_mut(sub_slice, [rows, cols])
    }
}

// ---------------------------------------------------------------------------
// Transpose implementations
// ---------------------------------------------------------------------------

impl<'a, T, Ref> TensorView<'a, T, 2, RowMajor, Ref> {
    /// Transpose a row-major 2-D tensor view to column-major.
    #[inline]
    pub fn transpose(self) -> TensorView<'a, T, 2, ColMajor, Ref> {
        TensorView {
            ptr:     self.ptr,
            shape:   [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
            _layout: PhantomData,
        }
    }
}

impl<'a, T, Ref> TensorView<'a, T, 2, ColMajor, Ref> {
    /// Transpose a column-major 2-D tensor view to row-major.
    #[inline]
    pub fn transpose(self) -> TensorView<'a, T, 2, RowMajor, Ref> {
        TensorView {
            ptr:     self.ptr,
            shape:   [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
            _layout: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// into_simd_view — zero-copy bridge from rank-1 TensorView to SimdView
// ---------------------------------------------------------------------------

use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::execution::Unmasked;
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
    pub fn into_simd_view<Arch, Align>(
        &self,
    ) -> Option<SimdView<'a, T, Arch, Align, Unmasked, &'a [T]>>
    where
        Arch: SimdArch + SimdKernel<T>,
        Align: Alignment,
    {
        // SAFETY: as_slice() returns a reference valid for 'a derived from the tensor's
        // original borrow; SimdView::new borrows it again for the same 'a.
        let slice: &'a [T] = unsafe {
            // The ptr was set from a &'a [T] originally — reconstruct the full-lifetime ref.
            &*self.ptr
        };
        SimdView::new(slice)
    }
}

// ---------------------------------------------------------------------------
// TensorCow: Clone-on-Write tensor container
// ---------------------------------------------------------------------------

/// A Clone-on-Write (CoW) container for strided tensors.
pub enum TensorCow<'a, T: 'a, const N: usize, Layout = RowMajor, Align: Alignment = Unaligned> {
    /// Borrowed read-only tensor view.
    Borrowed(TensorView<'a, T, N, Layout, &'a [T]>),
    /// Owned aligned tensor buffer.
    Owned {
        /// Underlying aligned memory.
        data: AlignedVec<T, Align>,
        /// Logical shape of the tensor.
        shape: [usize; N],
        /// Dimension strides.
        strides: [usize; N],
    },
}

impl<'a, T: Copy + 'a, const N: usize, Layout: self::Layout, Align> TensorCow<'a, T, N, Layout, Align>
where
    Align: Alignment,
{
    /// Create a borrowed `TensorCow` wrapping a `TensorView`.
    #[inline]
    pub fn borrowed(view: TensorView<'a, T, N, Layout, &'a [T]>) -> Self {
        Self::Borrowed(view)
    }

    /// Create an owned `TensorCow` from an `AlignedVec` and shape.
    #[inline]
    pub fn owned(data: AlignedVec<T, Align>, shape: [usize; N]) -> Self {
        let strides = row_major_strides(shape);
        Self::Owned { data, shape, strides }
    }

    /// Create an owned `TensorCow` with explicit strides.
    #[inline]
    pub fn owned_with_strides(data: AlignedVec<T, Align>, shape: [usize; N], strides: [usize; N]) -> Self {
        Self::Owned { data, shape, strides }
    }

    /// Obtain a read-only view of this tensor.
    #[inline]
    pub fn as_view(&self) -> TensorView<'_, T, N, Layout, &'_ [T]> {
        match self {
            Self::Borrowed(view) => *view,
            Self::Owned { data, shape, strides } => {
                TensorView::with_strides(data.as_slice(), *shape, *strides).unwrap()
            }
        }
    }

    /// Returns the total logical element count.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(view) => view.num_elements(),
            Self::Owned { shape, .. } => shape.iter().product(),
        }
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns logical shape.
    #[inline]
    pub fn shape(&self) -> [usize; N] {
        match self {
            Self::Borrowed(view) => view.shape(),
            Self::Owned { shape, .. } => *shape,
        }
    }

    /// Returns tensor strides.
    #[inline]
    pub fn strides(&self) -> [usize; N] {
        match self {
            Self::Borrowed(view) => view.strides(),
            Self::Owned { strides, .. } => *strides,
        }
    }

    /// Returns whether tensor is contiguous row-major.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        match self {
            Self::Borrowed(view) => view.is_contiguous(),
            Self::Owned { shape, strides, .. } => {
                let expected = row_major_strides(*shape);
                *strides == expected
            }
        }
    }

    /// Upgrades to `Owned` if currently borrowed and returns a mutable reference to the `AlignedVec`.
    #[inline]
    pub fn to_mut(&mut self) -> &mut AlignedVec<T, Align> {
        if let Self::Borrowed(view) = *self {
            let owned = AlignedVec::from_slice(view.as_slice());
            *self = Self::Owned {
                data: owned,
                shape: view.shape(),
                strides: view.strides(),
            };
        }
        match self {
            Self::Owned { data, .. } => data,
            _ => unreachable!(),
        }
    }

    /// Converts into the owned `AlignedVec` storage.
    #[inline]
    pub fn into_owned(self) -> AlignedVec<T, Align> {
        match self {
            Self::Borrowed(view) => AlignedVec::from_slice(view.as_slice()),
            Self::Owned { data, .. } => data,
        }
    }

    /// Reshapes the tensor to a different rank `M` without allocation.
    #[inline]
    pub fn reshape<const M: usize>(
        self,
        new_shape: [usize; M],
    ) -> Result<TensorCow<'a, T, M, RowMajor, Align>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        let old_count = self.len();
        let new_count = new_shape.iter().product::<usize>();
        if old_count != new_count {
            return Err(TensorError::ShapeMismatch);
        }
        match self {
            Self::Borrowed(view) => {
                let reshaped = view.reshape(new_shape)?;
                Ok(TensorCow::Borrowed(reshaped))
            }
            Self::Owned { data, .. } => {
                let strides = row_major_strides(new_shape);
                Ok(TensorCow::Owned { data, shape: new_shape, strides })
            }
        }
    }
}

impl<'a, T: Clone + 'a, const N: usize, Layout, Align> Clone for TensorCow<'a, T, N, Layout, Align>
where
    Align: Alignment,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(view) => Self::Borrowed(*view),
            Self::Owned { data, shape, strides } => Self::Owned {
                data: data.clone(),
                shape: *shape,
                strides: *strides,
            },
        }
    }
}

impl<'a, T: 'a, const N: usize, Layout, Align> core::ops::Deref for TensorCow<'a, T, N, Layout, Align>
where
    Align: Alignment,
{
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(view) => view.as_slice(),
            Self::Owned { data, .. } => data.as_slice(),
        }
    }
}

impl<'a, 'b, T, const N: usize, Layout1, Layout2, Align1, Align2>
    PartialEq<TensorCow<'b, T, N, Layout2, Align2>> for TensorCow<'a, T, N, Layout1, Align1>
where
    T: PartialEq,
    Align1: Alignment,
    Align2: Alignment,
{
    #[inline]
    fn eq(&self, other: &TensorCow<'b, T, N, Layout2, Align2>) -> bool {
        let s1: &[T] = self;
        let s2: &[T] = other;
        s1 == s2
      }
}

impl<'a, T, const N: usize, Layout, Align> Eq for TensorCow<'a, T, N, Layout, Align>
where
    T: Eq,
    Align: Alignment,
{}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute row-major strides for `shape`: `strides[i] = ∏_{j=i+1..N} shape[j]`.
#[inline(always)]
fn row_major_strides<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut strides = [1usize; N];
    let mut acc = 1usize;
    let mut i = N;
    while i > 0 {
        i -= 1;
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i]);
    }
    strides
}

/// Compute flat offset: `∑ idx[i] * strides[i]`.
#[inline(always)]
fn compute_offset(idx: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0usize;
    for i in 0..idx.len() {
        offset += idx[i] * strides[i];
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_major_strides_3d() {
        let s = row_major_strides([2usize, 3, 4]);
        assert_eq!(s, [12, 4, 1]);
    }

    #[test]
    fn test_tensor_view_get_2d() {
        let data: Vec<i32> = (0..12).collect();
        let t = TensorView::<i32, 2>::new(&data, [3, 4]).unwrap();
        assert_eq!(t.get([1, 2]).unwrap(), 6);
    }

    #[test]
    fn test_reshape() {
        let data: Vec<i32> = (0..12).collect();
        let t2d = TensorView::<i32, 2>::new(&data, [3, 4]).unwrap();
        let t1d = t2d.reshape([12]).unwrap();
        assert_eq!(t1d.num_elements(), 12);
        assert_eq!(t1d.get([11]).unwrap(), 11);
    }

    #[test]
    fn test_row_view() {
        let data: Vec<f32> = (0..9).map(|x| x as f32).collect();
        let t = TensorView::<f32, 2>::new(&data, [3, 3]).unwrap();
        let row1 = t.row_view(1).unwrap();
        assert_eq!(row1.num_elements(), 3);
        assert_eq!(row1.get([0]).unwrap(), 3.0);
        assert_eq!(row1.get([2]).unwrap(), 5.0);
    }
}
