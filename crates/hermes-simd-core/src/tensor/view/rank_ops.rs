//! Rank-specialized [`TensorView`] operations.
//!
//! Rank-2 row/column/diagonal views and iterators, rank-3 `matrix_at` batch
//! slicing, and the zero-copy transpose (shape/stride swap). These live apart
//! from the rank-agnostic core in [`super`] so the N-D substrate and its 2-D/3-D
//! specializations remain separately legible.

use core::marker::PhantomData;

use super::TensorView;
use crate::tensor::error::TensorError;
use crate::tensor::layout::{ColMajor, RowMajor};

// ---------------------------------------------------------------------------
// 2-D specific methods — read-only
// ---------------------------------------------------------------------------

impl<'a, 'b, T, L> TensorView<'a, T, 2, L, &'b [T]> {
    /// Return a zero-copy 1-D view of row `i`.
    ///
    /// # Errors
    /// Returns [`TensorError::IndexOutOfBounds`] if `i >= shape[0]`.
    #[inline]
    pub fn row_view(
        &self,
        i: usize,
    ) -> Result<TensorView<'a, T, 1, RowMajor, &'b [T]>, TensorError> {
        if i >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let row_start = i * self.strides[0];
        let row_len = self.shape[1];
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
    #[must_use]
    pub fn transpose_view(&self) -> TensorView<'a, T, 2, ColMajor, &'b [T]> {
        TensorView {
            ptr: self.ptr,
            shape: [self.shape[1], self.shape[0]],
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
        let nrows = self.shape[0];
        let row_str = self.strides[0];
        let slice = self.as_slice();
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
        let slice = self.as_slice();
        (0..diag_len).map(move |i| slice[i * diag_str])
    }
}

// ---------------------------------------------------------------------------
// 2-D specific methods — mutable
// ---------------------------------------------------------------------------

impl<'a, T, L> TensorView<'a, T, 2, L, &mut [T]> {
    /// Return a mutable 1-D view of row `i`.
    ///
    /// # Errors
    /// Returns [`TensorError::IndexOutOfBounds`] if `i >= shape[0]`.
    #[inline]
    pub fn row_view_mut(
        &mut self,
        i: usize,
    ) -> Result<TensorView<'a, T, 1, RowMajor, &mut [T]>, TensorError> {
        if i >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let row_start = i * self.strides[0];
        let row_len = self.shape[1];
        let slice = unsafe { &mut *self.ptr };
        let row_data = &mut slice[row_start..row_start + row_len];
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

impl<'a, T, L> TensorView<'a, T, 3, L, &'a [T]> {
    /// Return a 2-D view of the `b`-th matrix in a batched tensor.
    ///
    /// # Errors
    /// Returns [`TensorError::NotContiguous`] when the source is not
    /// contiguous, or [`TensorError::IndexOutOfBounds`] when `b` is outside
    /// the batch dimension.
    #[inline]
    pub fn matrix_at(
        &self,
        b: usize,
    ) -> Result<TensorView<'_, T, 2, RowMajor, &'_ [T]>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        if b >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let rows = self.shape[1];
        let cols = self.shape[2];
        let start = b * rows * cols;
        let full_slice: &[T] = self.as_slice();
        let slice = &full_slice[start..start + rows * cols];
        TensorView::<'_, T, 2, RowMajor, &'_ [T]>::new(slice, [rows, cols])
    }
}

impl<'a, T, L> TensorView<'a, T, 3, L, &'a mut [T]> {
    /// Return a mutable 2-D view of the `b`-th matrix in a batched tensor.
    ///
    /// # Errors
    /// Returns [`TensorError::NotContiguous`] when the source is not
    /// contiguous, or [`TensorError::IndexOutOfBounds`] when `b` is outside
    /// the batch dimension.
    #[inline]
    pub fn matrix_at_mut(
        &mut self,
        b: usize,
    ) -> Result<TensorView<'_, T, 2, RowMajor, &'_ mut [T]>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        if b >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let rows = self.shape[1];
        let cols = self.shape[2];
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
    #[must_use]
    pub fn transpose(self) -> TensorView<'a, T, 2, ColMajor, Ref> {
        TensorView {
            ptr: self.ptr,
            shape: [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
            _layout: PhantomData,
        }
    }
}

impl<'a, T, Ref> TensorView<'a, T, 2, ColMajor, Ref> {
    /// Transpose a column-major 2-D tensor view to row-major.
    #[inline]
    #[must_use]
    pub fn transpose(self) -> TensorView<'a, T, 2, RowMajor, Ref> {
        TensorView {
            ptr: self.ptr,
            shape: [self.shape[1], self.shape[0]],
            strides: [self.strides[1], self.strides[0]],
            _layout: PhantomData,
        }
    }
}
