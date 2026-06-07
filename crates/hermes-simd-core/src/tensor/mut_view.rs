//! Mutable N-dimensional strided tensor view.
//!
//! # Design
//!
//! `TensorMut<'a, T, const N: usize, Layout>` is the writable counterpart to
//! [`super::TensorView`].  It provides in-place element access, row mutation,
//! and scatter-write operations without any heap allocation.
//!
//! Like `TensorView`, all shape/stride arithmetic uses `[usize; N]` const arrays
//! resolved at compile time.  The `Layout` parameter is a zero-sized marker type
//! (`RowMajor` / `ColMajor`) tracked by `PhantomData`; it carries no runtime state.
//!
//! # Invariants
//!
//! The same invariants as `TensorView` apply:
//! - `strides[i] * shape[i]` must not overflow `usize`.
//! - `data.len() >= ∏ shape[i]` for any valid element access.

use core::marker::PhantomData;
use super::{TensorError, RowMajor, row_major_strides, compute_offset};

/// Mutable N-dimensional strided tensor view.
///
/// # Type Parameters
/// - `'a` — exclusive lifetime of the underlying data slice.
/// - `T` — element type.
/// - `N` — tensor rank (const generic).
/// - `Layout` — layout marker ZST ([`RowMajor`] or [`ColMajor`]).
pub struct TensorMut<'a, T, const N: usize, Layout = RowMajor> {
    data:    &'a mut [T],
    shape:   [usize; N],
    strides: [usize; N],
    _layout: PhantomData<Layout>,
}

impl<'a, T, const N: usize> TensorMut<'a, T, N, RowMajor> {
    /// Create a row-major mutable tensor view over `data` with the given `shape`.
    ///
    /// Strides are computed as C-order: `strides[i] = ∏_{j=i+1..N} shape[j]`.
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `∏ shape > data.len()`.
    #[inline]
    pub fn new(data: &'a mut [T], shape: [usize; N]) -> Result<Self, TensorError> {
        let elem_count = shape.iter().product::<usize>();
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        let strides = row_major_strides(shape);
        Ok(Self { data, shape, strides, _layout: PhantomData })
    }
}

impl<'a, T, const N: usize, Layout> TensorMut<'a, T, N, Layout> {
    /// Create a mutable tensor view with explicit strides.
    ///
    /// # Errors
    /// Returns [`TensorError::ShapeMismatch`] if `∏ shape > data.len()`.
    #[inline]
    pub fn with_strides(
        data: &'a mut [T],
        shape: [usize; N],
        strides: [usize; N],
    ) -> Result<Self, TensorError> {
        let elem_count = shape.iter().product::<usize>();
        if elem_count > data.len() {
            return Err(TensorError::ShapeMismatch);
        }
        Ok(Self { data, shape, strides, _layout: PhantomData })
    }

    /// The logical shape of this tensor.
    #[inline(always)]
    pub fn shape(&self) -> [usize; N] { self.shape }

    /// The strides of this tensor in element units.
    #[inline(always)]
    pub fn strides(&self) -> [usize; N] { self.strides }

    /// Number of elements in this tensor.
    #[inline]
    pub fn num_elements(&self) -> usize { self.shape.iter().product() }

    /// Whether this view is row-major contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        let expected = row_major_strides(self.shape);
        self.strides == expected
    }

    /// Bounds-checked mutable element access.
    ///
    /// # Errors
    /// Returns [`TensorError::IndexOutOfBounds`] if any `idx[i] >= shape[i]`.
    #[inline]
    pub fn get_mut(&mut self, idx: [usize; N]) -> Result<&mut T, TensorError> {
        for i in 0..N {
            if idx[i] >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds);
            }
        }
        let offset = compute_offset(&idx, &self.strides);
        Ok(&mut self.data[offset])
    }

    /// Unchecked mutable element access.
    ///
    /// # Safety
    /// `idx[i] < shape[i]` for all `i`.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, idx: [usize; N]) -> &mut T {
        let offset = compute_offset(&idx, &self.strides);
        self.data.get_unchecked_mut(offset)
    }

    /// View the underlying flat slice mutably (in storage order).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] { self.data }

    /// View the underlying flat slice immutably.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] { self.data }

    /// Apply a closure to every element in-place using the flat storage order.
    ///
    /// For contiguous views this is a simple slice iteration; no strides are needed.
    /// For non-contiguous views this still works correctly because the closure operates
    /// on the raw storage, not on logical indices.
    ///
    /// Zero-allocation: operates fully in-place.
    #[inline]
    pub fn map_inplace(&mut self, mut f: impl FnMut(T) -> T)
    where
        T: Copy,
    {
        for x in self.data.iter_mut() {
            *x = f(*x);
        }
    }

    /// Fill every element with `value`.
    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        for x in self.data.iter_mut() {
            *x = value;
        }
    }
}

// ---------------------------------------------------------------------------
// 2-D mutable methods
// ---------------------------------------------------------------------------

impl<'a, T, Layout> TensorMut<'a, T, 2, Layout> {
    /// Return a mutable slice for row `i`.
    ///
    /// Only valid for contiguous row-major layouts where `strides[0] == shape[1]`.
    ///
    /// # Errors
    /// - [`TensorError::IndexOutOfBounds`] if `i >= shape[0]`.
    /// - [`TensorError::NotContiguous`] if the view is not contiguous.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> Result<&mut [T], TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        if i >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let ncols = self.shape[1];
        let start = i * ncols;
        Ok(&mut self.data[start..start + ncols])
    }

    /// Iterate mutably over all rows of a contiguous 2-D tensor.
    ///
    /// Yields `shape[0]` mutable slices each of length `shape[1]`.
    ///
    /// # Errors
    /// Returns [`TensorError::NotContiguous`] if the view is not contiguous.
    #[inline]
    pub fn iter_rows_mut(&mut self) -> Result<core::slice::ChunksMut<'_, T>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        Ok(self.data.chunks_mut(self.shape[1]))
    }
}

// ---------------------------------------------------------------------------
// 3-D mutable methods
// ---------------------------------------------------------------------------

impl<'a, T, Layout> TensorMut<'a, T, 3, Layout> {
    /// Return a mutable flat slice for the `b`-th matrix in a batched tensor.
    ///
    /// Interprets shape as `[batch, rows, cols]`. Requires contiguous layout.
    ///
    /// # Errors
    /// - [`TensorError::IndexOutOfBounds`] if `b >= shape[0]`.
    /// - [`TensorError::NotContiguous`] if the view is not contiguous.
    #[inline]
    pub fn matrix_mut(&mut self, b: usize) -> Result<&mut [T], TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        if b >= self.shape[0] {
            return Err(TensorError::IndexOutOfBounds);
        }
        let rows = self.shape[1];
        let cols = self.shape[2];
        let start = b * rows * cols;
        Ok(&mut self.data[start..start + rows * cols])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_mut_new_and_fill() {
        let mut data = vec![0i32; 12];
        let mut t = TensorMut::<i32, 2>::new(&mut data, [3, 4]).unwrap();
        t.fill(7);
        assert!(t.as_slice().iter().all(|&x| x == 7));
    }

    #[test]
    fn test_tensor_mut_get_mut() {
        let mut data: Vec<i32> = (0..12).collect();
        let mut t = TensorMut::<i32, 2>::new(&mut data, [3, 4]).unwrap();
        *t.get_mut([1, 2]).unwrap() = 99;
        // Element [1][2] = index 1*4+2 = 6
        assert_eq!(data[6], 99);
    }

    #[test]
    fn test_tensor_mut_map_inplace() {
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut t = TensorMut::<f32, 1>::new(&mut data, [4]).unwrap();
        t.map_inplace(|x| x * 2.0);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_row_mut() {
        let mut data: Vec<i32> = (0..9).collect();
        let mut t = TensorMut::<i32, 2>::new(&mut data, [3, 3]).unwrap();
        let row = t.row_mut(1).unwrap();
        row.iter_mut().for_each(|x| *x += 100);
        assert_eq!(data[3], 103);
        assert_eq!(data[4], 104);
        assert_eq!(data[5], 105);
    }
}
