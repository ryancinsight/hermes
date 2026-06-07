//! Concrete data structures for sparse matrix formats.

use core::marker::PhantomData;

/// Trait providing uniform row/column dimension access for sparse data structs.
pub trait SparseShape {
    /// Number of rows in the matrix.
    fn nrows(&self) -> usize;
    /// Number of columns in the matrix.
    fn ncols(&self) -> usize;
}

/// Generic CSR matrix storage.
#[derive(Clone)]
pub struct CsrMatrix<T, V, I> {
    /// Non-zero values, length `nnz`.
    pub values: V,
    /// Column indices for each non-zero, length `nnz`.
    pub col_indices: I,
    /// Row pointers, length `nrows + 1`.
    pub row_ptr: I,
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, V, I> CsrMatrix<T, V, I> {
    /// Create a new `CsrMatrix`.
    #[inline(always)]
    pub fn new(values: V, col_indices: I, row_ptr: I, nrows: usize, ncols: usize) -> Self {
        Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
            _marker: PhantomData,
        }
    }
}

impl<T, V: AsRef<[T]>, I> CsrMatrix<T, V, I> {
    /// Number of stored non-zero elements.
    #[inline(always)]
    pub fn nnz(&self) -> usize {
        self.values.as_ref().len()
    }

    /// Fraction of entries that are zero (structural sparsity estimate).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / total as f64)
    }
}

impl<T, V, I> SparseShape for CsrMatrix<T, V, I> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T, V: AsRef<[T]>, I: AsRef<[i32]>> CsrMatrix<T, V, I> {
    /// Return a borrowed representation of this CSR data.
    #[inline]
    pub fn as_borrowed(&self) -> CsrMatrix<T, &[T], &[i32]> {
        CsrMatrix {
            values: self.values.as_ref(),
            col_indices: self.col_indices.as_ref(),
            row_ptr: self.row_ptr.as_ref(),
            nrows: self.nrows,
            ncols: self.ncols,
            _marker: PhantomData,
        }
    }
}

/// Backward-compatible type alias.
pub type CsrData<'a, T> = CsrMatrix<T, &'a [T], &'a [i32]>;

/// Generic Sliced ELLPACK matrix storage.
#[derive(Clone)]
pub struct SellPMatrix<T, const C: usize, V, I> {
    /// Padded non-zero values.
    pub values: V,
    /// Padded column indices.
    pub col_indices: I,
    /// Offsets into `values`/`col_indices` for each row-slice.
    pub slice_ptr: I,
    /// Maximum column count per slice (determines padding).
    pub slice_col_count: I,
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, const C: usize, V, I> SellPMatrix<T, C, V, I> {
    /// Create a new `SellPMatrix`.
    #[inline(always)]
    pub fn new(
        values: V,
        col_indices: I,
        slice_ptr: I,
        slice_col_count: I,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self {
            values,
            col_indices,
            slice_ptr,
            slice_col_count,
            nrows,
            ncols,
            _marker: PhantomData,
        }
    }

    /// Number of row slices `ceil(nrows / C)`.
    #[inline(always)]
    pub fn nslices(&self) -> usize {
        (self.nrows + C - 1) / C
    }
}

impl<T, const C: usize, V, I> SparseShape for SellPMatrix<T, C, V, I> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T, const C: usize, V: AsRef<[T]>, I: AsRef<[i32]>> SellPMatrix<T, C, V, I> {
    /// Return a borrowed representation of this SELL-p data.
    #[inline]
    pub fn as_borrowed(&self) -> SellPMatrix<T, C, &[T], &[i32]> {
        SellPMatrix {
            values: self.values.as_ref(),
            col_indices: self.col_indices.as_ref(),
            slice_ptr: self.slice_ptr.as_ref(),
            slice_col_count: self.slice_col_count.as_ref(),
            nrows: self.nrows,
            ncols: self.ncols,
            _marker: PhantomData,
        }
    }
}

/// Backward-compatible type alias.
pub type SellPData<'a, T, const C: usize> = SellPMatrix<T, C, &'a [T], &'a [i32]>;

/// Generic Blocked COO matrix storage.
#[derive(Clone)]
pub struct BlockedCooMatrix<T, const BM: usize, const BN: usize, V, I> {
    /// Flattened dense block data.
    pub blocks: V,
    /// Block row indices.
    pub block_row: I,
    /// Block column indices.
    pub block_col: I,
    /// Number of stored blocks.
    pub nblocks: usize,
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, const BM: usize, const BN: usize, V, I> BlockedCooMatrix<T, BM, BN, V, I> {
    /// Create a new `BlockedCooMatrix`.
    #[inline(always)]
    pub fn new(
        blocks: V,
        block_row: I,
        block_col: I,
        nblocks: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self {
            blocks,
            block_row,
            block_col,
            nblocks,
            nrows,
            ncols,
            _marker: PhantomData,
        }
    }

    /// Total number of stored non-zeros.
    #[inline(always)]
    pub fn nnz(&self) -> usize {
        self.nblocks * BM * BN
    }
}

impl<T, const BM: usize, const BN: usize, V, I> SparseShape for BlockedCooMatrix<T, BM, BN, V, I> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T, const BM: usize, const BN: usize, V: AsRef<[T]>, I: AsRef<[i32]>>
    BlockedCooMatrix<T, BM, BN, V, I>
{
    /// Return a borrowed representation of this Blocked COO data.
    #[inline]
    pub fn as_borrowed(&self) -> BlockedCooMatrix<T, BM, BN, &[T], &[i32]> {
        BlockedCooMatrix {
            blocks: self.blocks.as_ref(),
            block_row: self.block_row.as_ref(),
            block_col: self.block_col.as_ref(),
            nblocks: self.nblocks,
            nrows: self.nrows,
            ncols: self.ncols,
            _marker: PhantomData,
        }
    }
}

/// Backward-compatible type alias.
pub type BlockedCooData<'a, T, const BM: usize, const BN: usize> =
    BlockedCooMatrix<T, BM, BN, &'a [T], &'a [i32]>;

/// Generic Dense matrix with a boolean non-zero mask.
#[derive(Clone)]
pub struct DenseWithMaskMatrix<T, V, M> {
    /// Dense value array, row-major.
    pub values: V,
    /// Boolean non-zero mask.
    pub mask: M,
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, V, M> DenseWithMaskMatrix<T, V, M> {
    /// Create a new `DenseWithMaskMatrix`.
    #[inline(always)]
    pub fn new(values: V, mask: M, nrows: usize, ncols: usize) -> Self {
        Self {
            values,
            mask,
            nrows,
            ncols,
            _marker: PhantomData,
        }
    }
}

impl<T, V: AsRef<[T]>, M: AsRef<[bool]>> DenseWithMaskMatrix<T, V, M> {
    /// Count of structurally non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.mask.as_ref().iter().filter(|&&b| b).count()
    }

    /// Structural sparsity.
    #[inline]
    pub fn sparsity(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / total as f64)
    }
}

impl<T, V, M> SparseShape for DenseWithMaskMatrix<T, V, M> {
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T, V: AsRef<[T]>, M: AsRef<[bool]>> DenseWithMaskMatrix<T, V, M> {
    /// Return a borrowed representation of this Dense-with-Mask data.
    #[inline]
    pub fn as_borrowed(&self) -> DenseWithMaskMatrix<T, &[T], &[bool]> {
        DenseWithMaskMatrix {
            values: self.values.as_ref(),
            mask: self.mask.as_ref(),
            nrows: self.nrows,
            ncols: self.ncols,
            _marker: PhantomData,
        }
    }
}

/// Backward-compatible type alias.
pub type DenseWithMaskData<'a, T> = DenseWithMaskMatrix<T, &'a [T], &'a [bool]>;
