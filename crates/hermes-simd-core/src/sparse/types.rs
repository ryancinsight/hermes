//! Concrete data structures for sparse matrix formats.

/// Trait providing uniform row/column dimension access for sparse data structs.
pub trait SparseShape {
    /// Number of rows in the matrix.
    fn nrows(&self) -> usize;
    /// Number of columns in the matrix.
    fn ncols(&self) -> usize;
}

/// CSR matrix data references.
#[derive(Clone)]
pub struct CsrData<'a, T> {
    /// Non-zero values, length `nnz`.
    pub values: &'a [T],
    /// Column indices for each non-zero, length `nnz`.
    pub col_indices: &'a [i32],
    /// Row pointers, length `nrows + 1`.
    pub row_ptr: &'a [i32],
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
}

impl<'a, T> CsrData<'a, T> {
    /// Number of stored non-zero elements.
    #[inline(always)]
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Fraction of entries that are zero (structural sparsity estimate).
    #[inline]
    pub fn sparsity(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 { return 0.0; }
        1.0 - (self.nnz() as f64 / total as f64)
    }
}

impl<T> SparseShape for CsrData<'_, T> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}

/// SELL-p matrix data references.
#[derive(Clone)]
pub struct SellPData<'a, T, const C: usize> {
    /// Padded non-zero values.
    pub values: &'a [T],
    /// Padded column indices.
    pub col_indices: &'a [i32],
    /// Offsets into `values`/`col_indices` for each row-slice.
    pub slice_ptr: &'a [i32],
    /// Maximum column count per slice (determines padding).
    pub slice_col_count: &'a [i32],
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
}

impl<'a, T, const C: usize> SellPData<'a, T, C> {
    /// Number of row slices `ceil(nrows / C)`.
    #[inline(always)]
    pub fn nslices(&self) -> usize { (self.nrows + C - 1) / C }
}

impl<T, const C: usize> SparseShape for SellPData<'_, T, C> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}

/// Blocked COO matrix data references.
#[derive(Clone)]
pub struct BlockedCooData<'a, T, const BM: usize, const BN: usize> {
    /// Flattened dense block data.
    pub blocks: &'a [T],
    /// Block row indices.
    pub block_row: &'a [i32],
    /// Block column indices.
    pub block_col: &'a [i32],
    /// Number of stored blocks.
    pub nblocks: usize,
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
}

impl<'a, T, const BM: usize, const BN: usize> BlockedCooData<'a, T, BM, BN> {
    /// Total number of stored non-zeros.
    #[inline(always)]
    pub fn nnz(&self) -> usize { self.nblocks * BM * BN }
}

impl<T, const BM: usize, const BN: usize> SparseShape for BlockedCooData<'_, T, BM, BN> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}

/// Dense matrix with a boolean non-zero mask.
#[derive(Clone)]
pub struct DenseWithMaskData<'a, T> {
    /// Dense value array, row-major.
    pub values: &'a [T],
    /// Boolean non-zero mask.
    pub mask: &'a [bool],
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
}

impl<'a, T> DenseWithMaskData<'a, T> {
    /// Count of structurally non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize { self.mask.iter().filter(|&&b| b).count() }

    /// Structural sparsity.
    #[inline]
    pub fn sparsity(&self) -> f64 {
        let total = self.nrows * self.ncols;
        if total == 0 { return 0.0; }
        1.0 - (self.nnz() as f64 / total as f64)
    }
}

impl<T> SparseShape for DenseWithMaskData<'_, T> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}
