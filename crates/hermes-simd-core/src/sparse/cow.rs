//! Clone-on-Write sparse matrix containers.
//!
//! `SparseCow<'a, T, Format, Arch>` wraps either a zero-copy borrowed
//! `SparseView` or an owned heap allocation of the same format.
//!
//! # Design
//! - `Borrowed` variant holds a `SparseView<'a, T, Format, Arch>` — zero allocation,
//!   zero copy; the borrow expires with lifetime `'a`.
//! - `Owned` variant holds a format-specific `OwnedSparse<T, Format>` whose
//!   arrays live on the heap.  No allocation occurs until `to_owned()` is called.
//! - `SparseSpMv<T>` and `SparseOps<T>` forward to the inner view produced by
//!   `Format::owned_as_view`, achieving zero-cost monomorphization: the compiler
//!   erases both variants and emits code identical to a direct `SparseView` call.

use alloc::vec::Vec;
use crate::arch::SimdArch;
use crate::scalar::Scalar;
use crate::kernel::SimdKernel;
use super::{
    SparseShape, SparseView,
    Csr, SellP, BlockedCoo, DenseWithMask,
    types::{CsrMatrix, SellPMatrix, BlockedCooMatrix, DenseWithMaskMatrix,
             CsrData, SellPData, BlockedCooData, DenseWithMaskData},
    spmv::SparseSpMv,
    ops::SparseOps,
};

// ─────────────────────────────────────────────────────────────────────────────
// Owned sparse storage — one concrete type per format, heap-backed.
// ─────────────────────────────────────────────────────────────────────────────

/// Owned heap-backed CSR storage.
pub struct OwnedCsr<T> {
    values:      Vec<T>,
    col_indices: Vec<i32>,
    row_ptr:     Vec<i32>,
    nrows:       usize,
    ncols:       usize,
}

impl<T> OwnedCsr<T> {
    /// Construct owned CSR storage.
    #[inline]
    pub fn new(values: Vec<T>, col_indices: Vec<i32>, row_ptr: Vec<i32>, nrows: usize, ncols: usize) -> Self {
        Self { values, col_indices, row_ptr, nrows, ncols }
    }

    /// Return a borrowed `CsrData` view over this owned storage.
    #[inline]
    pub fn as_view(&self) -> CsrData<'_, T> {
        CsrMatrix::new(
            self.values.as_slice(),
            self.col_indices.as_slice(),
            self.row_ptr.as_slice(),
            self.nrows,
            self.ncols,
        )
    }
}

impl<T> SparseShape for OwnedCsr<T> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}

/// Owned heap-backed SELL-p storage.
pub struct OwnedSellP<T, const C: usize> {
    values:          Vec<T>,
    col_indices:     Vec<i32>,
    slice_ptr:       Vec<i32>,
    slice_col_count: Vec<i32>,
    nrows:           usize,
    ncols:           usize,
}

impl<T, const C: usize> OwnedSellP<T, C> {
    /// Construct owned SELL-p storage.
    #[inline]
    pub fn new(
        values: Vec<T>,
        col_indices: Vec<i32>,
        slice_ptr: Vec<i32>,
        slice_col_count: Vec<i32>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self { values, col_indices, slice_ptr, slice_col_count, nrows, ncols }
    }

    /// Return a borrowed `SellPData` view over this owned storage.
    #[inline]
    pub fn as_view(&self) -> SellPData<'_, T, C> {
        SellPMatrix::new(
            self.values.as_slice(),
            self.col_indices.as_slice(),
            self.slice_ptr.as_slice(),
            self.slice_col_count.as_slice(),
            self.nrows,
            self.ncols,
        )
    }
}

impl<T, const C: usize> SparseShape for OwnedSellP<T, C> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}

/// Owned heap-backed Blocked-COO storage.
pub struct OwnedBlockedCoo<T, const BM: usize, const BN: usize> {
    blocks:    Vec<T>,
    block_row: Vec<i32>,
    block_col: Vec<i32>,
    nblocks:   usize,
    nrows:     usize,
    ncols:     usize,
}

impl<T, const BM: usize, const BN: usize> OwnedBlockedCoo<T, BM, BN> {
    /// Construct owned Blocked-COO storage.
    #[inline]
    pub fn new(
        blocks:    Vec<T>,
        block_row: Vec<i32>,
        block_col: Vec<i32>,
        nblocks:   usize,
        nrows:     usize,
        ncols:     usize,
    ) -> Self {
        Self { blocks, block_row, block_col, nblocks, nrows, ncols }
    }

    /// Return a borrowed `BlockedCooData` view over this owned storage.
    #[inline]
    pub fn as_view(&self) -> BlockedCooData<'_, T, BM, BN> {
        BlockedCooMatrix::new(
            self.blocks.as_slice(),
            self.block_row.as_slice(),
            self.block_col.as_slice(),
            self.nblocks,
            self.nrows,
            self.ncols,
        )
    }
}

impl<T, const BM: usize, const BN: usize> SparseShape for OwnedBlockedCoo<T, BM, BN> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}

/// Owned heap-backed DenseWithMask storage.
pub struct OwnedDenseWithMask<T> {
    values: Vec<T>,
    mask:   Vec<bool>,
    nrows:  usize,
    ncols:  usize,
}

impl<T> OwnedDenseWithMask<T> {
    /// Construct owned DenseWithMask storage.
    #[inline]
    pub fn new(values: Vec<T>, mask: Vec<bool>, nrows: usize, ncols: usize) -> Self {
        Self { values, mask, nrows, ncols }
    }

    /// Return a borrowed `DenseWithMaskData` view over this owned storage.
    #[inline]
    pub fn as_view(&self) -> DenseWithMaskData<'_, T> {
        DenseWithMaskMatrix::new(
            self.values.as_slice(),
            self.mask.as_slice(),
            self.nrows,
            self.ncols,
        )
    }
}

impl<T> SparseShape for OwnedDenseWithMask<T> {
    #[inline(always)] fn nrows(&self) -> usize { self.nrows }
    #[inline(always)] fn ncols(&self) -> usize { self.ncols }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-format Clone-on-Write containers.
// Concrete enums avoid GAT lifetime restrictions that arise when trying to
// abstract over Format::Owned in a single generic enum.
// ─────────────────────────────────────────────────────────────────────────────

/// Clone-on-Write CSR sparse matrix.
pub enum CsrCow<'a, T, Arch: SimdArch> {
    /// Zero-copy borrowed CSR view.
    Borrowed(SparseView<'a, T, Csr, Arch>),
    /// Owned heap-backed CSR storage.
    Owned(OwnedCsr<T>),
}

impl<'a, T, Arch: SimdArch> CsrCow<'a, T, Arch> {
    /// Wrap a borrowed CSR view (zero allocation).
    #[inline(always)]
    pub fn borrowed(data: CsrData<'a, T>) -> Self {
        Self::Borrowed(SparseView::from_csr(data))
    }

    /// Wrap an owned CSR (heap-allocated).
    #[inline(always)]
    pub fn owned(storage: OwnedCsr<T>) -> Self {
        Self::Owned(storage)
    }

    /// Build an owned `CsrCow` from raw vectors.
    #[inline]
    pub fn from_vecs(
        values:      Vec<T>,
        col_indices: Vec<i32>,
        row_ptr:     Vec<i32>,
        nrows:       usize,
        ncols:       usize,
    ) -> Self {
        Self::Owned(OwnedCsr::new(values, col_indices, row_ptr, nrows, ncols))
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o)    => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o)    => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view (no heap allocation).
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool { matches!(self, Self::Borrowed(_)) }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool { matches!(self, Self::Owned(_)) }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self) where T: Clone {
        if let Self::Borrowed(v) = self {
            let d = v.csr_data();
            let owned = OwnedCsr::new(
                d.values.to_vec(),
                d.col_indices.to_vec(),
                d.row_ptr.to_vec(),
                d.nrows,
                d.ncols,
            );
            *self = Self::Owned(owned);
        }
    }
}

impl<'a, T: Scalar, Arch: SimdArch + SimdKernel<T>> SparseSpMv<T> for CsrCow<'a, T, Arch> {
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.spmv(x, y),
            Self::Owned(o)    => SparseView::<T, Csr, Arch>::from_csr(o.as_view()).spmv(x, y),
        }
    }
}

impl<'a, T: Scalar, Arch: SimdArch> SparseOps<T> for CsrCow<'a, T, Arch> {
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o)    => SparseView::<T, Csr, Arch>::from_csr(o.as_view()).sum_values(),
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o)    => SparseView::<T, Csr, Arch>::from_csr(o.as_view()).elementwise_mul_dense(dense, out_values),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SELL-p Clone-on-Write container
// ─────────────────────────────────────────────────────────────────────────────

/// Clone-on-Write SELL-p sparse matrix.
pub enum SellPCow<'a, T, const C: usize, Arch: SimdArch> {
    /// Zero-copy borrowed SELL-p view.
    Borrowed(SparseView<'a, T, SellP<C>, Arch>),
    /// Owned heap-backed SELL-p storage.
    Owned(OwnedSellP<T, C>),
}

impl<'a, T, const C: usize, Arch: SimdArch> SellPCow<'a, T, C, Arch> {
    /// Wrap a borrowed SELL-p view (zero allocation).
    #[inline(always)]
    pub fn borrowed(data: SellPData<'a, T, C>) -> Self {
        Self::Borrowed(SparseView::from_sellp(data))
    }

    /// Wrap an owned SELL-p (heap-allocated).
    #[inline(always)]
    pub fn owned(storage: OwnedSellP<T, C>) -> Self {
        Self::Owned(storage)
    }

    /// Build an owned `SellPCow` from raw vectors.
    #[inline]
    pub fn from_vecs(
        values:          Vec<T>,
        col_indices:     Vec<i32>,
        slice_ptr:       Vec<i32>,
        slice_col_count: Vec<i32>,
        nrows:           usize,
        ncols:           usize,
    ) -> Self {
        Self::Owned(OwnedSellP::new(values, col_indices, slice_ptr, slice_col_count, nrows, ncols))
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o)    => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o)    => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool { matches!(self, Self::Borrowed(_)) }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool { matches!(self, Self::Owned(_)) }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self) where T: Clone {
        if let Self::Borrowed(v) = self {
            let d = &v.data;
            let owned = OwnedSellP::new(
                d.values.to_vec(),
                d.col_indices.to_vec(),
                d.slice_ptr.to_vec(),
                d.slice_col_count.to_vec(),
                d.nrows,
                d.ncols,
            );
            *self = Self::Owned(owned);
        }
    }
}

impl<'a, T: Scalar, const C: usize, Arch: SimdArch + SimdKernel<T>> SparseSpMv<T>
    for SellPCow<'a, T, C, Arch>
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.spmv(x, y),
            Self::Owned(o)    => SparseView::<T, SellP<C>, Arch>::from_sellp(o.as_view()).spmv(x, y),
        }
    }
}

impl<'a, T: Scalar, const C: usize, Arch: SimdArch> SparseOps<T> for SellPCow<'a, T, C, Arch> {
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o)    => SparseView::<T, SellP<C>, Arch>::from_sellp(o.as_view()).sum_values(),
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o)    => SparseView::<T, SellP<C>, Arch>::from_sellp(o.as_view()).elementwise_mul_dense(dense, out_values),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blocked-COO Clone-on-Write container
// ─────────────────────────────────────────────────────────────────────────────

/// Clone-on-Write Blocked-COO sparse matrix.
pub enum BlockedCooCow<'a, T, const BM: usize, const BN: usize, Arch: SimdArch> {
    /// Zero-copy borrowed Blocked-COO view.
    Borrowed(SparseView<'a, T, BlockedCoo<BM, BN>, Arch>),
    /// Owned heap-backed Blocked-COO storage.
    Owned(OwnedBlockedCoo<T, BM, BN>),
}

impl<'a, T, const BM: usize, const BN: usize, Arch: SimdArch> BlockedCooCow<'a, T, BM, BN, Arch> {
    /// Wrap a borrowed Blocked-COO view (zero allocation).
    #[inline(always)]
    pub fn borrowed(data: BlockedCooData<'a, T, BM, BN>) -> Self {
        Self::Borrowed(SparseView::from_blocked_coo(data))
    }

    /// Wrap an owned Blocked-COO (heap-allocated).
    #[inline(always)]
    pub fn owned(storage: OwnedBlockedCoo<T, BM, BN>) -> Self {
        Self::Owned(storage)
    }

    /// Build an owned `BlockedCooCow` from raw vectors.
    #[inline]
    pub fn from_vecs(
        blocks:    Vec<T>,
        block_row: Vec<i32>,
        block_col: Vec<i32>,
        nblocks:   usize,
        nrows:     usize,
        ncols:     usize,
    ) -> Self {
        Self::Owned(OwnedBlockedCoo::new(blocks, block_row, block_col, nblocks, nrows, ncols))
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o)    => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o)    => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool { matches!(self, Self::Borrowed(_)) }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool { matches!(self, Self::Owned(_)) }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self) where T: Clone {
        if let Self::Borrowed(v) = self {
            let d = &v.data;
            let owned = OwnedBlockedCoo::new(
                d.blocks.to_vec(),
                d.block_row.to_vec(),
                d.block_col.to_vec(),
                d.nblocks,
                d.nrows,
                d.ncols,
            );
            *self = Self::Owned(owned);
        }
    }
}

impl<'a, T: Scalar, const BM: usize, const BN: usize, Arch: SimdArch + SimdKernel<T>>
    SparseSpMv<T> for BlockedCooCow<'a, T, BM, BN, Arch>
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.spmv(x, y),
            Self::Owned(o)    => SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo(o.as_view()).spmv(x, y),
        }
    }
}

impl<'a, T: Scalar, const BM: usize, const BN: usize, Arch: SimdArch>
    SparseOps<T> for BlockedCooCow<'a, T, BM, BN, Arch>
{
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o)    => SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo(o.as_view()).sum_values(),
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o)    => SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo(o.as_view()).elementwise_mul_dense(dense, out_values),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DenseWithMask Clone-on-Write container
// ─────────────────────────────────────────────────────────────────────────────

/// Clone-on-Write DenseWithMask sparse matrix.
pub enum DenseWithMaskCow<'a, T, Arch: SimdArch> {
    /// Zero-copy borrowed DenseWithMask view.
    Borrowed(SparseView<'a, T, DenseWithMask, Arch>),
    /// Owned heap-backed DenseWithMask storage.
    Owned(OwnedDenseWithMask<T>),
}

impl<'a, T, Arch: SimdArch> DenseWithMaskCow<'a, T, Arch> {
    /// Wrap a borrowed DenseWithMask view (zero allocation).
    #[inline(always)]
    pub fn borrowed(data: DenseWithMaskData<'a, T>) -> Self {
        Self::Borrowed(SparseView::from_dense_with_mask(data))
    }

    /// Wrap an owned DenseWithMask (heap-allocated).
    #[inline(always)]
    pub fn owned(storage: OwnedDenseWithMask<T>) -> Self {
        Self::Owned(storage)
    }

    /// Build an owned `DenseWithMaskCow` from raw vectors.
    #[inline]
    pub fn from_vecs(values: Vec<T>, mask: Vec<bool>, nrows: usize, ncols: usize) -> Self {
        Self::Owned(OwnedDenseWithMask::new(values, mask, nrows, ncols))
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o)    => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o)    => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool { matches!(self, Self::Borrowed(_)) }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool { matches!(self, Self::Owned(_)) }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self) where T: Clone {
        if let Self::Borrowed(v) = self {
            let d = &v.data;
            let owned = OwnedDenseWithMask::new(
                d.values.to_vec(),
                d.mask.to_vec(),
                d.nrows,
                d.ncols,
            );
            *self = Self::Owned(owned);
        }
    }
}

impl<'a, T: Scalar, Arch: SimdArch + SimdKernel<T>> SparseSpMv<T>
    for DenseWithMaskCow<'a, T, Arch>
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.spmv(x, y),
            Self::Owned(o)    => SparseView::<T, DenseWithMask, Arch>::from_dense_with_mask(o.as_view()).spmv(x, y),
        }
    }
}

impl<'a, T: Scalar, Arch: SimdArch> SparseOps<T> for DenseWithMaskCow<'a, T, Arch> {
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o)    => SparseView::<T, DenseWithMask, Arch>::from_dense_with_mask(o.as_view()).sum_values(),
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o)    => SparseView::<T, DenseWithMask, Arch>::from_dense_with_mask(o.as_view()).elementwise_mul_dense(dense, out_values),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public re-export: SparseCow as an alias to CsrCow for API compat.
// Users wanting other formats use CsrCow/SellPCow/BlockedCooCow/DenseWithMaskCow directly.
// ─────────────────────────────────────────────────────────────────────────────

/// Type alias: `SparseCow` defaults to the CSR Clone-on-Write container.
///
/// For other formats, use [`CsrCow`], [`SellPCow`], [`BlockedCooCow`], or
/// [`DenseWithMaskCow`] directly.
pub type SparseCow<'a, T, Arch> = CsrCow<'a, T, Arch>;
