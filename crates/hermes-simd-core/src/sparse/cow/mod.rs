//! Clone-on-Write sparse matrix containers.

use super::{
    ops::SparseOps,
    spmv::SparseSpMv,
    types::{BlockedCooData, CsrData, DenseWithMaskData, SellPData},
    BlockedCoo, Csr, DenseWithMask, SellP, SparseView,
};
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use alloc::vec::Vec;

pub mod owned;
pub use owned::{OwnedBlockedCoo, OwnedCsr, OwnedDenseWithMask, OwnedSellP};

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
        values: Vec<T>,
        col_indices: Vec<i32>,
        row_ptr: Vec<i32>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::Owned(OwnedCsr::new(values, col_indices, row_ptr, nrows, ncols))
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o) => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o) => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view (no heap allocation).
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self)
    where
        T: Clone,
    {
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
            Self::Owned(o) => SparseView::<T, Csr, Arch>::from_csr(o.as_view()).spmv(x, y),
        }
    }
}

impl<'a, T: Scalar, Arch: SimdArch> SparseOps<T> for CsrCow<'a, T, Arch> {
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o) => SparseView::<T, Csr, Arch>::from_csr(o.as_view()).sum_values(),
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o) => SparseView::<T, Csr, Arch>::from_csr(o.as_view())
                .elementwise_mul_dense(dense, out_values),
        }
    }
}

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
        values: Vec<T>,
        col_indices: Vec<i32>,
        slice_ptr: Vec<i32>,
        slice_col_count: Vec<i32>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::Owned(OwnedSellP::new(
            values,
            col_indices,
            slice_ptr,
            slice_col_count,
            nrows,
            ncols,
        ))
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o) => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o) => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self)
    where
        T: Clone,
    {
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
            Self::Owned(o) => SparseView::<T, SellP<C>, Arch>::from_sellp(o.as_view()).spmv(x, y),
        }
    }
}

impl<'a, T: Scalar, const C: usize, Arch: SimdArch> SparseOps<T> for SellPCow<'a, T, C, Arch> {
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o) => SparseView::<T, SellP<C>, Arch>::from_sellp(o.as_view()).sum_values(),
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o) => SparseView::<T, SellP<C>, Arch>::from_sellp(o.as_view())
                .elementwise_mul_dense(dense, out_values),
        }
    }
}

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
        blocks: Vec<T>,
        block_row: Vec<i32>,
        block_col: Vec<i32>,
        nblocks: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::Owned(OwnedBlockedCoo::new(
            blocks, block_row, block_col, nblocks, nrows, ncols,
        ))
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o) => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o) => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self)
    where
        T: Clone,
    {
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

impl<'a, T: Scalar, const BM: usize, const BN: usize, Arch: SimdArch + SimdKernel<T>> SparseSpMv<T>
    for BlockedCooCow<'a, T, BM, BN, Arch>
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.spmv(x, y),
            Self::Owned(o) => {
                SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo(o.as_view()).spmv(x, y)
            }
        }
    }
}

impl<'a, T: Scalar, const BM: usize, const BN: usize, Arch: SimdArch> SparseOps<T>
    for BlockedCooCow<'a, T, BM, BN, Arch>
{
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o) => {
                SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo(o.as_view())
                    .sum_values()
            }
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o) => {
                SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo(o.as_view())
                    .elementwise_mul_dense(dense, out_values)
            }
        }
    }
}

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
            Self::Owned(o) => o.nrows,
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o) => o.ncols,
        }
    }

    /// Returns `true` if this holds a borrowed view.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns `true` if this holds heap-owned data.
    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Promote to owned, cloning from the borrowed view if needed.
    #[inline]
    pub fn to_owned(&mut self)
    where
        T: Clone,
    {
        if let Self::Borrowed(v) = self {
            let d = &v.data;
            let owned =
                OwnedDenseWithMask::new(d.values.to_vec(), d.mask.to_vec(), d.nrows, d.ncols);
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
            Self::Owned(o) => {
                SparseView::<T, DenseWithMask, Arch>::from_dense_with_mask(o.as_view()).spmv(x, y)
            }
        }
    }
}

impl<'a, T: Scalar, Arch: SimdArch> SparseOps<T> for DenseWithMaskCow<'a, T, Arch> {
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o) => {
                SparseView::<T, DenseWithMask, Arch>::from_dense_with_mask(o.as_view()).sum_values()
            }
        }
    }
    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o) => {
                SparseView::<T, DenseWithMask, Arch>::from_dense_with_mask(o.as_view())
                    .elementwise_mul_dense(dense, out_values)
            }
        }
    }
}

/// Type alias: `SparseCow` defaults to the CSR Clone-on-Write container.
///
/// For other formats, use [`CsrCow`], [`SellPCow`], [`BlockedCooCow`], or
/// [`DenseWithMaskCow`] directly.
pub type SparseCow<'a, T, Arch> = CsrCow<'a, T, Arch>;
