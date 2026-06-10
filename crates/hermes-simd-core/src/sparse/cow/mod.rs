//! Clone-on-Write sparse matrix containers.
//!
//! One generic [`SparseCow`] covers every sparse format: the [`CowFormat`]
//! trait maps a format marker to its heap-owned storage and the two
//! conversions (re-borrow owned as a view, clone a view into owned). Adding a
//! new sparse format requires one `CowFormat` impl — the container, its
//! accessors, and the `SparseSpMv` / `SparseOps` forwarding are inherited.

use super::{
    ops::SparseOps,
    spmv::SparseSpMv,
    types::{BlockedCooData, CsrData, DenseWithMaskData, SellPData, SparseShape},
    BlockedCoo, Csr, DenseWithMask, SellP, SparseFormat, SparseView,
};
use crate::arch::SimdArch;
use crate::scalar::Scalar;
use alloc::vec::Vec;

pub mod owned;
pub use owned::{OwnedBlockedCoo, OwnedCsr, OwnedDenseWithMask, OwnedSellP};

/// Maps a sparse format marker to its owned storage and Cow conversions.
///
/// Sealed transitively through [`SparseFormat`].
pub trait CowFormat: SparseFormat {
    /// Heap-owned storage for this format.
    type Owned<T>: SparseShape + Send + Sync
    where
        T: Send + Sync;

    /// Re-borrow owned storage as the format's view storage (zero-copy).
    fn as_storage<T: Send + Sync>(owned: &Self::Owned<T>) -> Self::Storage<'_, T>;

    /// Clone a borrowed view storage into owned storage (one allocation set).
    fn to_owned_storage<T: Clone + Send + Sync>(storage: &Self::Storage<'_, T>) -> Self::Owned<T>;
}

impl CowFormat for Csr {
    type Owned<T>
        = OwnedCsr<T>
    where
        T: Send + Sync;

    #[inline(always)]
    fn as_storage<T: Send + Sync>(owned: &Self::Owned<T>) -> CsrData<'_, T> {
        owned.as_view()
    }

    #[inline]
    fn to_owned_storage<T: Clone + Send + Sync>(d: &CsrData<'_, T>) -> OwnedCsr<T> {
        OwnedCsr::new(
            d.values.to_vec(),
            d.col_indices.to_vec(),
            d.row_ptr.to_vec(),
            d.nrows,
            d.ncols,
        )
    }
}

impl<const C: usize> CowFormat for SellP<C> {
    type Owned<T>
        = OwnedSellP<T, C>
    where
        T: Send + Sync;

    #[inline(always)]
    fn as_storage<T: Send + Sync>(owned: &Self::Owned<T>) -> SellPData<'_, T, C> {
        owned.as_view()
    }

    #[inline]
    fn to_owned_storage<T: Clone + Send + Sync>(d: &SellPData<'_, T, C>) -> OwnedSellP<T, C> {
        OwnedSellP::new(
            d.values.to_vec(),
            d.col_indices.to_vec(),
            d.slice_ptr.to_vec(),
            d.slice_col_count.to_vec(),
            d.nrows,
            d.ncols,
        )
    }
}

impl<const BM: usize, const BN: usize> CowFormat for BlockedCoo<BM, BN> {
    type Owned<T>
        = OwnedBlockedCoo<T, BM, BN>
    where
        T: Send + Sync;

    #[inline(always)]
    fn as_storage<T: Send + Sync>(owned: &Self::Owned<T>) -> BlockedCooData<'_, T, BM, BN> {
        owned.as_view()
    }

    #[inline]
    fn to_owned_storage<T: Clone + Send + Sync>(
        d: &BlockedCooData<'_, T, BM, BN>,
    ) -> OwnedBlockedCoo<T, BM, BN> {
        OwnedBlockedCoo::new(
            d.blocks.to_vec(),
            d.block_row.to_vec(),
            d.block_col.to_vec(),
            d.nblocks,
            d.nrows,
            d.ncols,
        )
    }
}

impl CowFormat for DenseWithMask {
    type Owned<T>
        = OwnedDenseWithMask<T>
    where
        T: Send + Sync;

    #[inline(always)]
    fn as_storage<T: Send + Sync>(owned: &Self::Owned<T>) -> DenseWithMaskData<'_, T> {
        owned.as_view()
    }

    #[inline]
    fn to_owned_storage<T: Clone + Send + Sync>(
        d: &DenseWithMaskData<'_, T>,
    ) -> OwnedDenseWithMask<T> {
        OwnedDenseWithMask::new(d.values.to_vec(), d.mask.to_vec(), d.nrows, d.ncols)
    }
}

/// Clone-on-Write sparse matrix, generic over the storage format `F`.
///
/// `Borrowed` wraps a zero-copy [`SparseView`]; `Owned` holds heap-backed
/// storage. Reads never allocate; [`SparseCow::to_owned`] promotes exactly
/// once.
pub enum SparseCow<'a, T: Send + Sync, F: CowFormat, Arch: SimdArch> {
    /// Zero-copy borrowed view.
    Borrowed(SparseView<'a, T, F, Arch>),
    /// Owned heap-backed storage.
    Owned(F::Owned<T>),
}

impl<'a, T: Send + Sync, F: CowFormat, Arch: SimdArch> SparseCow<'a, T, F, Arch> {
    /// Wrap borrowed view storage (zero allocation).
    #[inline(always)]
    pub fn borrowed(data: F::Storage<'a, T>) -> Self {
        Self::Borrowed(SparseView::new(data))
    }

    /// Wrap owned storage (already heap-allocated).
    #[inline(always)]
    pub fn owned(storage: F::Owned<T>) -> Self {
        Self::Owned(storage)
    }

    /// Number of rows.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.nrows(),
            Self::Owned(o) => o.nrows(),
        }
    }

    /// Number of columns.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ncols(),
            Self::Owned(o) => o.ncols(),
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
    ///
    /// Idempotent: an already-owned container is left untouched.
    #[inline]
    pub fn to_owned(&mut self)
    where
        T: Clone,
    {
        if let Self::Borrowed(v) = self {
            *self = Self::Owned(F::to_owned_storage(v.storage()));
        }
    }
}

impl<'a, T, F, Arch> SparseSpMv<T> for SparseCow<'a, T, F, Arch>
where
    T: Scalar,
    F: CowFormat,
    Arch: SimdArch,
    for<'b> SparseView<'b, T, F, Arch>: SparseSpMv<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.spmv(x, y),
            Self::Owned(o) => SparseView::<T, F, Arch>::new(F::as_storage(o)).spmv(x, y),
        }
    }
}

impl<'a, T, F, Arch> SparseOps<T> for SparseCow<'a, T, F, Arch>
where
    T: Scalar,
    F: CowFormat,
    Arch: SimdArch,
    for<'b> SparseView<'b, T, F, Arch>: SparseOps<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        match self {
            Self::Borrowed(v) => v.sum_values(),
            Self::Owned(o) => SparseView::<T, F, Arch>::new(F::as_storage(o)).sum_values(),
        }
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        match self {
            Self::Borrowed(v) => v.elementwise_mul_dense(dense, out_values),
            Self::Owned(o) => SparseView::<T, F, Arch>::new(F::as_storage(o))
                .elementwise_mul_dense(dense, out_values),
        }
    }
}

impl<'a, T: Send + Sync, Arch: SimdArch> SparseCow<'a, T, Csr, Arch> {
    /// Build an owned CSR Cow from raw vectors.
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
}

impl<'a, T: Send + Sync, const C: usize, Arch: SimdArch> SparseCow<'a, T, SellP<C>, Arch> {
    /// Build an owned SELL-p Cow from raw vectors.
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
}

impl<'a, T: Send + Sync, const BM: usize, const BN: usize, Arch: SimdArch>
    SparseCow<'a, T, BlockedCoo<BM, BN>, Arch>
{
    /// Build an owned Blocked-COO Cow from raw vectors.
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
}

impl<'a, T: Send + Sync, Arch: SimdArch> SparseCow<'a, T, DenseWithMask, Arch> {
    /// Build an owned DenseWithMask Cow from raw vectors.
    #[inline]
    pub fn from_vecs(values: Vec<T>, mask: Vec<bool>, nrows: usize, ncols: usize) -> Self {
        Self::Owned(OwnedDenseWithMask::new(values, mask, nrows, ncols))
    }
}
