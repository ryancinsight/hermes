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
    types::{BlockedCooData, CsrData, DenseWithMaskData, SellPData, SparseShape, SparseValidate},
    BlockedCoo, Csr, DenseWithMask, SellP, SparseFormat, SparseView, Validated, ValidatedData,
};
use crate::arch::SimdArch;
use crate::mask::PackedMask;
use crate::scalar::Scalar;
use crate::vec::AlignedVec;

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
            AlignedVec::from_slice_clone(d.values),
            AlignedVec::from_slice(d.col_indices),
            AlignedVec::from_slice(d.row_ptr),
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
            AlignedVec::from_slice_clone(d.values),
            AlignedVec::from_slice(d.col_indices),
            AlignedVec::from_slice(d.slice_ptr),
            AlignedVec::from_slice(d.slice_col_count),
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
            AlignedVec::from_slice_clone(d.blocks),
            AlignedVec::from_slice(d.block_row),
            AlignedVec::from_slice(d.block_col),
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
        OwnedDenseWithMask::new(
            AlignedVec::from_slice_clone(d.values),
            PackedMask::from(&d.mask),
            d.nrows,
            d.ncols,
        )
    }
}

impl<F> CowFormat for Validated<F>
where
    F: CowFormat,
{
    type Owned<T>
        = ValidatedData<F::Owned<T>>
    where
        T: Send + Sync;

    #[inline(always)]
    fn as_storage<T: Send + Sync>(owned: &Self::Owned<T>) -> Self::Storage<'_, T> {
        ValidatedData::new_unchecked(F::as_storage(owned.storage()))
    }

    #[inline]
    fn to_owned_storage<T: Clone + Send + Sync>(storage: &Self::Storage<'_, T>) -> Self::Owned<T> {
        ValidatedData::new_unchecked(F::to_owned_storage(storage.storage()))
    }
}

/// Clone-on-Write sparse matrix, generic over the storage format `F`.
///
/// `Borrowed` wraps a zero-copy [`SparseView`]; `Owned` holds heap-backed
/// storage. Reads never allocate; [`SparseCow::to_owned`] promotes exactly
/// once.
///
/// # Examples
///
/// ```
/// use hermes_simd_core::{Csr, CsrData, Validated};
/// use hermes_simd_core::sparse::{SparseCow, SparseSpMv};
/// use hermes_simd_intrinsics::Scalar;
///
/// let values = [1.0_f64, 2.0, 3.0];
/// let col_indices = [0, 2, 1];
/// let row_ptr = [0, 2, 3];
/// let data = CsrData::new(&values, &col_indices, &row_ptr, 2, 3);
/// let matrix = SparseCow::<f64, Validated<Csr>, Scalar>::try_borrowed(data).unwrap();
///
/// let x = [10.0, 20.0, 30.0];
/// let mut y = [0.0; 2];
/// matrix.spmv(&x, &mut y);
///
/// assert_eq!(matrix.nrows(), 2);
/// assert_eq!(matrix.ncols(), 3);
/// assert!(matrix.is_borrowed());
/// assert_eq!(y, [70.0, 60.0]);
/// ```
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

impl<'a, T: Send + Sync, F: CowFormat, Arch: SimdArch> SparseCow<'a, T, Validated<F>, Arch>
where
    F::Storage<'a, T>: SparseValidate,
{
    /// Validate borrowed storage and wrap it in a zero-copy sparse Cow.
    ///
    /// # Errors
    /// Returns the format-specific validation error if `data` is malformed.
    #[inline]
    pub fn try_borrowed(data: F::Storage<'a, T>) -> Result<Self, crate::SimdError> {
        Ok(Self::Borrowed(SparseView::new(ValidatedData::new(data)?)))
    }
}

impl<T, F, Arch> SparseSpMv<T> for SparseCow<'_, T, F, Arch>
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

impl<T, F, Arch> SparseOps<T> for SparseCow<'_, T, F, Arch>
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

impl<T: Send + Sync + Clone, Arch: SimdArch> SparseCow<'_, T, Csr, Arch> {
    /// Build an owned CSR Cow from slices.
    #[inline]
    pub fn from_slices(
        values: &[T],
        col_indices: &[i32],
        row_ptr: &[i32],
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::Owned(OwnedCsr::new(
            AlignedVec::from_slice_clone(values),
            AlignedVec::from_slice(col_indices),
            AlignedVec::from_slice(row_ptr),
            nrows,
            ncols,
        ))
    }
}

impl<T: Send + Sync + Clone, Arch: SimdArch> SparseCow<'_, T, Validated<Csr>, Arch> {
    /// Build an owned validated CSR Cow from slices.
    ///
    /// # Errors
    /// Returns the CSR validation error if the sparse structure is malformed.
    #[inline]
    pub fn from_slices(
        values: &[T],
        col_indices: &[i32],
        row_ptr: &[i32],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, crate::SimdError> {
        Ok(Self::Owned(ValidatedData::new(OwnedCsr::new(
            AlignedVec::from_slice_clone(values),
            AlignedVec::from_slice(col_indices),
            AlignedVec::from_slice(row_ptr),
            nrows,
            ncols,
        ))?))
    }
}

impl<T: Send + Sync + Clone, const C: usize, Arch: SimdArch> SparseCow<'_, T, SellP<C>, Arch> {
    /// Build an owned SELL-p Cow from slices.
    #[inline]
    pub fn from_slices(
        values: &[T],
        col_indices: &[i32],
        slice_ptr: &[i32],
        slice_col_count: &[i32],
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::Owned(OwnedSellP::new(
            AlignedVec::from_slice_clone(values),
            AlignedVec::from_slice(col_indices),
            AlignedVec::from_slice(slice_ptr),
            AlignedVec::from_slice(slice_col_count),
            nrows,
            ncols,
        ))
    }
}

impl<T: Send + Sync + Clone, const C: usize, Arch: SimdArch>
    SparseCow<'_, T, Validated<SellP<C>>, Arch>
{
    /// Build an owned validated SELL-p Cow from slices.
    ///
    /// # Errors
    /// Returns the SELL-p validation error if the sparse structure is malformed.
    #[inline]
    pub fn from_slices(
        values: &[T],
        col_indices: &[i32],
        slice_ptr: &[i32],
        slice_col_count: &[i32],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, crate::SimdError> {
        Ok(Self::Owned(ValidatedData::new(OwnedSellP::new(
            AlignedVec::from_slice_clone(values),
            AlignedVec::from_slice(col_indices),
            AlignedVec::from_slice(slice_ptr),
            AlignedVec::from_slice(slice_col_count),
            nrows,
            ncols,
        ))?))
    }
}

impl<T: Send + Sync + Clone, const BM: usize, const BN: usize, Arch: SimdArch>
    SparseCow<'_, T, BlockedCoo<BM, BN>, Arch>
{
    /// Build an owned Blocked-COO Cow from slices.
    #[inline]
    pub fn from_slices(
        blocks: &[T],
        block_row: &[i32],
        block_col: &[i32],
        nblocks: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self::Owned(OwnedBlockedCoo::new(
            AlignedVec::from_slice_clone(blocks),
            AlignedVec::from_slice(block_row),
            AlignedVec::from_slice(block_col),
            nblocks,
            nrows,
            ncols,
        ))
    }
}

impl<T: Send + Sync + Clone, const BM: usize, const BN: usize, Arch: SimdArch>
    SparseCow<'_, T, Validated<BlockedCoo<BM, BN>>, Arch>
{
    /// Build an owned validated Blocked-COO Cow from slices.
    ///
    /// # Errors
    /// Returns the Blocked-COO validation error if the sparse structure is malformed.
    #[inline]
    pub fn from_slices(
        blocks: &[T],
        block_row: &[i32],
        block_col: &[i32],
        nblocks: usize,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, crate::SimdError> {
        Ok(Self::Owned(ValidatedData::new(OwnedBlockedCoo::new(
            AlignedVec::from_slice_clone(blocks),
            AlignedVec::from_slice(block_row),
            AlignedVec::from_slice(block_col),
            nblocks,
            nrows,
            ncols,
        ))?))
    }
}

impl<T: Send + Sync + Clone, Arch: SimdArch> SparseCow<'_, T, DenseWithMask, Arch> {
    /// Build an owned `DenseWithMask` Cow from slices, bit-packing the mask
    /// once at this construction boundary.
    #[inline]
    pub fn from_slices(values: &[T], mask: &[bool], nrows: usize, ncols: usize) -> Self {
        Self::Owned(OwnedDenseWithMask::new(
            AlignedVec::from_slice_clone(values),
            PackedMask::from_bools(mask),
            nrows,
            ncols,
        ))
    }
}

impl<T: Send + Sync, F: CowFormat, Arch: SimdArch> crate::sparse::types::SparseValidate
    for SparseCow<'_, T, F, Arch>
where
    for<'b> F::Storage<'b, T>: crate::sparse::types::SparseValidate,
    F::Owned<T>: crate::sparse::types::SparseValidate,
{
    #[inline]
    fn validate(&self) -> Result<(), crate::SimdError> {
        match self {
            Self::Borrowed(v) => v.validate(),
            Self::Owned(o) => o.validate(),
        }
    }
}
