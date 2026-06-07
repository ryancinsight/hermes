//! Format-parameterized sparse matrix views.

use core::marker::PhantomData;
use crate::arch::SimdArch;
use super::{
    Csr, SellP, BlockedCoo, DenseWithMask,
    SparseFormat, SparseShape,
    CsrData, SellPData, BlockedCooData, DenseWithMaskData,
};

/// Format-parameterized sparse matrix view.
pub struct SparseView<'a, T: 'a, Format, Arch>
where
    Format: SparseFormat,
    Arch: SimdArch,
{
    pub(crate) data: Format::Storage<'a, T>,
    _arch: PhantomData<Arch>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a, T: 'a, Format, Arch> SparseView<'a, T, Format, Arch>
where
    Format: SparseFormat,
    Arch: SimdArch,
{
    /// Create a `SparseView` from the format's storage representation.
    #[inline(always)]
    pub fn new(data: Format::Storage<'a, T>) -> Self {
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
    }

    /// Get a reference to the underlying storage representation.
    #[inline(always)]
    pub fn storage(&self) -> &Format::Storage<'a, T> {
        &self.data
    }

    /// Number of rows in the sparse matrix.
    #[inline(always)]
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns in the sparse matrix.
    #[inline(always)]
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<'a, T: 'a, Arch> SparseView<'a, T, Csr, Arch>
where
    Arch: SimdArch,
{
    /// Create a `SparseView` over CSR data.
    #[inline]
    pub fn from_csr(data: CsrData<'a, T>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }

    /// Access the underlying CSR data.
    #[inline(always)]
    pub fn csr_data(&self) -> &CsrData<'a, T> {
        &self.data
    }
}

impl<'a, T: 'a, const C: usize, Arch> SparseView<'a, T, SellP<C>, Arch>
where
    Arch: SimdArch,
{
    /// Create a `SparseView` over SELL-p data (generic C).
    #[inline]
    pub fn from_sellp(data: SellPData<'a, T, C>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }
}

impl<'a, T: 'a, Arch: SimdArch> SparseView<'a, T, SellP<4>, Arch> {
    /// Create a `SparseView` over SELL-p data with C=4.
    #[inline]
    pub fn from_sellp4(data: SellPData<'a, T, 4>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }
}

impl<'a, T: 'a, Arch: SimdArch> SparseView<'a, T, SellP<8>, Arch> {
    /// Create a `SparseView` over SELL-p data with C=8.
    #[inline]
    pub fn from_sellp8(data: SellPData<'a, T, 8>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }
}

impl<'a, T: 'a, const BM: usize, const BN: usize, Arch> SparseView<'a, T, BlockedCoo<BM, BN>, Arch>
where
    Arch: SimdArch,
{
    /// Create a `SparseView` over Blocked-COO data (generic BM, BN).
    #[inline]
    pub fn from_blocked_coo(data: BlockedCooData<'a, T, BM, BN>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }
}

impl<'a, T: 'a, Arch: SimdArch> SparseView<'a, T, BlockedCoo<4, 4>, Arch> {
    /// Create a `SparseView` over Blocked-COO 4x4 data.
    #[inline]
    pub fn from_blocked_coo_4x4(data: BlockedCooData<'a, T, 4, 4>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }
}

impl<'a, T: 'a, Arch: SimdArch> SparseView<'a, T, BlockedCoo<8, 8>, Arch> {
    /// Create a `SparseView` over Blocked-COO 8x8 data.
    #[inline]
    pub fn from_blocked_coo_8x8(data: BlockedCooData<'a, T, 8, 8>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }
}

impl<'a, T: 'a, Arch> SparseView<'a, T, DenseWithMask, Arch>
where
    Arch: SimdArch,
{
    /// Create a `SparseView` over dense-with-mask data.
    #[inline]
    pub fn from_dense_with_mask(data: DenseWithMaskData<'a, T>) -> Self {
        Self { data, _arch: PhantomData, _lifetime: PhantomData }
    }
}

/// Extension trait providing generic dimension access for any `SparseView`.
pub trait SparseViewShape {
    /// Number of rows.
    fn nrows(&self) -> usize;
    /// Number of columns.
    fn ncols(&self) -> usize;
}

impl<'a, T: 'a, Format, Arch> SparseViewShape for SparseView<'a, T, Format, Arch>
where
    Format: SparseFormat,
    Arch: SimdArch,
    Format::Storage<'a, T>: SparseShape,
{
    #[inline(always)]
    fn nrows(&self) -> usize { self.data.nrows() }
    #[inline(always)]
    fn ncols(&self) -> usize { self.data.ncols() }
}
