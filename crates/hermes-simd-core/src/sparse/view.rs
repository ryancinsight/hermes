//! Format-parameterized sparse matrix views.

use super::{
    BlockedCoo, BlockedCooData, Csr, CsrData, DenseWithMask, DenseWithMaskData, SellP, SellPData,
    SparseFormat, SparseShape, Validated, ValidatedData,
};
use crate::arch::{assert_arch_executable, SimdArch};
use core::marker::PhantomData;

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
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline(always)]
    pub fn new(data: Format::Storage<'a, T>) -> Self {
        assert_arch_executable::<Arch>();
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
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_csr(data: CsrData<'a, T>) -> Self {
        assert_arch_executable::<Arch>();
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
    }

    /// Access the underlying CSR data.
    #[inline(always)]
    pub fn csr_data(&self) -> &CsrData<'a, T> {
        &self.data
    }
}

impl<'a, T: 'a, Arch> SparseView<'a, T, Validated<Csr>, Arch>
where
    Arch: SimdArch,
{
    /// Validate CSR storage and create a SpMV-ready view.
    #[inline]
    pub fn try_from_csr(data: CsrData<'a, T>) -> Result<Self, crate::SimdError> {
        Ok(Self {
            data: ValidatedData::new(data)?,
            _arch: PhantomData,
            _lifetime: PhantomData,
        })
    }

    /// Create a SpMV-ready view from already-validated CSR storage.
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_validated_csr(data: ValidatedData<CsrData<'a, T>>) -> Self {
        assert_arch_executable::<Arch>();
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T: 'a, const C: usize, Arch> SparseView<'a, T, SellP<C>, Arch>
where
    Arch: SimdArch,
{
    /// Create a `SparseView` over SELL-p data (generic C).
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_sellp(data: SellPData<'a, T, C>) -> Self {
        assert_arch_executable::<Arch>();
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T: 'a, const C: usize, Arch> SparseView<'a, T, Validated<SellP<C>>, Arch>
where
    Arch: SimdArch,
{
    /// Validate SELL-p storage and create a SpMV-ready view.
    #[inline]
    pub fn try_from_sellp(data: SellPData<'a, T, C>) -> Result<Self, crate::SimdError> {
        Ok(Self {
            data: ValidatedData::new(data)?,
            _arch: PhantomData,
            _lifetime: PhantomData,
        })
    }

    /// Create a SpMV-ready view from already-validated SELL-p storage.
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_validated_sellp(data: ValidatedData<SellPData<'a, T, C>>) -> Self {
        assert_arch_executable::<Arch>();
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T: 'a, const BM: usize, const BN: usize, Arch> SparseView<'a, T, BlockedCoo<BM, BN>, Arch>
where
    Arch: SimdArch,
{
    /// Create a `SparseView` over Blocked-COO data (generic BM, BN).
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_blocked_coo(data: BlockedCooData<'a, T, BM, BN>) -> Self {
        assert_arch_executable::<Arch>();
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T: 'a, const BM: usize, const BN: usize, Arch>
    SparseView<'a, T, Validated<BlockedCoo<BM, BN>>, Arch>
where
    Arch: SimdArch,
{
    /// Validate Blocked-COO storage and create a SpMV-ready view.
    #[inline]
    pub fn try_from_blocked_coo(
        data: BlockedCooData<'a, T, BM, BN>,
    ) -> Result<Self, crate::SimdError> {
        Ok(Self {
            data: ValidatedData::new(data)?,
            _arch: PhantomData,
            _lifetime: PhantomData,
        })
    }

    /// Create a SpMV-ready view from already-validated Blocked-COO storage.
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_validated_blocked_coo(data: ValidatedData<BlockedCooData<'a, T, BM, BN>>) -> Self {
        assert_arch_executable::<Arch>();
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T: 'a, Arch> SparseView<'a, T, DenseWithMask, Arch>
where
    Arch: SimdArch,
{
    /// Create a `SparseView` over dense-with-mask data.
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_dense_with_mask(data: DenseWithMaskData<'a, T>) -> Self {
        assert_arch_executable::<Arch>();
        Self {
            data,
            _arch: PhantomData,
            _lifetime: PhantomData,
        }
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
    fn nrows(&self) -> usize {
        self.data.nrows()
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl<'a, T: 'a, Format, Arch> super::types::SparseValidate for SparseView<'a, T, Format, Arch>
where
    Format: SparseFormat,
    Arch: SimdArch,
    Format::Storage<'a, T>: super::types::SparseValidate,
{
    #[inline]
    fn validate(&self) -> Result<(), crate::SimdError> {
        self.data.validate()
    }
}
