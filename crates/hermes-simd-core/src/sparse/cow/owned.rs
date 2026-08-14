//! Owned heap-backed sparse storage types.

use super::super::{
    types::{
        BlockedCooData, BlockedCooMatrix, CsrData, CsrMatrix, DenseWithMaskData,
        DenseWithMaskMatrix, SellPData, SellPMatrix, SparseValidate,
    },
    SparseShape,
};
use crate::align::Aligned;
use crate::vec::AlignedVec;

/// Owned heap-backed CSR storage.
pub struct OwnedCsr<T> {
    pub(crate) values: AlignedVec<T, Aligned<64>>,
    pub(crate) col_indices: AlignedVec<i32, Aligned<64>>,
    pub(crate) row_ptr: AlignedVec<i32, Aligned<64>>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<T> OwnedCsr<T> {
    /// Construct owned CSR storage.
    #[inline]
    #[must_use]
    pub fn new(
        values: AlignedVec<T, Aligned<64>>,
        col_indices: AlignedVec<i32, Aligned<64>>,
        row_ptr: AlignedVec<i32, Aligned<64>>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
        }
    }

    /// Return a borrowed `CsrData` view over this owned storage.
    #[inline]
    #[must_use]
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
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T> SparseValidate for OwnedCsr<T> {
    #[inline]
    fn validate(&self) -> Result<(), crate::SimdError> {
        self.as_view().validate()
    }
}

/// Owned heap-backed SELL-p storage.
pub struct OwnedSellP<T, const C: usize> {
    pub(crate) values: AlignedVec<T, Aligned<64>>,
    pub(crate) col_indices: AlignedVec<i32, Aligned<64>>,
    pub(crate) slice_ptr: AlignedVec<i32, Aligned<64>>,
    pub(crate) slice_col_count: AlignedVec<i32, Aligned<64>>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<T, const C: usize> OwnedSellP<T, C> {
    /// Construct owned SELL-p storage.
    #[inline]
    #[must_use]
    pub fn new(
        values: AlignedVec<T, Aligned<64>>,
        col_indices: AlignedVec<i32, Aligned<64>>,
        slice_ptr: AlignedVec<i32, Aligned<64>>,
        slice_col_count: AlignedVec<i32, Aligned<64>>,
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
        }
    }

    /// Return a borrowed `SellPData` view over this owned storage.
    #[inline]
    #[must_use]
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
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T, const C: usize> SparseValidate for OwnedSellP<T, C> {
    #[inline]
    fn validate(&self) -> Result<(), crate::SimdError> {
        self.as_view().validate()
    }
}

/// Owned heap-backed Blocked-COO storage.
pub struct OwnedBlockedCoo<T, const BM: usize, const BN: usize> {
    pub(crate) blocks: AlignedVec<T, Aligned<64>>,
    pub(crate) block_row: AlignedVec<i32, Aligned<64>>,
    pub(crate) block_col: AlignedVec<i32, Aligned<64>>,
    pub(crate) nblocks: usize,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<T, const BM: usize, const BN: usize> OwnedBlockedCoo<T, BM, BN> {
    /// Construct owned Blocked-COO storage.
    #[inline]
    #[must_use]
    pub fn new(
        blocks: AlignedVec<T, Aligned<64>>,
        block_row: AlignedVec<i32, Aligned<64>>,
        block_col: AlignedVec<i32, Aligned<64>>,
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
        }
    }

    /// Return a borrowed `BlockedCooData` view over this owned storage.
    #[inline]
    #[must_use]
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
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T, const BM: usize, const BN: usize> SparseValidate for OwnedBlockedCoo<T, BM, BN> {
    #[inline]
    fn validate(&self) -> Result<(), crate::SimdError> {
        self.as_view().validate()
    }
}

/// Owned heap-backed DenseWithMask storage.
pub struct OwnedDenseWithMask<T> {
    pub(crate) values: AlignedVec<T, Aligned<64>>,
    pub(crate) mask: AlignedVec<bool, Aligned<64>>,
    pub(crate) nrows: usize,
    pub(crate) ncols: usize,
}

impl<T> OwnedDenseWithMask<T> {
    /// Construct owned DenseWithMask storage.
    #[inline]
    #[must_use]
    pub fn new(
        values: AlignedVec<T, Aligned<64>>,
        mask: AlignedVec<bool, Aligned<64>>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self {
            values,
            mask,
            nrows,
            ncols,
        }
    }

    /// Return a borrowed `DenseWithMaskData` view over this owned storage.
    #[inline]
    #[must_use]
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
    #[inline(always)]
    fn nrows(&self) -> usize {
        self.nrows
    }
    #[inline(always)]
    fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T> SparseValidate for OwnedDenseWithMask<T> {
    #[inline]
    fn validate(&self) -> Result<(), crate::SimdError> {
        self.as_view().validate()
    }
}
