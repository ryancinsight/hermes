//! Sparse matrix formats and representations.

pub mod cow;
pub mod ops;
pub mod spmv;
pub mod types;
pub mod view;

pub use cow::{
    // Format-to-owned-storage mapping for Cow containers
    CowFormat,
    OwnedBlockedCoo,
    // Owned heap-backed storage types
    OwnedCsr,
    OwnedDenseWithMask,
    OwnedSellP,
    // Generic Clone-on-Write sparse container
    SparseCow,
};
pub use ops::SparseOps;
pub use spmv::SparseSpMv;
pub use types::{BlockedCooData, CsrData, DenseWithMaskData, SellPData, SparseShape};
pub use view::{SparseView, SparseViewShape};

/// Compressed Sparse Row format marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Csr;

/// Sliced ELLPACK format marker.
///
/// `C` is the row-slice width (number of rows per slice). Typical values: 4 or 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SellP<const C: usize>;

/// Blocked COO format marker.
///
/// - `BM`: block row count
/// - `BN`: block column count
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockedCoo<const BM: usize, const BN: usize>;

/// Dense storage with a boolean mask indicating non-zero elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseWithMask;

impl crate::private::Sealed for Csr {}
impl<const C: usize> crate::private::Sealed for SellP<C> {}
impl<const BM: usize, const BN: usize> crate::private::Sealed for BlockedCoo<BM, BN> {}
impl crate::private::Sealed for DenseWithMask {}

/// Marker trait for sparse matrix storage formats.
///
/// Sealed to prevent external format implementations. The GAT `Storage<'a, T>`
/// maps this format marker to its concrete data struct, eliminating the need
/// for an internal enum discriminant.
pub trait SparseFormat: crate::private::Sealed + Send + Sync + 'static {
    /// Human-readable format name for diagnostics.
    const NAME: &'static str;

    /// The concrete data struct type for this format, parameterized by
    /// lifetime `'a` and element type `T`.
    type Storage<'a, T: 'a>: SparseShape;
}

impl SparseFormat for Csr {
    const NAME: &'static str = "CSR";
    type Storage<'a, T: 'a> = CsrData<'a, T>;
}

impl<const C: usize> SparseFormat for SellP<C> {
    const NAME: &'static str = "SELL-p";
    type Storage<'a, T: 'a> = SellPData<'a, T, C>;
}

impl<const BM: usize, const BN: usize> SparseFormat for BlockedCoo<BM, BN> {
    const NAME: &'static str = "Blocked-COO";
    type Storage<'a, T: 'a> = BlockedCooData<'a, T, BM, BN>;
}

impl SparseFormat for DenseWithMask {
    const NAME: &'static str = "DenseWithMask";
    type Storage<'a, T: 'a> = DenseWithMaskData<'a, T>;
}
