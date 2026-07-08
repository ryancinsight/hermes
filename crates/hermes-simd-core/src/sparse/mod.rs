//! Sparse matrix formats and representations.
//!
//! Format selection is workload-dependent:
//! - CSR is the baseline for irregular sparsity with compact zero-copy storage.
//! - SELL-p groups rows into const-generic row slices and is best when rows have
//!   similar non-zero counts, because padding overhead stays bounded and the
//!   vectorized path can load one slice lane per row.
//! - Blocked COO is suited to locally dense block structure; const block
//!   dimensions monomorphize the inner block loops without a runtime format
//!   switch.
//! - Dense-with-mask keeps dense row-major values plus a boolean structural
//!   mask; it is useful when the dense layout is already required by a caller,
//!   but it is memory-bound for low non-zero densities because it stores every
//!   value and mask bit.
//!
//! `crates/hermes-simd-benches/benches/sparse_bench.rs` records the empirical
//! crossover data. Its scalability sweep varies row count and structural
//! non-zero density while keeping values borrowed at the kernel boundary.

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
pub use types::{
    BlockedCooData, CsrData, DenseWithMaskData, SellPData, SparseShape, ValidatedData,
};
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

/// Typestate marker for sparse formats whose structural invariants were checked
/// before kernel entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Validated<F>(core::marker::PhantomData<F>);

impl crate::private::Sealed for Csr {}
impl<const C: usize> crate::private::Sealed for SellP<C> {}
impl<const BM: usize, const BN: usize> crate::private::Sealed for BlockedCoo<BM, BN> {}
impl crate::private::Sealed for DenseWithMask {}
impl<F: SparseFormat> crate::private::Sealed for Validated<F> {}

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

impl<F: SparseFormat> SparseFormat for Validated<F> {
    const NAME: &'static str = F::NAME;
    type Storage<'a, T: 'a> = ValidatedData<F::Storage<'a, T>>;
}
