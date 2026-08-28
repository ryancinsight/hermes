//! Sparse matrix-vector multiplication (`SpMV`) kernels.
//!
//! # Safety
//!
//! Every kernel call in the format-owned child modules is
//! `#[target_feature]`-gated and is therefore sound only on a host implementing
//! `Arch`. That holds by construction rather than by inspection:
//! [`SimdView::new`](crate::view::SimdView::new) returns `None` for an
//! architecture the host cannot execute, and the sparse and copy-on-write
//! constructors assert the same condition. Per-site `SAFETY` comments record
//! only the obligations beyond that capability proof.

mod blocked_coo;
mod csr;
mod dense_with_mask;
mod sellp;

use crate::kernel::SimdStorage;
use crate::scalar::Scalar;

/// Unified trait for sparse matrix-vector multiplication.
pub trait SparseSpMv<T> {
    /// Perform matrix-vector multiplication: `y += A * x`.
    ///
    /// # Panics
    /// Panics if the dimensions of `x` or `y` are incompatible with the matrix,
    /// or if a raw storage representation is structurally invalid.
    fn spmv(&self, x: &[T], y: &mut [T]);
}

/// Build an `Arch::IndexVector` from a slice of `i32` column indices.
///
/// # Safety
/// All implementations of `SimdStorage` in this workspace define `IndexVector`
/// with the layout `[i32; LANE_COUNT]`. This function reads `&[i32]` of length
/// `>= LANE_COUNT` as one `Arch::IndexVector` via an unaligned read (so element
/// alignment, not vector alignment, is the only requirement). The size half of
/// the layout invariant is enforced at compile time per backend by the
/// `const` assert below; the length contract is enforced at runtime.
#[inline(always)]
pub(crate) unsafe fn build_index_vector<T: Scalar, Arch: SimdStorage<T>>(
    cols: &[i32],
) -> Arch::IndexVector {
    const {
        assert!(
            core::mem::size_of::<Arch::IndexVector>()
                == Arch::LANE_COUNT * core::mem::size_of::<i32>(),
            "IndexVector size must equal LANE_COUNT * size_of::<i32>()"
        );
    };
    assert!(
        cols.len() >= Arch::LANE_COUNT,
        "cols slice length {} is less than LANE_COUNT {}",
        cols.len(),
        Arch::LANE_COUNT
    );
    let ptr = cols.as_ptr().cast::<Arch::IndexVector>();
    // SAFETY: the size assertion and caller-provided slice length establish a
    // complete IndexVector byte range; unaligned access permits slice alignment.
    unsafe { core::ptr::read_unaligned(ptr) }
}

#[inline(never)]
fn validate_spmv_sizes(x_len: usize, y_len: usize, ncols: usize, nrows: usize, format_name: &str) {
    assert!(
        x_len >= ncols,
        "x too short for {format_name} ncols (got {x_len}, expected >= {ncols})"
    );
    assert!(
        y_len >= nrows,
        "y too short for {format_name} nrows (got {y_len}, expected >= {nrows})"
    );
}
