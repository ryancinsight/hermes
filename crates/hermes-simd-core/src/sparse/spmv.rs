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

/// Fused scaled sparse matrix-vector multiplication.
///
/// Implemented only by formats with a fused `y += alpha * A * x` kernel, so a
/// single row pass scales the dot product before the single `y[r]` write
/// instead of replaying the row for a separate `axpy`. Formats without a
/// fused kernel keep [`SparseSpMv`].
pub trait SparseScaledSpMv<T> {
    /// Perform scaled matrix-vector multiplication: `y += alpha * A * x`.
    ///
    /// `alpha == 0` (either sign) returns after the size checks with `y`
    /// bit-for-bit unchanged like BLAS level 2: `x` and the stored values
    /// are not read, so a non-finite `x` or an overflowing dot product cannot
    /// reach `y`. Otherwise IEEE 754 applies to floats: NaN, infinity or
    /// overflow in a row's dot product, or a NaN `alpha`, propagates into that
    /// row's `y` entry. Rows without stored entries are skipped, so they leave
    /// `y` untouched whatever `alpha` is, a NaN or infinite one included.
    ///
    /// # Overflow
    /// For an integer `T` the caller keeps every product, partial sum,
    /// `alpha * dot` and updated `y` entry inside the range of `T`. Otherwise
    /// the result is unspecified and memory-safe. For a primitive integer,
    /// lane arithmetic is backend-defined, while the scalar tail, the scaling
    /// and the `y` update panic under overflow checks (dev and test profiles)
    /// and wrap otherwise; row length and backend decide which runs in a lane.
    /// The `I8`, `I16` and `I32` newtypes wrap in every profile and never
    /// panic on overflow.
    ///
    /// # Panics
    /// Panics if `x` or `y` is shorter than the column or row count (also for
    /// a zero `alpha`), if the storage is structurally invalid, or, for a
    /// primitive integer `T` under overflow checks, on integer overflow.
    fn spmv_scaled(&self, alpha: T, x: &[T], y: &mut [T]);
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
