//! Register-blocked matrix–vector product micro-kernel (`y += A · x`).
//!
//! # Theorem (operand reuse)
//! GEMV is memory-bound: it performs `2·nrows·ncols` flops over `nrows·ncols`
//! matrix elements, an arithmetic intensity of ~2 flops/element, so throughput is
//! limited by how often each loaded value is reused. Processing `TILE_M` rows of
//! `A` together loads each `x` vector once and applies it to all `TILE_M` rows,
//! cutting `x` traffic by a factor of `TILE_M` and keeping the `TILE_M`
//! accumulators in registers (one per row) to break the per-row FMA dependency
//! chain. The row remainder (`nrows mod TILE_M`) is handled by a single-row
//! cleanup so any `nrows` is supported with no shape restriction.
//!
//! A **leading dimension** `lda` (the stride between consecutive rows, `lda ≥
//! ncols`) lets the kernel operate on a row-major **sub-matrix** or a transposed
//! view whose rows are contiguous but spaced apart — e.g. the trailing column
//! block of a column-major working buffer in a reflector apply. The packed case
//! is exactly `lda = ncols`; the contiguous per-row dot is unchanged, only the
//! row base address advances by `lda`.

use crate::{
    align::Alignment,
    arch::SimdArch,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
};

#[inline(never)]
fn check_gemv_dimensions(
    a_len: usize,
    x_len: usize,
    y_len: usize,
    nrows: usize,
    ncols: usize,
    lda: usize,
) -> Result<(), SimdError> {
    // Row `r` occupies `[r·lda, r·lda + ncols)`; the last row needs the most span.
    // Overflow ⇒ no slice can satisfy it ⇒ reject, closing the OOB-load path under
    // release `overflow-checks = false` (see `tiling::dims`).
    let a_needed =
        super::dims::checked_strided_span(nrows, ncols, lda).ok_or(SimdError::LengthMismatch)?;
    if lda < ncols || a_len < a_needed || x_len < ncols || y_len < nrows {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

/// Compute `y += A · x` with `A` row-major `nrows × ncols` (packed: row stride
/// `ncols`), blocking `TILE_M` rows so each `x` vector is reused across them.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if the operand spans are too small for the dims.
#[inline]
pub(super) fn gemv_impl<T, Arch, Align, const TILE_M: usize>(
    a: &SimdView<'_, T, Arch, Align>,
    x: &SimdView<'_, T, Arch, Align>,
    y: &mut [T],
    nrows: usize,
    ncols: usize,
) -> Result<(), SimdError>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    gemv_strided_impl::<T, Arch, Align, TILE_M>(a, x, y, nrows, ncols, ncols)
}

/// Compute `y += A · x` with `A` a row-major sub-matrix: `nrows × ncols` with row
/// stride `lda ≥ ncols` (rows contiguous over `ncols`, spaced `lda` apart).
/// `lda = ncols` is the packed [`gemv_impl`].
///
/// # Errors
/// [`SimdError::LengthMismatch`] if `lda < ncols` or the operand spans are too
/// small (`a` needs `(nrows−1)·lda + ncols`, `x ≥ ncols`, `y ≥ nrows`).
#[inline]
pub(super) fn gemv_strided_impl<T, Arch, Align, const TILE_M: usize>(
    a: &SimdView<'_, T, Arch, Align>,
    x: &SimdView<'_, T, Arch, Align>,
    y: &mut [T],
    nrows: usize,
    ncols: usize,
    lda: usize,
) -> Result<(), SimdError>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    struct AssertM<const TILE_M: usize>;
    impl<const TILE_M: usize> AssertM<TILE_M> {
        const OK: () = assert!(TILE_M >= 1, "TILE_M must be at least 1");
    }
    let _ = AssertM::<TILE_M>::OK;

    check_gemv_dimensions(a.len(), x.len(), y.len(), nrows, ncols, lda)?;

    let a_slice = a.as_slice();
    let x_slice = x.as_slice();

    let lane_count = Arch::LANE_COUNT;
    let simd_len = (ncols / lane_count) * lane_count;
    let tail = ncols - simd_len;

    let load = |ptr: *const T| -> Arch::Vector {
        if crate::align::is_aligned_for_arch::<Arch, Align>() {
            unsafe { Arch::load_aligned(ptr) }
        } else {
            unsafe { Arch::load_unaligned(ptr) }
        }
    };

    // The `ncols % lane_count` trailing columns fold into the same vector
    // accumulator as one final masked fmadd (inactive lanes load zero and
    // contribute `a·0 = 0`), replacing a per-row scalar tail loop. `x` beyond
    // `simd_len` is loaded once and reused across every row.
    let tail_mask = unsafe { Arch::leading_k_mask(tail) };
    let x_tail = if tail > 0 {
        // SAFETY: masked load touches only lanes `[simd_len, ncols) ≤ ncols`, and
        // `x.len() ≥ ncols` (checked by `check_gemv_dimensions`).
        unsafe {
            Arch::masked_load_unaligned(x_slice.as_ptr().add(simd_len), tail_mask, Arch::zero())
        }
    } else {
        unsafe { Arch::zero() }
    };

    let mut r = 0;
    while r + TILE_M <= nrows {
        // Initialize TILE_M accumulators to zero.
        let mut accumulators = [unsafe { Arch::zero() }; TILE_M];

        let mut c = 0;
        while c < simd_len {
            // Load x vector (reused across all TILE_M rows).
            let x_vec = load(unsafe { x_slice.as_ptr().add(c) });

            for i in 0..TILE_M {
                let row_idx = r + i;
                let a_vec = load(unsafe { a_slice.as_ptr().add(row_idx * lda + c) });
                accumulators[i] = unsafe { Arch::fmadd(a_vec, x_vec, accumulators[i]) };
            }
            c += lane_count;
        }

        for i in 0..TILE_M {
            let row_idx = r + i;
            if tail > 0 {
                // SAFETY: masked load of lanes `[simd_len, ncols) ≤ ncols` in row
                // `row_idx < nrows` of the validated `A` span (stride `lda`).
                let a_tail = unsafe {
                    Arch::masked_load_unaligned(
                        a_slice.as_ptr().add(row_idx * lda + simd_len),
                        tail_mask,
                        Arch::zero(),
                    )
                };
                accumulators[i] = unsafe { Arch::fmadd(a_tail, x_tail, accumulators[i]) };
            }
            y[row_idx] += unsafe { Arch::sum_reduce(accumulators[i]) };
        }

        r += TILE_M;
    }

    // Cleanup remaining rows (fewer than TILE_M).
    while r < nrows {
        let mut acc = unsafe { Arch::zero() };
        let mut c = 0;
        while c < simd_len {
            let x_vec = load(unsafe { x_slice.as_ptr().add(c) });
            let a_vec = load(unsafe { a_slice.as_ptr().add(r * lda + c) });
            acc = unsafe { Arch::fmadd(a_vec, x_vec, acc) };
            c += lane_count;
        }
        if tail > 0 {
            // SAFETY: masked load of lanes `[simd_len, ncols) ≤ ncols` in row
            // `r < nrows` of the validated `A` span.
            let a_tail = unsafe {
                Arch::masked_load_unaligned(
                    a_slice.as_ptr().add(r * lda + simd_len),
                    tail_mask,
                    Arch::zero(),
                )
            };
            acc = unsafe { Arch::fmadd(a_tail, x_tail, acc) };
        }
        y[r] += unsafe { Arch::sum_reduce(acc) };
        r += 1;
    }

    Ok(())
}
