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
//!
//! # Safety
//!
//! The `Arch::*` kernels are `#[target_feature]`-gated and sound only on a host
//! implementing `Arch` — established by the `SimdView` operands, whose
//! constructor rejects an unsupported architecture. The raw-pointer loads read
//! `A`, `x`, and `y` at offsets derived from the caller-supplied `nrows`,
//! `ncols`, and `lda`; `check_gemv_dimensions` validates those against the actual
//! operand lengths (with overflow rejected, closing the OOB path under release
//! `overflow-checks = false`) before any unchecked access, so per-site `SAFETY`
//! comments cite that validation and the per-window bound.

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

    // SAFETY: target-feature kernel (module invariant); every call passes a
    // pointer whose `LANE_COUNT` read stays within `A`/`x` (offsets below bounded
    // by `simd_len <= ncols` and the validated stride), gated on `Align`.
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
    // SAFETY: kernels covered by the module invariant; the masked load touches
    // only lanes `[simd_len, ncols) <= ncols` and `x.len() >= ncols` (checked by
    // `check_gemv_dimensions`).
    let (tail_mask, x_tail) = unsafe {
        let tail_mask = Arch::leading_k_mask(tail);
        let x_tail = if tail > 0 {
            Arch::masked_load_unaligned(x_slice.as_ptr().add(simd_len), tail_mask, Arch::zero())
        } else {
            Arch::zero()
        };
        (tail_mask, x_tail)
    };

    let mut r = 0;
    // SAFETY: `r + TILE_M <= nrows`, so every `row_idx = r + i < nrows`; the
    // vector loads read `[row_idx*lda + c, +LANE_COUNT)` with `c < simd_len <=
    // ncols` and the tail masked load reads `[simd_len, ncols)` — both within the
    // `A` span validated by `check_gemv_dimensions` (stride `lda`), and `x`
    // windows within `x.len() >= ncols`. Kernels covered by the module invariant.
    while r + TILE_M <= nrows {
        unsafe {
            let mut accumulators = [Arch::zero(); TILE_M];

            let mut c = 0;
            while c < simd_len {
                // `x` vector reused across all TILE_M rows.
                let x_vec = load(x_slice.as_ptr().add(c));
                for i in 0..TILE_M {
                    let a_vec = load(a_slice.as_ptr().add((r + i) * lda + c));
                    accumulators[i] = Arch::fmadd(a_vec, x_vec, accumulators[i]);
                }
                c += lane_count;
            }

            for i in 0..TILE_M {
                let row_idx = r + i;
                if tail > 0 {
                    let a_tail = Arch::masked_load_unaligned(
                        a_slice.as_ptr().add(row_idx * lda + simd_len),
                        tail_mask,
                        Arch::zero(),
                    );
                    accumulators[i] = Arch::fmadd(a_tail, x_tail, accumulators[i]);
                }
                y[row_idx] += Arch::sum_reduce(accumulators[i]);
            }
        }
        r += TILE_M;
    }

    // Cleanup remaining rows (fewer than TILE_M).
    // SAFETY: `r < nrows`; the loads read `[r*lda + c, +LANE_COUNT)` (`c <
    // simd_len`) and the masked tail `[simd_len, ncols)` within the validated `A`
    // span, and `x` within `x.len() >= ncols`.
    while r < nrows {
        unsafe {
            let mut acc = Arch::zero();
            let mut c = 0;
            while c < simd_len {
                let x_vec = load(x_slice.as_ptr().add(c));
                let a_vec = load(a_slice.as_ptr().add(r * lda + c));
                acc = Arch::fmadd(a_vec, x_vec, acc);
                c += lane_count;
            }
            if tail > 0 {
                let a_tail = Arch::masked_load_unaligned(
                    a_slice.as_ptr().add(r * lda + simd_len),
                    tail_mask,
                    Arch::zero(),
                );
                acc = Arch::fmadd(a_tail, x_tail, acc);
            }
            y[r] += Arch::sum_reduce(acc);
        }
        r += 1;
    }

    Ok(())
}
