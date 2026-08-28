//! Register-blocked transposed matrix–vector product (`y += Aᵀ · x`).
//!
//! # Theorem (output reuse, reduction-free vectorization)
//! For `A` row-major `nrows × ncols`, `Aᵀx` is `y[j] = Σᵢ A[i,j]·xᵢ`, equivalently
//! `y += Σᵢ xᵢ · A[i,:]` — the sum of the **rows** of `A` scaled by the entries
//! of `x`. Because each row `A[i,:]` is contiguous, this update vectorizes across
//! the `ncols` output lanes with **no horizontal reduction** (the dual `A·x`
//! needs one reduction per output element; `Aᵀx` needs none). Holding `TILE_N`
//! output lane-chunks of `y` in registers across all `nrows` reuses each
//! accumulator `nrows` times — `y` is loaded and stored once per chunk rather
//! than once per row — and the `TILE_N` independent chunks break the per-chunk
//! FMA dependency chain. The `ncols mod lane` columns use one partial-memory
//! masked-FMA tail, so any shape is supported without reading or writing beyond
//! the live slice. ∎
//!
//! # Safety
//!
//! The `Arch::*` kernels are `#[target_feature]`-gated and sound only on a host
//! implementing `Arch` — established by the `SimdView` operands, whose
//! constructor rejects an unsupported architecture. The raw-pointer loads/stores
//! address `A`, `x`, and `y` at offsets derived from the caller-supplied dims;
//! `check_gemv_t_dimensions` validates those against the actual operand lengths
//! (overflow rejected) before any unchecked access, so per-site `SAFETY`
//! comments cite that validation and the per-window bound.

use crate::{
    align::Alignment,
    arch::SimdArch,
    kernel::{SimdArith, SimdLoadStore, SimdMask},
    scalar::Scalar,
    view::{SimdError, SimdView},
};

#[inline(never)]
fn check_gemv_t_dimensions(
    a_len: usize,
    x_len: usize,
    y_len: usize,
    nrows: usize,
    ncols: usize,
    lda: usize,
) -> Result<(), SimdError> {
    // Shared sub-matrix span (SSOT with the forward GEMV checker); overflow ⇒
    // reject, closing the OOB-load path under release `overflow-checks = false`.
    let a_needed =
        super::dims::checked_strided_span(nrows, ncols, lda).ok_or(SimdError::LengthMismatch)?;
    if lda < ncols || a_len < a_needed || x_len < nrows || y_len < ncols {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

/// Compute `y += Aᵀ · x` with `A` row-major `nrows × ncols` (packed: row stride
/// `ncols`), `x` length `nrows`, `y` length `ncols`, blocking `TILE_N` output
/// lane-chunks so each `y` accumulator is reused across all rows.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if the operand spans are too small for the dims.
#[inline]
pub(super) fn gemv_transpose_impl<T, Arch, Align, const TILE_N: usize>(
    a: &SimdView<'_, T, Arch, Align>,
    x: &SimdView<'_, T, Arch, Align>,
    y: &mut [T],
    nrows: usize,
    ncols: usize,
) -> Result<(), SimdError>
where
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdMask<T>,
    Align: Alignment,
    T: Scalar,
{
    gemv_transpose_strided_impl::<T, Arch, Align, TILE_N>(a, x, y, nrows, ncols, ncols)
}

/// Compute `y += Aᵀ · x` where `A` is a row-major **sub-matrix**: `nrows × ncols`
/// with row stride `lda ≥ ncols` (rows contiguous over `ncols`, spaced `lda`
/// apart). `lda = ncols` is the packed [`gemv_transpose_impl`].
///
/// # Errors
/// [`SimdError::LengthMismatch`] if `lda < ncols` or the operand spans are too
/// small (`a` needs `(nrows−1)·lda + ncols`, `x ≥ nrows`, `y ≥ ncols`).
#[inline]
pub(super) fn gemv_transpose_strided_impl<T, Arch, Align, const TILE_N: usize>(
    a: &SimdView<'_, T, Arch, Align>,
    x: &SimdView<'_, T, Arch, Align>,
    y: &mut [T],
    nrows: usize,
    ncols: usize,
    lda: usize,
) -> Result<(), SimdError>
where
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdMask<T>,
    Align: Alignment,
    T: Scalar,
{
    struct AssertN<const TILE_N: usize>;
    impl<const TILE_N: usize> AssertN<TILE_N> {
        const OK: () = assert!(TILE_N >= 1, "TILE_N must be at least 1");
    }
    let () = AssertN::<TILE_N>::OK;

    check_gemv_t_dimensions(a.len(), x.len(), y.len(), nrows, ncols, lda)?;

    let a_slice = a.as_slice();
    let x_slice = x.as_slice();

    let lane_count = Arch::LANE_COUNT;
    let simd_cols = (ncols / lane_count) * lane_count;

    // SAFETY: target-feature kernel (module invariant); every call passes a
    // pointer whose `LANE_COUNT` read stays within `A` (offsets bounded below by
    // `simd_cols` and the validated stride), gated on `Align` for the aligned form.
    let load = |ptr: *const T| -> Arch::Vector {
        if crate::align::is_aligned_for_arch::<Arch, Align>() {
            unsafe { Arch::load_aligned(ptr) }
        } else {
            unsafe { Arch::load_unaligned(ptr) }
        }
    };

    // Block TILE_N output lane-chunks; each accumulator is reused over all rows.
    // `y` is a caller slice of unknown alignment, so load/store it unaligned.
    let mut c = 0;
    // SAFETY: the loop guard keeps `c + TILE_N*LANE_COUNT <= simd_cols <= ncols`,
    // so every `y`/`A` window `[c + t*LANE_COUNT, +LANE_COUNT)` (t < TILE_N) is
    // within `y` (len >= ncols) and, at row base `i*lda` for `i < nrows`, within
    // the validated `A` span; `x_slice[i]` is checked indexing. Kernels covered
    // by the module invariant.
    while c + TILE_N * lane_count <= simd_cols {
        unsafe {
            let mut acc = [Arch::zero(); TILE_N];
            for (t, slot) in acc.iter_mut().enumerate() {
                *slot = Arch::load_unaligned(y.as_ptr().add(c + t * lane_count));
            }
            for i in 0..nrows {
                let xi = Arch::splat(x_slice[i]);
                let base = i * lda + c;
                for (t, slot) in acc.iter_mut().enumerate() {
                    let a_vec = load(a_slice.as_ptr().add(base + t * lane_count));
                    *slot = Arch::fmadd(xi, a_vec, *slot);
                }
            }
            for (t, &accv) in acc.iter().enumerate() {
                Arch::store_unaligned(y.as_mut_ptr().add(c + t * lane_count), accv);
            }
        }
        c += TILE_N * lane_count;
    }

    // Remaining single lane-chunks (fewer than TILE_N).
    // SAFETY: `c < simd_cols <= ncols`, so the `y`/`A` window `[c, c+LANE_COUNT)`
    // is within `y` and the validated `A` span at each row `i < nrows`.
    while c < simd_cols {
        unsafe {
            let mut acc = Arch::load_unaligned(y.as_ptr().add(c));
            for i in 0..nrows {
                let xi = Arch::splat(x_slice[i]);
                let a_vec = load(a_slice.as_ptr().add(i * lda + c));
                acc = Arch::fmadd(xi, a_vec, acc);
            }
            Arch::store_unaligned(y.as_mut_ptr().add(c), acc);
        }
        c += lane_count;
    }

    // Masked tail (ncols not a multiple of the lane count).
    //
    // The partial-memory contract keeps inactive lanes from accessing `a` or
    // `y`. The masked FMA preserves the same provider-owned fused arithmetic
    // seam as full vectors; non-dyadic results may therefore differ from scalar
    // multiply-plus-add by one rounding step.
    if simd_cols < ncols {
        let tail = ncols - simd_cols;
        // SAFETY: `tail < lane_count` and the validated output span contains
        // exactly the accessible suffix `[simd_cols, ncols)`.
        let mask = unsafe { Arch::leading_k_mask(tail) };
        let mut acc = unsafe {
            Arch::masked_load_partial(y.as_ptr().add(simd_cols), tail, mask, Arch::zero())
        };

        for i in 0..nrows {
            let row_start = i * lda + simd_cols;
            // SAFETY: dimension validation proves the row contains exactly this
            // accessible suffix and the leading mask selects no other lanes.
            let a_tail = unsafe {
                Arch::masked_load_partial(a_slice.as_ptr().add(row_start), tail, mask, Arch::zero())
            };
            // SAFETY: all operands are valid registers; `mask` selects the live
            // tail lanes and inactive lanes retain the accumulator. `x[i]` is
            // broadcast because every output tail lane uses the same row scalar.
            let xi = unsafe { Arch::splat(x_slice[i]) };
            acc = unsafe { Arch::masked_fmadd(xi, a_tail, acc, mask) };
        }

        // SAFETY: identical validated suffix and mask to the load above.
        unsafe { Arch::masked_store_partial(y.as_mut_ptr().add(simd_cols), tail, mask, acc) };
    }

    Ok(())
}
