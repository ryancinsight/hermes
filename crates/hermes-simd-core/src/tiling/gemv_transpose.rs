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
//! FMA dependency chain. The `ncols mod lane` columns use a scalar tail, so any
//! shape is supported. ∎

use crate::{
    align::Alignment,
    arch::SimdArch,
    kernel::SimdKernel,
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
) -> Result<(), SimdError> {
    if a_len < nrows * ncols || x_len < nrows || y_len < ncols {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

/// Compute `y += Aᵀ · x` with `A` row-major `nrows × ncols`, `x` length `nrows`,
/// `y` length `ncols`, blocking `TILE_N` output lane-chunks so each `y`
/// accumulator is reused across all rows.
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
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    struct AssertN<const TILE_N: usize>;
    impl<const TILE_N: usize> AssertN<TILE_N> {
        const OK: () = assert!(TILE_N >= 1, "TILE_N must be at least 1");
    }
    let _ = AssertN::<TILE_N>::OK;

    check_gemv_t_dimensions(a.len(), x.len(), y.len(), nrows, ncols)?;

    let a_slice = a.as_slice();
    let x_slice = x.as_slice();

    let lane_count = Arch::LANE_COUNT;
    let simd_cols = (ncols / lane_count) * lane_count;

    let load = |ptr: *const T| -> Arch::Vector {
        if Align::IS_ALIGNED {
            unsafe { Arch::load_aligned(ptr) }
        } else {
            unsafe { Arch::load_unaligned(ptr) }
        }
    };

    // Block TILE_N output lane-chunks; each accumulator is reused over all rows.
    // `y` is a caller slice of unknown alignment, so load/store it unaligned.
    let mut c = 0;
    while c + TILE_N * lane_count <= simd_cols {
        let mut acc = [unsafe { Arch::zero() }; TILE_N];
        for (t, slot) in acc.iter_mut().enumerate() {
            *slot = unsafe { Arch::load_unaligned(y.as_ptr().add(c + t * lane_count)) };
        }
        for i in 0..nrows {
            let xi = unsafe { Arch::splat(x_slice[i]) };
            let base = i * ncols + c;
            for (t, slot) in acc.iter_mut().enumerate() {
                let a_vec = load(unsafe { a_slice.as_ptr().add(base + t * lane_count) });
                *slot = unsafe { Arch::fmadd(xi, a_vec, *slot) };
            }
        }
        for (t, &accv) in acc.iter().enumerate() {
            unsafe { Arch::store_unaligned(y.as_mut_ptr().add(c + t * lane_count), accv) };
        }
        c += TILE_N * lane_count;
    }

    // Remaining single lane-chunks (fewer than TILE_N).
    while c < simd_cols {
        let mut acc = unsafe { Arch::load_unaligned(y.as_ptr().add(c)) };
        for i in 0..nrows {
            let xi = unsafe { Arch::splat(x_slice[i]) };
            let a_vec = load(unsafe { a_slice.as_ptr().add(i * ncols + c) });
            acc = unsafe { Arch::fmadd(xi, a_vec, acc) };
        }
        unsafe { Arch::store_unaligned(y.as_mut_ptr().add(c), acc) };
        c += lane_count;
    }

    // Scalar tail (ncols not a multiple of the lane count).
    for c_tail in simd_cols..ncols {
        let mut s = y[c_tail];
        for i in 0..nrows {
            s = s + x_slice[i] * a_slice[i * ncols + c_tail];
        }
        y[c_tail] = s;
    }

    Ok(())
}
