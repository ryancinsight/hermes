//! Register-blocked, cache-aware matrix multiplication (`C += A · B`).
//!
//! Row-major operands: `A` is `m × k`, `B` is `k × n`, `C` is `m × n`. The
//! kernel is the textbook Goto/BLIS layering — an inner register micro-kernel
//! (`gemm_register_tile`) wrapped by a packing/loop-order policy (`gemm_impl`)
//! that keeps the reused operand cache-resident.
//!
//! # Theorem 1 (correctness)
//! On return `C[i,j] = C₀[i,j] + Σ_{p=0}^{k-1} A[i,p]·B[p,j]` for every
//! `0 ≤ i < m`, `0 ≤ j < n`, where `C₀` is the input value of `C`.
//!
//! *Proof.* The column range `[0, n)` is partitioned into full register panels
//! `[col, col+block_n)` with `block_n = TILE_N·LANE_COUNT` plus the tail
//! `[simd_n_len, n)`; the row range `[0, m)` into register blocks of `TILE_M`
//! (last block possibly short, `current_tile_m`). Every `(i,j)` lies in exactly
//! one (row-block, column-panel-or-tail) cell. For panel cells
//! `gemm_register_tile` loads `C[i, j]`, executes
//! `acc ← acc + Σ_p A[i,p]·B[p,j]` over all `p ∈ [0,k)`, and stores `acc`; tail
//! cells run the identical fmadd recurrence through `leading_k_mask`-guarded
//! lane groups (inactive lanes load zero, accumulate `a·0`, and are excluded
//! from the masked store, so they touch no cell). Cells are disjoint, so each
//! `C[i,j]` is read-modify-written exactly once with the complete `p`-sum. ∎
//!
//! # Theorem 2 (packing invariance)
//! For fixed `TILE_M, TILE_N` the packed and in-place paths produce
//! bit-identical `C`.
//!
//! *Proof.* Packing copies `B[p, col+t] ↦ Bp[p·block_n + t]` (a bijective
//! relayout: values are preserved, only addresses change), and the micro-kernel
//! reads `Bp[p·block_n + j·LANE_COUNT + l]` exactly where the in-place path reads
//! `B[p·n + col + j·LANE_COUNT + l]`. For each accumulator `acc[i][j]` the
//! sequence of operations is `acc ← fmadd(splat(A[i,p]), Bᵥ(p), acc)` for
//! `p = 0,1,…,k−1` in that order in **both** paths — identical operands in
//! identical order. IEEE-754 FMA is a deterministic function of its operands, so
//! the rounded results coincide bit-for-bit; the loop reorder (panel-outer vs
//! row-outer) only permutes writes to disjoint cells (Theorem 1). ∎
//!
//! # Theorem 3 (register residency)
//! The micro-kernel's hot loop holds `TILE_M·TILE_N + TILE_N + 1` vector values
//! live; it avoids spills iff that count `≤ REGS` (the architectural vector
//! register file: 16 YMM on AVX2, 32 ZMM on AVX-512).
//!
//! *Proof.* Across the `p`-loop the `TILE_M·TILE_N` accumulators are
//! loop-carried (live throughout). Each iteration materializes `TILE_N` B-vectors
//! (reused over the `TILE_M` rows) and broadcasts one A-scalar at a time
//! (`a_reg`), so at most `TILE_N + 1` further vectors are simultaneously live.
//! Summing gives the bound; exceeding `REGS` forces the allocator to spill an
//! accumulator to the stack each iteration, reintroducing memory traffic into the
//! hot loop. For AVX2 f64 this selects `TILE_M=TILE_N=3` (9+3+1 = 13 ≤ 16, with
//! headroom for loop temporaries); `<3,4>` (17) and `<4,3>` (16, zero headroom)
//! both spill. ∎
//!
//! # Theorem 4 (cache cost model for packing)
//! In-place GEMM re-reads each B column panel `⌈m/TILE_M⌉` times. If `B`
//! (`k·n·sizeof(T)` bytes) exceeds the last private cache, each pass refetches it
//! from a slower level, so B traffic is `Θ(m·k·n/TILE_M)` slow-memory words.
//! Packing copies `B` once into a contiguous `k·block_n` panel that fits L1 and is
//! reused from L1 across all row blocks, reducing slow-memory B traffic to the
//! single `Θ(k·n)` pack pass. Packing therefore wins exactly when `B` does not fit
//! cache; below that threshold the pack's copy + allocation is pure overhead. The
//! crossover is encoded by `GEMM_PACK_B_BYTES_THRESHOLD` (gated additionally on
//! `m > TILE_M`, since with a single row block there is no reuse to amortize). ∎
//!
//! # Safety
//!
//! The `Arch::*` kernels are `#[target_feature]`-gated and sound only on a host
//! implementing `Arch` — established by the `SimdView` operands, whose
//! constructor rejects an unsupported architecture. `check_tiled_gemm_dimensions`
//! validates the caller-supplied `m`, `n`, `k` against the actual `A`/`B`/`C`
//! lengths (overflow rejected, closing the OOB path under release
//! `overflow-checks = false`) before the register micro-kernel runs, so every
//! tile offset `(r+i)*n + col_n + j*LANE_COUNT` and its `A`/`B` counterparts stay
//! within the validated spans; `gemm_register_tile` carries the full contract in
//! its own `# Safety` section.

use crate::{
    align::Alignment,
    arch::SimdArch,
    kernel::SimdKernel,
    scalar::Scalar,
    vec::AlignedVec,
    view::{SimdError, SimdView},
};

/// Byte threshold of the right-hand operand `B` (`k × n`) above which
/// [`gemm_impl`] packs B column panels into a contiguous scratch (Theorem 4).
///
/// Set to a conservative 512 KiB — at or below a typical L2 — so the crossover is
/// taken slightly early rather than late. Measured f64 on AVX2: ~6% faster at
/// 512² (B = 2 MiB), no benefit and a small loss below (B resident in L2).
pub(super) const GEMM_PACK_B_BYTES_THRESHOLD: usize = 512 * 1024;

#[inline(never)]
fn check_tiled_gemm_dimensions(
    a_len: usize,
    b_len: usize,
    c_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), SimdError> {
    // Overflow in any operand area ⇒ reject, closing the OOB load/store path under
    // release `overflow-checks = false` (see `tiling::dims`).
    let a_needed = super::dims::checked_area(m, k).ok_or(SimdError::LengthMismatch)?;
    let b_needed = super::dims::checked_area(k, n).ok_or(SimdError::LengthMismatch)?;
    let c_needed = super::dims::checked_area(m, n).ok_or(SimdError::LengthMismatch)?;
    if a_len < a_needed || b_len < b_needed || c_len < c_needed {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

/// Compute one `TILE_M × (TILE_N·LANE_COUNT)` register tile of `C += A · B`.
///
/// `b_base`/`b_row_stride` abstract over the B source: either the in-place matrix
/// (row stride `n`) or a packed contiguous column panel (row stride `block_n`).
/// A single monomorphized micro-kernel therefore serves both code paths (SSOT —
/// no second contraction implementation). See module Theorem 3 for the register
/// budget realized by this loop nest.
///
/// # Safety
/// `c` is a row-major `m × n` matrix whose tile at `(r, col_n)` with
/// `current_tile_m ≤ TILE_M` rows and `TILE_N·LANE_COUNT` columns lies within it;
/// `a_slice` is a row-major `m × k` matrix; `b_base` addresses at least `k` rows
/// of `TILE_N·LANE_COUNT` contiguous elements at stride `b_row_stride`. The
/// caller's [`check_tiled_gemm_dimensions`] guarantees these spans.
// The arguments (A/C operands, tile origin `r`/`col_n`, partial-row count, the
// `n`/`k` dimensions, and the abstracted B source `b_base`/`b_row_stride`) are all
// load-bearing inputs to a hot micro-kernel; bundling them into a struct would add
// indirection to the inner loop for no clarity gain.
#[expect(
    clippy::too_many_arguments,
    reason = "The hot micro-kernel keeps each matrix operand and tile coordinate explicit"
)]
#[inline(always)]
unsafe fn gemm_register_tile<T, Arch, const TILE_M: usize, const TILE_N: usize>(
    a_slice: &[T],
    c: &mut [T],
    r: usize,
    current_tile_m: usize,
    col_n: usize,
    n: usize,
    k: usize,
    b_base: *const T,
    b_row_stride: usize,
) where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    let lane_count = Arch::LANE_COUNT;
    let mut accumulators = [[unsafe { Arch::zero() }; TILE_N]; TILE_M];

    for i in 0..current_tile_m {
        let row_idx = r + i;
        for j in 0..TILE_N {
            let c_ptr = unsafe { c.as_ptr().add(row_idx * n + col_n + j * lane_count) };
            accumulators[i][j] = unsafe { Arch::load_unaligned(c_ptr) };
        }
    }

    for kk in 0..k {
        let mut b_regs = [unsafe { Arch::zero() }; TILE_N];
        for j in 0..TILE_N {
            let b_ptr = unsafe { b_base.add(kk * b_row_stride + j * lane_count) };
            b_regs[j] = unsafe { Arch::load_unaligned(b_ptr) };
        }
        for i in 0..current_tile_m {
            let a_reg = unsafe { Arch::splat(a_slice[(r + i) * k + kk]) };
            for j in 0..TILE_N {
                accumulators[i][j] = unsafe { Arch::fmadd(a_reg, b_regs[j], accumulators[i][j]) };
            }
        }
    }

    for i in 0..current_tile_m {
        let row_idx = r + i;
        for j in 0..TILE_N {
            let c_ptr = unsafe { c.as_mut_ptr().add(row_idx * n + col_n + j * lane_count) };
            unsafe { Arch::store_unaligned(c_ptr, accumulators[i][j]) };
        }
    }
}

/// Compute `C += A · B` (row-major) with register blocking and size-gated
/// B-panel packing. See the module theorems for correctness (1), packing
/// invariance (2), register residency (3), and the packing cost model (4).
///
/// # Errors
/// [`SimdError::LengthMismatch`] if the operand spans are too small for the dims.
#[inline]
pub(super) fn gemm_impl<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
    a: &SimdView<'_, T, Arch, Align>,
    b: &SimdView<'_, T, Arch, Align>,
    c: &mut [T],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), SimdError>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    struct AssertGEMM<const M: usize, const N: usize>;
    impl<const M: usize, const N: usize> AssertGEMM<M, N> {
        const OK: () = {
            assert!(M >= 1, "TILE_M must be at least 1");
            assert!(N >= 1, "TILE_N must be at least 1");
            assert!(M * N <= 64, "TILE_M * TILE_N must be <= 64");
        };
    }
    let _ = AssertGEMM::<TILE_M, TILE_N>::OK;

    check_tiled_gemm_dimensions(a.len(), b.len(), c.len(), m, n, k)?;

    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    let lane_count = Arch::LANE_COUNT;
    let block_n = TILE_N * lane_count;
    let simd_n_len = (n / block_n) * block_n;

    // Packed path (Theorem 4): pack each B column panel once and reuse it across
    // all row blocks. Gated on B exceeding cache and on there being more than one
    // row block; smaller problems take the in-place path with no allocation.
    let b_bytes = n
        .saturating_mul(k)
        .saturating_mul(core::mem::size_of::<T>());
    if m > TILE_M && simd_n_len > 0 && b_bytes >= GEMM_PACK_B_BYTES_THRESHOLD {
        let mut packed = AlignedVec::<T, crate::align::Aligned<64>>::with_capacity(k * block_n);
        unsafe {
            packed.set_len(k * block_n);
        }
        let mut col_n = 0;
        while col_n < simd_n_len {
            for kk in 0..k {
                // SAFETY: `kk < k` and `col_n + block_n ≤ simd_n_len ≤ n`, so the
                // source span `[kk*n+col_n, +block_n)` lies within `b` (validated by
                // `check_tiled_gemm_dimensions`) and the destination
                // `[kk*block_n, +block_n)` within the `k*block_n` scratch.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        b_slice.as_ptr().add(kk * n + col_n),
                        packed.as_mut_ptr().add(kk * block_n),
                        block_n,
                    );
                }
            }

            let mut r = 0;
            while r < m {
                let current_tile_m = if r + TILE_M <= m { TILE_M } else { m - r };
                // SAFETY: the packed panel holds `k` rows of `block_n` contiguous
                // elements (row stride `block_n`); the C tile at `(r, col_n)` is
                // within the validated `m × n` output.
                unsafe {
                    gemm_register_tile::<T, Arch, TILE_M, TILE_N>(
                        a_slice,
                        c,
                        r,
                        current_tile_m,
                        col_n,
                        n,
                        k,
                        packed.as_ptr(),
                        block_n,
                    );
                }
                r += TILE_M;
            }
            col_n += block_n;
        }
    } else {
        let mut r = 0;
        while r < m {
            let current_tile_m = if r + TILE_M <= m { TILE_M } else { m - r };
            let mut col_n = 0;
            while col_n < simd_n_len {
                // SAFETY: B is read in place at row stride `n`; `col_n + block_n ≤ n`
                // and the C tile lies within the validated `m × n` output.
                unsafe {
                    gemm_register_tile::<T, Arch, TILE_M, TILE_N>(
                        a_slice,
                        c,
                        r,
                        current_tile_m,
                        col_n,
                        n,
                        k,
                        b_slice.as_ptr().add(col_n),
                        n,
                    );
                }
                col_n += block_n;
            }
            r += TILE_M;
        }
    }

    // Masked-vector cleanup for the `n % block_n` trailing columns: process them
    // in `lane_count`-wide groups with a `leading_k_mask` on the final partial
    // group, `TILE_M`-row blocked so each masked B row-load is reused across the
    // row block — the same fmadd contraction (and therefore the same
    // fused-multiply rounding) as the register tiles. Previously these columns
    // ran a strided scalar triple loop, costing up to `block_n − 1` scalar
    // columns (≈ half the FLOPs for `n` just under a block multiple). Inactive
    // mask lanes load as zero and are never stored: `a·0` accumulates into a
    // zero-initialized lane that the masked store discards, so they cannot
    // contaminate active lanes. Each (row, col) is written exactly once,
    // independent of the tiling above.
    let mut col = simd_n_len;
    while col < n {
        let w = core::cmp::min(lane_count, n - col);
        // SAFETY: `w ≤ LANE_COUNT` by construction of `min`.
        let mask = unsafe { Arch::leading_k_mask(w) };
        let mut r = 0;
        while r < m {
            let current_tile_m = if r + TILE_M <= m { TILE_M } else { m - r };
            let mut acc = [unsafe { Arch::zero() }; TILE_M];
            for (i, slot) in acc.iter_mut().take(current_tile_m).enumerate() {
                // SAFETY: row `r + i < m` and the masked load touches only lanes
                // `[col, col + w)` with `col + w ≤ n`, inside the validated
                // `m × n` C span.
                *slot = unsafe {
                    Arch::masked_load_unaligned(
                        c.as_ptr().add((r + i) * n + col),
                        mask,
                        Arch::zero(),
                    )
                };
            }
            for kk in 0..k {
                // SAFETY: masked load of lanes `[col, col + w) ≤ n` in row
                // `kk < k` of the validated `k × n` B span.
                let b_reg = unsafe {
                    Arch::masked_load_unaligned(
                        b_slice.as_ptr().add(kk * n + col),
                        mask,
                        Arch::zero(),
                    )
                };
                for (i, slot) in acc.iter_mut().take(current_tile_m).enumerate() {
                    // SAFETY: pure register ops; `a_slice[(r+i)*k + kk]` is inside
                    // the validated `m × k` A span.
                    unsafe {
                        let a_reg = Arch::splat(a_slice[(r + i) * k + kk]);
                        *slot = Arch::fmadd(a_reg, b_reg, *slot);
                    }
                }
            }
            for (i, slot) in acc.iter().take(current_tile_m).enumerate() {
                // SAFETY: identical span to the masked load above; only the first
                // `w` lanes are written.
                unsafe {
                    Arch::masked_store_unaligned(
                        c.as_mut_ptr().add((r + i) * n + col),
                        mask,
                        *slot,
                    );
                };
            }
            r += TILE_M;
        }
        col += lane_count;
    }

    Ok(())
}
