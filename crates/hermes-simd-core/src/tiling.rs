//! Register-blocked tiling for high-throughput dot products and GEMV kernels.
//!
//! # Motivation
//!
//! A standard `dot` loop has a loop-carried FMA dependency chain that limits throughput
//! to one FMA per cycle. Unrolling into `TILE_M` independent accumulator registers breaks
//! this chain, saturating the two-port FMA execution units on modern CPUs:
//!
//! - **AVX-512** (32 ZMM regs): `TILE_M = 8` targets two FMA ports x 4-cycle latency.
//! - **AVX2** (16 YMM regs): `TILE_M = 4` matches `SimdKernel::UNROLL_FACTOR`.
//! - **Scalar**: `TILE_M = 1` is equivalent to the standard loop; const monomorphization
//!   eliminates all tiling overhead at zero cost.
//!
//! The existing `UNROLL_FACTOR` in `SimdKernel` addresses pipeline depth; `TILE_M` here
//! addresses accumulator register pressure separately — these two dimensions compose.

use crate::{
    align::Alignment,
    arch::SimdArch,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
};
use core::marker::PhantomData;

/// Compute the dot product of two slices using `TILE_M` independent vector accumulators.
///
/// The inner loop processes `TILE_M * LANE_COUNT` elements per iteration, holding
/// `TILE_M` accumulator registers simultaneously to saturate FMA throughput.
///
/// # Type Parameters
/// - `T`: scalar element type
/// - `Arch`: SIMD architecture (`Avx512`, `Avx2`, `Neon`, `Scalar`)
/// - `Align`: alignment typestate
/// - `TILE_M`: number of independent accumulator registers (compile-time constant)
///
/// # Performance Notes
/// - `TILE_M = 4` is safe for all architectures (matches `UNROLL_FACTOR`).
/// - `TILE_M = 8` is optimal for AVX-512 with 32 ZMM registers.
/// - Const monomorphization ensures zero overhead for any fixed `TILE_M`.
///
/// # Errors
/// Returns [`SimdError::LengthMismatch`] if `a.len() != b.len()`.
/// Trait representing a monomorphized register-blocking/tiling strategy.
///
/// Implemented as a blanket over `TilingPolicy<TILE_M, TILE_N>`. The const generic
/// parameters encode the tile shape so the compiler emits loop-unrolled, register-blocked
/// kernels with zero runtime overhead.
///
/// # Examples
///
/// ```rust
/// use hermes_simd_core::tiling::{TilingPolicy, TilingStrategy};
/// use hermes_simd_core::view::SimdView;
/// use hermes_simd_core::align::Unaligned;
/// use hermes_simd_intrinsics::Scalar;
///
/// let a = [1.0_f32; 4];
/// let b = [2.0_f32; 4];
/// let va = SimdView::<f32, Scalar, Unaligned>::new(&a).unwrap();
/// let vb = SimdView::<f32, Scalar, Unaligned>::new(&b).unwrap();
/// let dot = <TilingPolicy<1, 1> as TilingStrategy<f32, Scalar, Unaligned>>::dot(&va, &vb)
///     .expect("lengths equal");
/// assert!((dot - 8.0_f32).abs() < 1e-6);
/// ```
pub trait TilingStrategy<T, Arch: SimdArch, Align: Alignment> {
    /// The number of rows in the register block.
    const TILE_M: usize;
    /// The number of columns (vectors of size `LANE_COUNT`) in the register block.
    const TILE_N: usize;

    /// Perform tiled matrix multiplication `c += a * b` using this strategy.
    fn gemm(
        a: &SimdView<'_, T, Arch, Align>,
        b: &SimdView<'_, T, Arch, Align>,
        c: &mut [T],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), SimdError>;

    /// Perform tiled matrix-vector multiplication `y += A * x` using this strategy.
    fn gemv(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError>;

    /// Perform tiled dot product computation using this strategy.
    fn dot(
        a: &SimdView<'_, T, Arch, Align>,
        b: &SimdView<'_, T, Arch, Align>,
    ) -> Result<T, SimdError>;
}

/// Compute the dot product of two slices using `TILE_M` independent vector accumulators.
///
/// The inner loop processes `TILE_M * LANE_COUNT` elements per iteration, holding
/// `TILE_M` accumulator registers simultaneously to saturate FMA throughput.
#[inline(always)]
pub fn tiled_dot<T, Arch, Align, const TILE_M: usize>(
    a: &SimdView<'_, T, Arch, Align>,
    b: &SimdView<'_, T, Arch, Align>,
) -> Result<T, SimdError>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    <TilingPolicy<TILE_M, 1> as TilingStrategy<T, Arch, Align>>::dot(a, b)
}

/// Zero-sized strategy marker for tiled execution policy.
///
/// Encode tile shape in the type system as a ZST so tiling parameters are
/// resolved at compile time with no runtime storage.
///
/// # Examples
///
/// Dot product via the `tiled_dot` free function (preferred API):
///
/// ```rust
/// use hermes_simd_core::tiling::tiled_dot;
/// use hermes_simd_core::view::SimdView;
/// use hermes_simd_core::align::Unaligned;
/// use hermes_simd_intrinsics::Scalar;
///
/// let a = [1.0_f32, 2.0, 3.0, 4.0];
/// let b = [1.0_f32, 1.0, 1.0, 1.0];
/// let va = SimdView::<f32, Scalar, Unaligned>::new(&a).unwrap();
/// let vb = SimdView::<f32, Scalar, Unaligned>::new(&b).unwrap();
/// // TILE_M = 4 unrolls into 4 independent FMA accumulators.
/// let result = tiled_dot::<f32, Scalar, Unaligned, 4>(&va, &vb).unwrap();
/// assert!((result - 10.0_f32).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilingPolicy<const TILE_M: usize, const TILE_N: usize>;

impl<const TILE_M: usize, const TILE_N: usize> TilingPolicy<TILE_M, TILE_N> {
    /// Standard tile for AVX2 (4 accumulators x 8 f32 lanes = 256 bits).
    pub const AVX2_STANDARD: TilingPolicy<4, 4> = TilingPolicy;
    /// Optimal tile for AVX-512 (8 accumulators x 16 f32 lanes = 1024 bits).
    pub const AVX512_OPTIMAL: TilingPolicy<8, 4> = TilingPolicy;
    /// Scalar degenerate tile: `TILE_M = 1`, `TILE_N = 1` — no tiling overhead.
    ///
    /// Named `SCALAR_DEGENERATE` (SCREAMING_SNAKE_CASE) to satisfy Rust naming lints.
    pub const SCALAR_DEGENERATE: TilingPolicy<1, 1> = TilingPolicy;

    /// Verify that the tile shape is valid at compile time.
    pub const fn validate(&self) {
        assert!(TILE_M >= 1, "TILE_M must be >= 1");
        assert!(TILE_N >= 1, "TILE_N must be >= 1");
    }
}

/// Zero-sized type used to force `TilingPolicy` into a generic bound without runtime cost.
pub struct _TileMarker<const M: usize, const N: usize>(PhantomData<TilingPolicy<M, N>>);

/// Byte threshold of the right-hand operand `B` (`k × n`) above which GEMM packs
/// B column panels into a contiguous scratch before the register kernel.
///
/// Below it `B` stays resident in L2 across the `⌈m / TILE_M⌉` row-block passes,
/// so packing only adds a copy and an allocation (measured ~50% slower at 64²).
/// Above it `B` is evicted between passes and re-streamed from L3/DRAM, so the
/// one-pass pack pays for itself (measured ~6% faster at 512² f64, growing with
/// size). Set to a conservative 512 KiB — at or below a typical L2 — so the
/// crossover is taken slightly early rather than late. The pack is also gated on
/// `m > TILE_M` (more than one row block, otherwise there is no reuse to exploit).
const GEMM_PACK_B_BYTES_THRESHOLD: usize = 512 * 1024;

#[inline(never)]
fn check_gemv_dimensions(
    a_len: usize,
    x_len: usize,
    y_len: usize,
    nrows: usize,
    ncols: usize,
) -> Result<(), SimdError> {
    if a_len < nrows * ncols || x_len < ncols || y_len < nrows {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

/// Compute a register-blocked tiled GEMV: `y += A * x`.
///
/// Processes `TILE_M` rows of `A` simultaneously to reuse loaded elements of `x` across those rows.
/// Accepts `SimdView`-typed operands to enforce alignment typestates at the call boundary.
#[inline(always)]
pub fn tiled_gemv<T, Arch, Align, const TILE_M: usize>(
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
    <TilingPolicy<TILE_M, 1> as TilingStrategy<T, Arch, Align>>::gemv(a, x, y, nrows, ncols)
}

#[inline(never)]
fn check_tiled_gemm_dimensions(
    a_len: usize,
    b_len: usize,
    c_len: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), SimdError> {
    if a_len < m * k || b_len < k * n || c_len < m * n {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

/// Compute a register-blocked tiled GEMM: `c += A * B`.
///
/// Multiplies matrix `a` (dimensions `m * k`) and matrix `b` (dimensions `k * n`),
/// accumulating the result into `c` (dimensions `m * n`).
#[inline(always)]
pub fn tiled_gemm<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
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
    <TilingPolicy<TILE_M, TILE_N> as TilingStrategy<T, Arch, Align>>::gemm(a, b, c, m, n, k)
}

/// Compute one `TILE_M × (TILE_N·LANE_COUNT)` register tile of `C += A · B`.
///
/// `b_base`/`b_row_stride` abstract over the B source: either the in-place matrix
/// (row stride `n`) or a packed contiguous column panel (row stride `block_n`).
/// A single monomorphized micro-kernel therefore serves both the direct and the
/// packed code paths (SSOT — no second contraction implementation). The A operand
/// is broadcast on the fly (one live register); each k-row of B is loaded into
/// `TILE_N` vector registers and reused across all `TILE_M` output rows, so the
/// live register set is `TILE_M·TILE_N` accumulators + `TILE_N` B-vectors + 1
/// A-scalar (sized with the tile shape to fit the architectural register file).
///
/// # Safety
/// `c` is a row-major `m × n` matrix whose tile at `(r, col_n)` with
/// `current_tile_m ≤ TILE_M` rows and `TILE_N·LANE_COUNT` columns lies within it;
/// `a_slice` is a row-major `m × k` matrix; `b_base` addresses at least `k` rows of
/// `TILE_N·LANE_COUNT` contiguous elements at stride `b_row_stride`. The caller's
/// [`check_tiled_gemm_dimensions`] guarantees these spans.
// The arguments (A/C operands, tile origin `r`/`col_n`, partial-row count, the
// `n`/`k` dimensions, and the abstracted B source `b_base`/`b_row_stride`) are all
// load-bearing inputs to a hot micro-kernel; bundling them into a struct would add
// indirection to the inner loop for no clarity gain.
#[allow(clippy::too_many_arguments)]
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

impl<T, Arch, Align, const TILE_M: usize, const TILE_N: usize> TilingStrategy<T, Arch, Align>
    for TilingPolicy<TILE_M, TILE_N>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    const TILE_M: usize = TILE_M;
    const TILE_N: usize = TILE_N;

    #[inline]
    fn gemm(
        a: &SimdView<'_, T, Arch, Align>,
        b: &SimdView<'_, T, Arch, Align>,
        c: &mut [T],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), SimdError> {
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

        // Loop order and B-panel packing (Goto/BLIS layered GEMM).
        //
        // Every TILE_M-row block consumes the same `block_n`-wide column panel of
        // B. Reading it in place re-streams that panel `⌈m / TILE_M⌉` times over a
        // stride-`n` access pattern (only `block_n` of every `n` row elements are
        // touched), so it is evicted and refetched from L2/L3 on each pass —
        // bandwidth-bound. Packing the panel once into a contiguous `k × block_n`
        // scratch (sequential, TLB-friendly) lets all row blocks reuse it from L1,
        // returning the kernel toward compute-bound. The pack is a single O(k·n)
        // pass amortized over O(m·n·k) FMAs, so it pays once there is more than one
        // row block (`m > TILE_M`); smaller problems take the in-place path with no
        // allocation. The per-accumulator FMA order is identical in both paths, so
        // results are bitwise-identical regardless of packing.
        let b_bytes = n
            .saturating_mul(k)
            .saturating_mul(core::mem::size_of::<T>());
        if m > TILE_M && simd_n_len > 0 && b_bytes >= GEMM_PACK_B_BYTES_THRESHOLD {
            let mut packed = alloc::vec![T::ZERO; k * block_n];
            let mut col_n = 0;
            while col_n < simd_n_len {
                for kk in 0..k {
                    // SAFETY: `kk < k` and `col_n + block_n ≤ simd_n_len ≤ n`, so the
                    // source span `[kk*n+col_n, +block_n)` lies within `b` (validated
                    // by `check_tiled_gemm_dimensions`) and the destination
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

        // Scalar cleanup for the `n % block_n` trailing columns (all rows). Each
        // (row, col) is written exactly once, independent of the tiling above.
        for col_tail in simd_n_len..n {
            for row in 0..m {
                let mut sum = T::ZERO;
                for kk in 0..k {
                    sum += a_slice[row * k + kk] * b_slice[kk * n + col_tail];
                }
                c[row * n + col_tail] += sum;
            }
        }

        Ok(())
    }

    #[inline]
    fn gemv(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError> {
        struct AssertM<const TILE_M: usize>;
        impl<const TILE_M: usize> AssertM<TILE_M> {
            const OK: () = assert!(TILE_M >= 1, "TILE_M must be at least 1");
        }
        let _ = AssertM::<TILE_M>::OK;

        check_gemv_dimensions(a.len(), x.len(), y.len(), nrows, ncols)?;

        let a_slice = a.as_slice();
        let x_slice = x.as_slice();

        let lane_count = Arch::LANE_COUNT;
        let simd_len = (ncols / lane_count) * lane_count;

        let load = |ptr: *const T| -> Arch::Vector {
            if Align::IS_ALIGNED {
                unsafe { Arch::load_aligned(ptr) }
            } else {
                unsafe { Arch::load_unaligned(ptr) }
            }
        };

        let mut r = 0;
        while r + TILE_M <= nrows {
            // Initialize TILE_M accumulators to zero
            let mut accumulators = [unsafe { Arch::zero() }; TILE_M];

            let mut c = 0;
            while c < simd_len {
                // Load x vector (reused across all TILE_M rows)
                let x_vec = load(unsafe { x_slice.as_ptr().add(c) });

                for i in 0..TILE_M {
                    let row_idx = r + i;
                    let a_vec = load(unsafe { a_slice.as_ptr().add(row_idx * ncols + c) });
                    accumulators[i] = unsafe { Arch::fmadd(a_vec, x_vec, accumulators[i]) };
                }
                c += lane_count;
            }

            // Reduce accumulators and handle scalar tail for these TILE_M rows
            for i in 0..TILE_M {
                let row_idx = r + i;
                let mut sum = unsafe { Arch::sum_reduce(accumulators[i]) };

                // Scalar tail loop
                for c_tail in simd_len..ncols {
                    sum += a_slice[row_idx * ncols + c_tail] * x_slice[c_tail];
                }
                y[row_idx] += sum;
            }

            r += TILE_M;
        }

        // Cleanup remaining rows (less than TILE_M)
        while r < nrows {
            let mut sum = T::ZERO;
            let mut c = 0;
            if simd_len > 0 {
                let mut acc = unsafe { Arch::zero() };
                while c < simd_len {
                    let x_vec = load(unsafe { x_slice.as_ptr().add(c) });
                    let a_vec = load(unsafe { a_slice.as_ptr().add(r * ncols + c) });
                    acc = unsafe { Arch::fmadd(a_vec, x_vec, acc) };
                    c += lane_count;
                }
                sum = unsafe { Arch::sum_reduce(acc) };
            }
            for c_tail in c..ncols {
                sum += a_slice[r * ncols + c_tail] * x_slice[c_tail];
            }
            y[r] += sum;
            r += 1;
        }

        Ok(())
    }

    #[inline]
    fn dot(
        a: &SimdView<'_, T, Arch, Align>,
        b: &SimdView<'_, T, Arch, Align>,
    ) -> Result<T, SimdError> {
        struct AssertM<const TILE_M: usize>;
        impl<const TILE_M: usize> AssertM<TILE_M> {
            const OK: () = assert!(TILE_M >= 1, "TILE_M must be at least 1");
        }
        let _ = AssertM::<TILE_M>::OK;

        crate::view::check_lengths_equal(a.len(), b.len())?;

        let len = a.len();
        let lane_count = Arch::LANE_COUNT;
        let tile_width = lane_count * TILE_M;
        let tiled_len = (len / tile_width) * tile_width;

        let load = |ptr: *const T| -> Arch::Vector {
            if Align::IS_ALIGNED {
                unsafe { Arch::load_aligned(ptr) }
            } else {
                unsafe { Arch::load_unaligned(ptr) }
            }
        };

        let mut ptr_a = a.as_slice().as_ptr();
        let mut ptr_b = b.as_slice().as_ptr();

        // TILE_M independent accumulators initialized via mul (not zero+fmadd)
        // to avoid an extra dependency on the zero register.
        let mut accumulators: [Arch::Vector; TILE_M] = {
            let mut arr = [unsafe { Arch::zero() }; TILE_M];
            if tiled_len > 0 {
                for i in 0..TILE_M {
                    let va = load(unsafe { ptr_a.add(i * lane_count) });
                    let vb = load(unsafe { ptr_b.add(i * lane_count) });
                    arr[i] = unsafe { Arch::mul(va, vb) };
                }
                unsafe {
                    ptr_a = ptr_a.add(tile_width);
                    ptr_b = ptr_b.add(tile_width);
                }
            }
            arr
        };

        if tiled_len > tile_width {
            let iterations = (tiled_len / tile_width) - 1;
            for _ in 0..iterations {
                for i in 0..TILE_M {
                    let va = load(unsafe { ptr_a.add(i * lane_count) });
                    let vb = load(unsafe { ptr_b.add(i * lane_count) });
                    accumulators[i] = unsafe { Arch::fmadd(va, vb, accumulators[i]) };
                }
                unsafe {
                    ptr_a = ptr_a.add(tile_width);
                    ptr_b = ptr_b.add(tile_width);
                }
            }
        }

        // Horizontal reduce across TILE_M accumulators
        let mut total = T::ZERO;
        if tiled_len > 0 {
            let mut combined = accumulators[0];
            for i in 1..TILE_M {
                combined = unsafe { Arch::add(combined, accumulators[i]) };
            }
            total = unsafe { Arch::sum_reduce(combined) };
        }

        // Scalar tail
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        for i in tiled_len..len {
            total += a_slice[i] * b_slice[i];
        }

        Ok(total)
    }
}
