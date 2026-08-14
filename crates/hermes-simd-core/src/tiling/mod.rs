//! Register-blocked, cache-aware tiling for dot products, GEMV, and GEMM.
//!
//! # Motivation
//!
//! A standard `dot`/GEMV/GEMM loop has a loop-carried FMA dependency chain that
//! limits throughput to one FMA per FMA-latency window. Unrolling into `TILE_M`
//! independent accumulator registers breaks this chain, saturating the FMA issue
//! ports on modern CPUs:
//!
//! - **AVX-512** (32 ZMM regs): larger tiles (e.g. `TILE_M = 8`) fit the register
//!   file; the GEMM dispatcher selects `<3,4>` for f64.
//! - **AVX2** (16 YMM regs): the GEMM dispatcher selects the register-resident
//!   `<3,3>` f64 tile (see [`gemm`] Theorem 3).
//! - **Scalar**: `TILE_M = TILE_N = 1` is the standard loop; const
//!   monomorphization eliminates all tiling overhead at zero cost.
//!
//! # Structure (separation of concerns)
//!
//! This module owns the [`TilingStrategy`] trait surface and the zero-sized
//! [`TilingPolicy`] tile-shape marker; the per-operation kernels live in vertical
//! leaf modules, each carrying its own correctness/throughput theorems:
//!
//! - [`dot`] — register-blocked dot product (dependency-chain throughput theorem).
//! - [`gemv`] — matrix–vector product (operand-reuse theorem).
//! - [`gemm`] — matrix multiplication (correctness, packing-invariance,
//!   register-residency, and cache cost-model theorems).
//!
//! The [`TilingPolicy`] trait methods are thin monomorphizing delegators to those
//! leaf kernels — one authoritative implementation per operation (SSOT/DRY).

use crate::{
    align::Alignment,
    arch::SimdArch,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
};
use core::marker::PhantomData;

mod dims;
pub mod dot;
pub mod gemm;
pub mod gemv;
pub mod gemv_transpose;

/// Trait representing a monomorphized register-blocking/tiling strategy.
///
/// Implemented as a blanket over [`TilingPolicy<TILE_M, TILE_N>`]. The const
/// generic parameters encode the tile shape so the compiler emits loop-unrolled,
/// register-blocked kernels with zero runtime overhead.
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
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when the operand spans or output
    /// length do not satisfy the supplied matrix dimensions.
    fn gemm(
        a: &SimdView<'_, T, Arch, Align>,
        b: &SimdView<'_, T, Arch, Align>,
        c: &mut [T],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), SimdError>;

    /// Perform tiled matrix-vector multiplication `y += A * x` using this strategy.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when the operand spans or output
    /// length do not satisfy the supplied matrix dimensions.
    fn gemv(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError>;

    /// Perform tiled transposed matrix-vector multiplication `y += Aᵀ * x`
    /// (`A` row-major `nrows × ncols`, `x` length `nrows`, `y` length `ncols`).
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when the operand spans or output
    /// length do not satisfy the supplied matrix dimensions.
    fn gemv_transpose(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError>;

    /// Perform tiled matrix-vector multiplication `y += A * x` over a row-major
    /// **sub-matrix**: `nrows × ncols` with row stride `lda ≥ ncols`
    /// (`lda = ncols` is the packed [`Self::gemv`]).
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when `lda` is too small or the
    /// operand spans or output length do not satisfy the supplied dimensions.
    fn gemv_strided(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), SimdError>;

    /// Perform tiled transposed matrix-vector multiplication `y += Aᵀ * x` over a
    /// row-major **sub-matrix**: `nrows × ncols` with row stride `lda ≥ ncols`
    /// (`lda = ncols` is the packed [`Self::gemv_transpose`]).
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when `lda` is too small or the
    /// operand spans or output length do not satisfy the supplied dimensions.
    fn gemv_transpose_strided(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), SimdError>;

    /// Perform tiled dot product computation using this strategy.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when the operand lengths differ.
    fn dot(
        a: &SimdView<'_, T, Arch, Align>,
        b: &SimdView<'_, T, Arch, Align>,
    ) -> Result<T, SimdError>;
}

/// Compute the dot product of two slices using `TILE_M` independent vector accumulators.
///
/// The inner loop processes `TILE_M * LANE_COUNT` elements per iteration, holding
/// `TILE_M` accumulator registers simultaneously to saturate FMA throughput.
///
/// # Errors
/// Returns [`SimdError::LengthMismatch`] when the operand lengths differ.
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
    /// Named `SCALAR_DEGENERATE` (`SCREAMING_SNAKE_CASE`) to satisfy Rust naming lints.
    pub const SCALAR_DEGENERATE: TilingPolicy<1, 1> = TilingPolicy;

    /// Verify that the tile shape is valid at compile time.
    ///
    /// # Panics
    ///
    /// Panics when either tile dimension is zero.
    pub const fn validate(&self) {
        assert!(TILE_M >= 1, "TILE_M must be >= 1");
        assert!(TILE_N >= 1, "TILE_N must be >= 1");
    }
}

/// Zero-sized type used to force `TilingPolicy` into a generic bound without runtime cost.
pub struct _TileMarker<const M: usize, const N: usize>(PhantomData<TilingPolicy<M, N>>);

/// Compute a register-blocked tiled GEMV: `y += A * x`.
///
/// Processes `TILE_M` rows of `A` simultaneously to reuse loaded elements of `x` across those rows.
/// Accepts `SimdView`-typed operands to enforce alignment typestates at the call boundary.
///
/// # Errors
/// Returns [`SimdError::LengthMismatch`] when the operand spans or output
/// length do not satisfy the supplied matrix dimensions.
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

/// Compute a register-blocked tiled GEMM: `c += A * B`.
///
/// Multiplies matrix `a` (dimensions `m * k`) and matrix `b` (dimensions `k * n`),
/// accumulating the result into `c` (dimensions `m * n`).
///
/// # Errors
/// Returns [`SimdError::LengthMismatch`] when the operand spans or output
/// length do not satisfy the supplied matrix dimensions.
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
        gemm::gemm_impl::<T, Arch, Align, TILE_M, TILE_N>(a, b, c, m, n, k)
    }

    #[inline]
    fn gemv(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError> {
        gemv::gemv_impl::<T, Arch, Align, TILE_M>(a, x, y, nrows, ncols)
    }

    #[inline]
    fn gemv_transpose(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<(), SimdError> {
        // `TILE_N` blocks the output (`y`) lane-chunks for the transpose, mirroring
        // how `TILE_N` blocks the `B`/`c` columns in GEMM.
        gemv_transpose::gemv_transpose_impl::<T, Arch, Align, TILE_N>(a, x, y, nrows, ncols)
    }

    #[inline]
    fn gemv_strided(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), SimdError> {
        gemv::gemv_strided_impl::<T, Arch, Align, TILE_M>(a, x, y, nrows, ncols, lda)
    }

    #[inline]
    fn gemv_transpose_strided(
        a: &SimdView<'_, T, Arch, Align>,
        x: &SimdView<'_, T, Arch, Align>,
        y: &mut [T],
        nrows: usize,
        ncols: usize,
        lda: usize,
    ) -> Result<(), SimdError> {
        gemv_transpose::gemv_transpose_strided_impl::<T, Arch, Align, TILE_N>(
            a, x, y, nrows, ncols, lda,
        )
    }

    #[inline]
    fn dot(
        a: &SimdView<'_, T, Arch, Align>,
        b: &SimdView<'_, T, Arch, Align>,
    ) -> Result<T, SimdError> {
        dot::dot_impl::<T, Arch, Align, TILE_M>(a, b)
    }
}
