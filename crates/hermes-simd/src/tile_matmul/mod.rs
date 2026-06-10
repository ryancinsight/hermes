//! Register-blocked tile matrix multiplication module.

use hermes_simd_core::view::SimdError;

/// Brain/AMX tile GEMM implementation for BF16/F32.
pub mod bf16;
/// Brain/AMX tile GEMM implementation for INT8/INT32.
pub mod int8;
/// Low-precision integer unpacking utility.
pub mod unpack;

pub use unpack::*;

#[inline(never)]
pub(crate) fn validate_gemm_sizes(
    a_len: usize,
    b_len: usize,
    c_len: usize,
    m: usize,
    _n: usize,
    k: usize,
    a_stride: usize,
    b_stride: usize,
    c_stride: usize,
) -> Result<(), SimdError> {
    if a_len < m * a_stride || b_len < k * b_stride || c_len < m * c_stride {
        return Err(SimdError::LengthMismatch);
    }
    Ok(())
}

#[inline(always)]
pub(crate) unsafe fn tile_loop_generic<
    TA,
    TB,
    TC,
    Arch,
    const M: usize,
    const N: usize,
    const K: usize,
>(
    m: usize,
    n: usize,
    k: usize,
    a: *const TA,
    a_stride: usize,
    b: *const TB,
    b_stride: usize,
    c: *mut TC,
    c_stride: usize,
) where
    Arch: hermes_simd_core::view::TileMatrixMultiply<TA, TB, TC, Arch, Arch, M, N, K>,
{
    let mut i = 0;
    while i + M <= m {
        let mut j = 0;
        while j + N <= n {
            let mut kk = 0;
            while kk + K <= k {
                Arch::tile_matmul(
                    c.add(i * c_stride + j),
                    c_stride,
                    a.add(i * a_stride + kk),
                    a_stride,
                    b.add(kk * b_stride + j),
                    b_stride,
                );
                kk += K;
            }
            j += N;
        }
        i += M;
    }
}

/// Trait for performing register-tiled GEMM and dynamic tile dispatching.
pub trait TiledGemm<TA, TB, TC> {
    /// Perform matrix multiplication `c += a * b` using register-blocked/tiled SIMD.
    ///
    /// # Safety
    /// - Pointers must be valid and slices must have matching dimensions.
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[TA],
        a_stride: usize,
        b: &[TB],
        b_stride: usize,
        c: &mut [TC],
        c_stride: usize,
    ) -> Result<(), SimdError>;

    /// Dynamically dispatches tile matrix multiplication for a single tile of shape MxNxK.
    ///
    /// # Safety
    /// - Pointers must be valid and aligned as per the chosen backend requirements.
    unsafe fn dispatch_tile_matmul(
        c: *mut TC,
        c_stride: usize,
        a: *const TA,
        a_stride: usize,
        b: *const TB,
        b_stride: usize,
    );
}

/// Perform matrix multiplication `c += a * b` using register-blocked/tiled SIMD.
///
/// Automatically dispatches to the most performant backend available (e.g. Intel AMX, AVX-512, or Scalar).
///
/// # Safety
/// - Pointers must be valid and slices must have matching dimensions.
#[inline]
pub unsafe fn gemm<TA, TB, TC>(
    m: usize,
    n: usize,
    k: usize,
    a: &[TA],
    a_stride: usize,
    b: &[TB],
    b_stride: usize,
    c: &mut [TC],
    c_stride: usize,
) -> Result<(), SimdError>
where
    (TA, TB, TC): TiledGemm<TA, TB, TC>,
{
    <(TA, TB, TC) as TiledGemm<TA, TB, TC>>::gemm(m, n, k, a, a_stride, b, b_stride, c, c_stride)
}

/// Dynamically dispatches tile matrix multiplication for a single tile of shape MxNxK.
///
/// # Safety
/// - Pointers must be valid and aligned as per the chosen backend requirements.
#[inline]
pub unsafe fn dispatch_tile_matmul<TA, TB, TC>(
    c: *mut TC,
    c_stride: usize,
    a: *const TA,
    a_stride: usize,
    b: *const TB,
    b_stride: usize,
) where
    (TA, TB, TC): TiledGemm<TA, TB, TC>,
{
    <(TA, TB, TC) as TiledGemm<TA, TB, TC>>::dispatch_tile_matmul(
        c, c_stride, a, a_stride, b, b_stride,
    )
}
