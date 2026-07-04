use super::{tile_loop_generic, validate_gemm_sizes, TiledGemm};
#[cfg(target_arch = "x86_64")]
use crate::cpu::{AmxSupport, Avx512Support};
use eunomia::{I32, I8};
use hermes_simd_core::view::{SimdError, TileMatrixMultiply};
use hermes_simd_intrinsics::Scalar;
#[cfg(target_arch = "x86_64")]
use hermes_simd_intrinsics::{AmxInt8, Avx512, AvxVnni};

/// Scalar cleanup for the rows/columns/K-depth the 16×16×64 tile loop leaves
/// uncovered (`m % 16` rows, `n % 16` columns, `k % 64` tail on covered tiles).
///
/// One implementation shared by every dispatch arm and both element newtypes so
/// the wrapping accumulation semantics stay identical across backends.
fn gemm_i8_remainder(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    a_stride: usize,
    b: &[i8],
    b_stride: usize,
    c: &mut [i32],
    c_stride: usize,
) {
    let tile_m_bound = (m / 16) * 16;
    let tile_n_bound = (n / 16) * 16;
    let tile_k_bound = (k / 64) * 64;
    for r in 0..m {
        for col in 0..n {
            if r >= tile_m_bound || col >= tile_n_bound {
                let mut sum = 0i32;
                for kk in 0..k {
                    sum = sum.wrapping_add(
                        (a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32),
                    );
                }
                c[r * c_stride + col] += sum;
            } else if tile_k_bound < k {
                let mut sum = 0i32;
                for kk in tile_k_bound..k {
                    sum = sum.wrapping_add(
                        (a[r * a_stride + kk] as i32) * (b[kk * b_stride + col] as i32),
                    );
                }
                c[r * c_stride + col] += sum;
            }
        }
    }
}

/// Dispatched int8 GEMM body shared by the `i8` and `I8` trait impls (`I8`/`I32`
/// are `#[repr(transparent)]` over `i8`/`i32`, so the newtype impl delegates via
/// layout-preserving slice casts).
///
/// Backend ladder: AMX → AVX-512 VNNI → AVX-VNNI (256-bit) → scalar, each tier
/// entered only after its runtime probe passes.
///
/// # Safety
/// Caller must have validated operand extents via `validate_gemm_sizes`.
unsafe fn gemm_i8_dispatched(
    m: usize,
    n: usize,
    k: usize,
    a: &[i8],
    a_stride: usize,
    b: &[i8],
    b_stride: usize,
    c: &mut [i32],
    c_stride: usize,
) -> Result<(), SimdError> {
    #[cfg(target_arch = "x86_64")]
    {
        let decision = crate::dispatcher::AdaptiveDispatcher::select_backend(
            m,
            n,
            k,
            a.as_ptr(),
            a.len(),
            b.as_ptr(),
            b.len(),
        );

        match decision {
            crate::dispatcher::DispatchDecision::Amx => {
                <AmxInt8 as hermes_simd_intrinsics::x86_64::amx::AmxGemm<i8, i8, i32>>::amx_gemm(
                    m,
                    n,
                    k,
                    a.as_ptr(),
                    a_stride,
                    b.as_ptr(),
                    b_stride,
                    c.as_mut_ptr(),
                    c_stride,
                );
                return Ok(());
            }
            crate::dispatcher::DispatchDecision::Avx512 => {
                tile_loop_generic::<i8, i8, i32, Avx512, 16, 16, 64>(
                    m,
                    n,
                    k,
                    a.as_ptr(),
                    a_stride,
                    b.as_ptr(),
                    b_stride,
                    c.as_mut_ptr(),
                    c_stride,
                );
                gemm_i8_remainder(m, n, k, a, a_stride, b, b_stride, c, c_stride);
                return Ok(());
            }
            crate::dispatcher::DispatchDecision::AvxVnni => {
                tile_loop_generic::<i8, i8, i32, AvxVnni, 16, 16, 64>(
                    m,
                    n,
                    k,
                    a.as_ptr(),
                    a_stride,
                    b.as_ptr(),
                    b_stride,
                    c.as_mut_ptr(),
                    c_stride,
                );
                gemm_i8_remainder(m, n, k, a, a_stride, b, b_stride, c, c_stride);
                return Ok(());
            }
            crate::dispatcher::DispatchDecision::Scalar => {}
        }
    }

    tile_loop_generic::<i8, i8, i32, Scalar, 16, 16, 64>(
        m,
        n,
        k,
        a.as_ptr(),
        a_stride,
        b.as_ptr(),
        b_stride,
        c.as_mut_ptr(),
        c_stride,
    );
    gemm_i8_remainder(m, n, k, a, a_stride, b, b_stride, c, c_stride);
    Ok(())
}

impl TiledGemm<i8, i8, i32> for (i8, i8, i32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut i32,
        c_stride: usize,
        a: *const i8,
        a_stride: usize,
        b: *const i8,
        b_stride: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            if <i8 as AmxSupport>::has_amx() && hermes_simd_intrinsics::AmxSession::is_active() {
                return <AmxInt8 as TileMatrixMultiply<
                    i8,
                    i8,
                    i32,
                    AmxInt8,
                    AmxInt8,
                    16,
                    16,
                    64,
                >>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if <i8 as Avx512Support>::has_avx512() {
                return <Avx512 as TileMatrixMultiply<i8, i8, i32, Avx512, Avx512, 16, 16, 64>>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
            if crate::cpu::has_avx_vnni() {
                return <AvxVnni as TileMatrixMultiply<
                    i8,
                    i8,
                    i32,
                    AvxVnni,
                    AvxVnni,
                    16,
                    16,
                    64,
                >>::tile_matmul(c, c_stride, a, a_stride, b, b_stride);
            }
        }
        <Scalar as TileMatrixMultiply<i8, i8, i32, Scalar, Scalar, 16, 16, 64>>::tile_matmul(
            c, c_stride, a, a_stride, b, b_stride,
        );
    }

    #[inline]
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[i8],
        a_stride: usize,
        b: &[i8],
        b_stride: usize,
        c: &mut [i32],
        c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(
            a.len(),
            b.len(),
            c.len(),
            m,
            n,
            k,
            a_stride,
            b_stride,
            c_stride,
        )?;
        gemm_i8_dispatched(m, n, k, a, a_stride, b, b_stride, c, c_stride)
    }
}

impl TiledGemm<I8, I8, I32> for (I8, I8, I32) {
    #[inline]
    unsafe fn dispatch_tile_matmul(
        c: *mut I32,
        c_stride: usize,
        a: *const I8,
        a_stride: usize,
        b: *const I8,
        b_stride: usize,
    ) {
        // SAFETY: `I8`/`I32` are `#[repr(transparent)]` over `i8`/`i32`, so the
        // pointer casts preserve layout and the `i8` dispatcher's tile contract.
        <(i8, i8, i32) as TiledGemm<i8, i8, i32>>::dispatch_tile_matmul(
            c as *mut i32,
            c_stride,
            a as *const i8,
            a_stride,
            b as *const i8,
            b_stride,
        );
    }

    #[inline]
    unsafe fn gemm(
        m: usize,
        n: usize,
        k: usize,
        a: &[I8],
        a_stride: usize,
        b: &[I8],
        b_stride: usize,
        c: &mut [I32],
        c_stride: usize,
    ) -> Result<(), SimdError> {
        validate_gemm_sizes(
            a.len(),
            b.len(),
            c.len(),
            m,
            n,
            k,
            a_stride,
            b_stride,
            c_stride,
        )?;
        // SAFETY: `I8`/`I32` are `#[repr(transparent)]` over `i8`/`i32`; the
        // reborrowed slices alias the same memory with identical layout and
        // length, and `c` is exclusively borrowed for the call's duration.
        let a_raw = core::slice::from_raw_parts(a.as_ptr() as *const i8, a.len());
        let b_raw = core::slice::from_raw_parts(b.as_ptr() as *const i8, b.len());
        let c_raw = core::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut i32, c.len());
        gemm_i8_dispatched(m, n, k, a_raw, a_stride, b_raw, b_stride, c_raw, c_stride)
    }
}
