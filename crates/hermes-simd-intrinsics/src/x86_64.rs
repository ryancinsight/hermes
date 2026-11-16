//! `x86_64` hardware specialized SIMD kernels module.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn sfence() {
    // SAFETY: `_mm_sfence` (SSE) is baseline on x86_64 and part of the x86
    // intrinsic surface used by the x86 backends in this module.
    unsafe { core::arch::x86_64::_mm_sfence() };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[cfg_attr(not(debug_assertions), allow(unused_variables))]
pub(crate) fn debug_assert_mask_within_valid_lanes<T, K>(valid_lanes: usize, mask: K::Mask)
where
    T: hermes_simd_core::scalar::Scalar,
    K: hermes_simd_core::kernel::BackendKernel<T>,
{
    hermes_simd_core::kernel::backend::debug_assert_mask_within_valid_lanes::<T, K>(
        valid_lanes,
        mask,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn compress_selected_lanes<T: Copy + Default, const LANES: usize>(
    src: [T; LANES],
    mask_bits: u32,
) -> [T; LANES] {
    let mut out = [T::default(); LANES];
    let mut k = 0usize;
    let mut i = 0usize;
    while i < LANES {
        if ((mask_bits >> i) & 1) != 0 {
            out[k] = src[i];
            k += 1;
        }
        i += 1;
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn expand_selected_lanes<T: Copy, const LANES: usize>(
    src: [T; LANES],
    mut out: [T; LANES],
    mask_bits: u32,
) -> [T; LANES] {
    let mut k = 0usize;
    let mut i = 0usize;
    while i < LANES {
        if ((mask_bits >> i) & 1) != 0 {
            out[i] = src[k];
            k += 1;
        }
        i += 1;
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_f16;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_f32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_f64;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512_f16;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512_f32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512_f64;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod amx;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512_tiling;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx_vnni_tiling;

// Avx2 emulated kernels for integers and Eunomia wrappers.
crate::impl_emulated_kernel!(
    crate::Avx2,
    i8,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    i16,
    16,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    i32,
    8,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::Bf16,
    16,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::I8,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::I16,
    16,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::I32,
    8,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::F32,
    8,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::F64,
    4,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::Bf8,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::Bf4,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::F8,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx2,
    eunomia::F4,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);

// Avx512 emulated kernels for integers and Eunomia wrappers.
crate::impl_emulated_kernel!(
    crate::Avx512,
    i8,
    64,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    i16,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    i32,
    16,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::Bf16,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::I8,
    64,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::I16,
    32,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::I32,
    16,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::F32,
    16,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::F64,
    8,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::Bf8,
    64,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::Bf4,
    64,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::F8,
    64,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
crate::impl_emulated_kernel!(
    crate::Avx512,
    eunomia::F4,
    64,
    cfg(any(target_arch = "x86", target_arch = "x86_64"))
);
