//! x86_64 hardware specialized SIMD kernels module.

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
