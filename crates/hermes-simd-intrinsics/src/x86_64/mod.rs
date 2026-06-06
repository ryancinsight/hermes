//! x86_64 hardware specialized SIMD kernels module.

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

// Avx2 emulated kernels for half::bf16, i8, i16, i32 and newtypes
crate::impl_emulated_kernel!(crate::Avx2, half::bf16, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, i8, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, i16, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, i32, 8, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::Bf16, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::I8, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::I16, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::I32, 8, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::F16, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::F32, 8, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::F64, 4, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::Bf8, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::Bf4, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::F8, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx2, hermes_numeric::F4, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));

// Avx512 emulated kernels for half::bf16, i8, i16, i32 and newtypes
crate::impl_emulated_kernel!(crate::Avx512, half::bf16, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, i8, 64, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, i16, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, i32, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::Bf16, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::I8, 64, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::I16, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::I32, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::F16, 32, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::F32, 16, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::F64, 8, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::Bf8, 64, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::Bf4, 64, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::F8, 64, cfg(any(target_arch = "x86", target_arch = "x86_64")));
crate::impl_emulated_kernel!(crate::Avx512, hermes_numeric::F4, 64, cfg(any(target_arch = "x86", target_arch = "x86_64")));
