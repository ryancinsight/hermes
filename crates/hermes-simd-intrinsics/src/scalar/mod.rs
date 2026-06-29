//! Fallback scalar SIMD kernels module.

pub mod f16;
pub mod f32;
pub mod f64;
pub mod tiling;

crate::impl_emulated_kernel!(crate::Scalar, half::bf16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, i8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, i16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, i32, 4, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::Bf16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::I8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::I16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::I32, 4, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::F16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::F32, 4, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::F64, 2, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::Bf8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::Bf4, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::F8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, eunomia::F4, 16, cfg(all()));
