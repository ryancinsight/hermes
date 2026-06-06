//! Fallback scalar SIMD kernels module.

pub mod f16;
pub mod f32;
pub mod f64;
pub mod tiling;

crate::impl_emulated_kernel!(crate::Scalar, half::bf16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, i8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, i16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, i32, 4, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::Bf16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::I8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::I16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::I32, 4, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::F16, 8, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::F32, 4, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::F64, 2, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::Bf8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::Bf4, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::F8, 16, cfg(all()));
crate::impl_emulated_kernel!(crate::Scalar, hermes_numeric::F4, 16, cfg(all()));
