//! aarch64 (NEON / SVE) hardware-specialized SIMD kernels module.

#[cfg(target_arch = "aarch64")]
pub mod neon_f16;

#[cfg(target_arch = "aarch64")]
pub mod neon_f32;

#[cfg(target_arch = "aarch64")]
pub mod neon_f64;

/// SVE-shaped emulated backend, always compiled so downstream crates can
/// reference [`SveArch`](sve::SveArch) without a target-arch guard. Native SVE
/// intrinsics remain a separate backend item pending stable Rust support.
pub mod sve;

// Neon emulated kernels for integers and Eunomia wrappers.
crate::impl_emulated_kernel!(crate::Neon, i8, 16, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, i16, 8, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, i32, 4, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::Bf16, 8, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::I8, 16, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::I16, 8, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::I32, 4, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::F32, 4, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::F64, 2, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::Bf8, 16, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::Bf4, 16, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::F8, 16, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, eunomia::F4, 16, cfg(target_arch = "aarch64"));
