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

// Neon emulated kernels for half::bf16, i8, i16, i32
crate::impl_emulated_kernel!(crate::Neon, half::bf16, 8, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, i8, 16, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, i16, 8, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(crate::Neon, i32, 4, cfg(target_arch = "aarch64"));
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::Bf16,
    8,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::I8,
    16,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::I16,
    8,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::I32,
    4,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::F16,
    8,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::F32,
    4,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::F64,
    2,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::Bf8,
    16,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::Bf4,
    16,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::F8,
    16,
    cfg(target_arch = "aarch64")
);
crate::impl_emulated_kernel!(
    crate::Neon,
    hermes_numeric::F4,
    16,
    cfg(target_arch = "aarch64")
);
