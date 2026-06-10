//! Monomorphized vector register types for the hermes-simd workspace.
//!
//! Provides compile-time configured type aliases that map to target-optimal registers
//! and explicit aliases for each hardware backend.

#![cfg_attr(not(feature = "std"), no_std)]

pub use hermes_simd_core::scalar::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8, I16, I32, I8};
pub use hermes_simd_core::view::{Mask, Vector};
pub use hermes_simd_intrinsics::{Avx2, Avx512, Neon, Scalar};

// -----------------------------------------------------------------------------
// Preferred Architecture Typestate Selection
// -----------------------------------------------------------------------------

/// The optimal target architecture typestate compiled for the current host CPU target.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx512f"
))]
pub type PreferredArch = hermes_simd_intrinsics::Avx512;

/// The optimal target architecture typestate compiled for the current host CPU target.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512f"),
    target_feature = "avx2"
))]
pub type PreferredArch = hermes_simd_intrinsics::Avx2;

/// The optimal target architecture typestate compiled for the current host CPU target.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_feature = "avx512f"),
    not(target_feature = "avx2")
))]
pub type PreferredArch = hermes_simd_intrinsics::Scalar;

/// The optimal target architecture typestate compiled for the current host CPU target.
#[cfg(target_arch = "aarch64")]
pub type PreferredArch = hermes_simd_intrinsics::Neon;

/// The optimal target architecture typestate compiled for the current host CPU target.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
pub type PreferredArch = hermes_simd_intrinsics::Scalar;

// -----------------------------------------------------------------------------
// Generic Wrappers
// -----------------------------------------------------------------------------

/// Generic f32 SIMD vector register.
pub type VectorF32<A> = Vector<F32, A>;
/// Generic f64 SIMD vector register.
pub type VectorF64<A> = Vector<F64, A>;
/// Generic standard f16 SIMD vector register.
pub type VectorF16<A> = Vector<F16, A>;
/// Generic bfloat16 SIMD vector register.
pub type VectorBf16<A> = Vector<Bf16, A>;
/// Generic bfloat8 SIMD vector register.
pub type VectorBf8<A> = Vector<Bf8, A>;
/// Generic bfloat4 SIMD vector register.
pub type VectorBf4<A> = Vector<Bf4, A>;
/// Generic float8 SIMD vector register.
pub type VectorF8<A> = Vector<F8, A>;
/// Generic float4 SIMD vector register.
pub type VectorF4<A> = Vector<F4, A>;
/// Generic i8 SIMD vector register.
pub type VectorI8<A> = Vector<I8, A>;
/// Generic i16 SIMD vector register.
pub type VectorI16<A> = Vector<I16, A>;
/// Generic i32 SIMD vector register.
pub type VectorI32<A> = Vector<I32, A>;

/// Generic f32 SIMD lane selection mask.
pub type MaskF32<A> = Mask<F32, A>;
/// Generic f64 SIMD lane selection mask.
pub type MaskF64<A> = Mask<F64, A>;
/// Generic standard f16 SIMD lane selection mask.
pub type MaskF16<A> = Mask<F16, A>;
/// Generic bfloat16 SIMD lane selection mask.
pub type MaskBf16<A> = Mask<Bf16, A>;
/// Generic bfloat8 SIMD lane selection mask.
pub type MaskBf8<A> = Mask<Bf8, A>;
/// Generic bfloat4 SIMD lane selection mask.
pub type MaskBf4<A> = Mask<Bf4, A>;
/// Generic float8 SIMD lane selection mask.
pub type MaskF8<A> = Mask<F8, A>;
/// Generic float4 SIMD lane selection mask.
pub type MaskF4<A> = Mask<F4, A>;
/// Generic i8 SIMD lane selection mask.
pub type MaskI8<A> = Mask<I8, A>;
/// Generic i16 SIMD lane selection mask.
pub type MaskI16<A> = Mask<I16, A>;
/// Generic i32 SIMD lane selection mask.
pub type MaskI32<A> = Mask<I32, A>;

// -----------------------------------------------------------------------------
// Preferred SIMD Target Aliases
// -----------------------------------------------------------------------------

/// Optimal f32 SIMD register compiled for the host.
pub type SimdF32 = Vector<F32, PreferredArch>;
/// Optimal f64 SIMD register compiled for the host.
pub type SimdF64 = Vector<F64, PreferredArch>;
/// Optimal f16 SIMD register compiled for the host.
pub type SimdF16 = Vector<F16, PreferredArch>;
/// Optimal bfloat16 SIMD register compiled for the host.
pub type SimdBf16 = Vector<Bf16, PreferredArch>;
/// Optimal bfloat8 SIMD register compiled for the host.
pub type SimdBf8 = Vector<Bf8, PreferredArch>;
/// Optimal bfloat4 SIMD register compiled for the host.
pub type SimdBf4 = Vector<Bf4, PreferredArch>;
/// Optimal float8 SIMD register compiled for the host.
pub type SimdF8 = Vector<F8, PreferredArch>;
/// Optimal float4 SIMD register compiled for the host.
pub type SimdF4 = Vector<F4, PreferredArch>;
/// Optimal i8 SIMD register compiled for the host.
pub type SimdI8 = Vector<I8, PreferredArch>;
/// Optimal i16 SIMD register compiled for the host.
pub type SimdI16 = Vector<I16, PreferredArch>;
/// Optimal i32 SIMD register compiled for the host.
pub type SimdI32 = Vector<I32, PreferredArch>;

/// Optimal f32 SIMD mask register compiled for the host.
pub type SimdMaskF32 = Mask<F32, PreferredArch>;
/// Optimal f64 SIMD mask register compiled for the host.
pub type SimdMaskF64 = Mask<F64, PreferredArch>;
/// Optimal f16 SIMD mask register compiled for the host.
pub type SimdMaskF16 = Mask<F16, PreferredArch>;
/// Optimal bfloat16 SIMD mask register compiled for the host.
pub type SimdMaskBf16 = Mask<Bf16, PreferredArch>;
/// Optimal bfloat8 SIMD mask register compiled for the host.
pub type SimdMaskBf8 = Mask<Bf8, PreferredArch>;
/// Optimal bfloat4 SIMD mask register compiled for the host.
pub type SimdMaskBf4 = Mask<Bf4, PreferredArch>;
/// Optimal float8 SIMD mask register compiled for the host.
pub type SimdMaskF8 = Mask<F8, PreferredArch>;
/// Optimal float4 SIMD mask register compiled for the host.
pub type SimdMaskF4 = Mask<F4, PreferredArch>;
/// Optimal i8 SIMD mask register compiled for the host.
pub type SimdMaskI8 = Mask<I8, PreferredArch>;
/// Optimal i16 SIMD mask register compiled for the host.
pub type SimdMaskI16 = Mask<I16, PreferredArch>;
/// Optimal i32 SIMD mask register compiled for the host.
pub type SimdMaskI32 = Mask<I32, PreferredArch>;

// -----------------------------------------------------------------------------
// Concrete Target-Bound Register Aliases
// -----------------------------------------------------------------------------

/// Concrete 1-element scalar emulation f32 vector register.
pub type ScalarF32 = Vector<F32, Scalar>;
/// Concrete 1-element scalar emulation f64 vector register.
pub type ScalarF64 = Vector<F64, Scalar>;
/// Concrete 1-element scalar emulation f16 vector register.
pub type ScalarF16 = Vector<F16, Scalar>;
/// Concrete 1-element scalar emulation bfloat16 vector register.
pub type ScalarBf16 = Vector<Bf16, Scalar>;
/// Concrete 1-element scalar emulation bfloat8 vector register.
pub type ScalarBf8 = Vector<Bf8, Scalar>;
/// Concrete 1-element scalar emulation bfloat4 vector register.
pub type ScalarBf4 = Vector<Bf4, Scalar>;
/// Concrete 1-element scalar emulation float8 vector register.
pub type ScalarF8 = Vector<F8, Scalar>;
/// Concrete 1-element scalar emulation float4 vector register.
pub type ScalarF4 = Vector<F4, Scalar>;
/// Concrete 1-element scalar emulation i8 vector register.
pub type ScalarI8 = Vector<I8, Scalar>;
/// Concrete 1-element scalar emulation i16 vector register.
pub type ScalarI16 = Vector<I16, Scalar>;
/// Concrete 1-element scalar emulation i32 vector register.
pub type ScalarI32 = Vector<I32, Scalar>;

/// Concrete 1-element scalar emulation f32 mask register.
pub type ScalarMaskF32 = Mask<F32, Scalar>;
/// Concrete 1-element scalar emulation f64 mask register.
pub type ScalarMaskF64 = Mask<F64, Scalar>;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86_aliases::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_aliases {
    use super::*;

    // AVX2 Vector types
    /// Concrete AVX2 f32 vector register (8 lanes).
    pub type Avx2F32 = Vector<F32, Avx2>;
    /// Concrete AVX2 f64 vector register (4 lanes).
    pub type Avx2F64 = Vector<F64, Avx2>;
    /// Concrete AVX2 standard f16 vector register (16 lanes).
    pub type Avx2F16 = Vector<F16, Avx2>;
    /// Concrete AVX2 bfloat16 vector register (16 lanes).
    pub type Avx2Bf16 = Vector<Bf16, Avx2>;
    /// Concrete AVX2 bfloat8 vector register (32 lanes).
    pub type Avx2Bf8 = Vector<Bf8, Avx2>;
    /// Concrete AVX2 bfloat4 vector register (32 lanes).
    pub type Avx2Bf4 = Vector<Bf4, Avx2>;
    /// Concrete AVX2 float8 vector register (32 lanes).
    pub type Avx2F8 = Vector<F8, Avx2>;
    /// Concrete AVX2 float4 vector register (32 lanes).
    pub type Avx2F4 = Vector<F4, Avx2>;
    /// Concrete AVX2 i8 vector register (32 lanes).
    pub type Avx2I8 = Vector<I8, Avx2>;
    /// Concrete AVX2 i16 vector register (16 lanes).
    pub type Avx2I16 = Vector<I16, Avx2>;
    /// Concrete AVX2 i32 vector register (8 lanes).
    pub type Avx2I32 = Vector<I32, Avx2>;

    // AVX2 Mask types
    /// Concrete AVX2 f32 mask register (8 lanes).
    pub type Avx2MaskF32 = Mask<F32, Avx2>;
    /// Concrete AVX2 f64 mask register (4 lanes).
    pub type Avx2MaskF64 = Mask<F64, Avx2>;
    /// Concrete AVX2 f16 mask register (16 lanes).
    pub type Avx2MaskF16 = Mask<F16, Avx2>;
    /// Concrete AVX2 bfloat16 mask register (16 lanes).
    pub type Avx2MaskBf16 = Mask<Bf16, Avx2>;

    // AVX-512 Vector types
    /// Concrete AVX-512 f32 vector register (16 lanes).
    pub type Avx512F32 = Vector<F32, Avx512>;
    /// Concrete AVX-512 f64 vector register (8 lanes).
    pub type Avx512F64 = Vector<F64, Avx512>;
    /// Concrete AVX-512 standard f16 vector register (32 lanes).
    pub type Avx512F16 = Vector<F16, Avx512>;
    /// Concrete AVX-512 bfloat16 vector register (32 lanes).
    pub type Avx512Bf16 = Vector<Bf16, Avx512>;
    /// Concrete AVX-512 bfloat8 vector register (64 lanes).
    pub type Avx512Bf8 = Vector<Bf8, Avx512>;
    /// Concrete AVX-512 bfloat4 vector register (64 lanes).
    pub type Avx512Bf4 = Vector<Bf4, Avx512>;
    /// Concrete AVX-512 float8 vector register (64 lanes).
    pub type Avx512F8 = Vector<F8, Avx512>;
    /// Concrete AVX-512 float4 vector register (64 lanes).
    pub type Avx512F4 = Vector<F4, Avx512>;
    /// Concrete AVX-512 i8 vector register (64 lanes).
    pub type Avx512I8 = Vector<I8, Avx512>;
    /// Concrete AVX-512 i16 vector register (32 lanes).
    pub type Avx512I16 = Vector<I16, Avx512>;
    /// Concrete AVX-512 i32 vector register (16 lanes).
    pub type Avx512I32 = Vector<I32, Avx512>;

    // AVX-512 Mask types
    /// Concrete AVX-512 f32 mask register (16 lanes).
    pub type Avx512MaskF32 = Mask<F32, Avx512>;
    /// Concrete AVX-512 f64 mask register (8 lanes).
    pub type Avx512MaskF64 = Mask<F64, Avx512>;
    /// Concrete AVX-512 f16 mask register (32 lanes).
    pub type Avx512MaskF16 = Mask<F16, Avx512>;
    /// Concrete AVX-512 bfloat16 mask register (32 lanes).
    pub type Avx512MaskBf16 = Mask<Bf16, Avx512>;
}

#[cfg(target_arch = "aarch64")]
pub use aarch64_aliases::*;

#[cfg(target_arch = "aarch64")]
mod aarch64_aliases {
    use super::*;

    // NEON Vector types
    /// Concrete NEON f32 vector register (4 lanes).
    pub type NeonF32 = Vector<F32, Neon>;
    /// Concrete NEON f64 vector register (2 lanes).
    pub type NeonF64 = Vector<F64, Neon>;
    /// Concrete NEON standard f16 vector register (8 lanes).
    pub type NeonF16 = Vector<F16, Neon>;
    /// Concrete NEON bfloat16 vector register (8 lanes).
    pub type NeonBf16 = Vector<Bf16, Neon>;
    /// Concrete NEON bfloat8 vector register (16 lanes).
    pub type NeonBf8 = Vector<Bf8, Neon>;
    /// Concrete NEON bfloat4 vector register (16 lanes).
    pub type NeonBf4 = Vector<Bf4, Neon>;
    /// Concrete NEON float8 vector register (16 lanes).
    pub type NeonF8 = Vector<F8, Neon>;
    /// Concrete NEON float4 vector register (16 lanes).
    pub type NeonF4 = Vector<F4, Neon>;
    /// Concrete NEON i8 vector register (16 lanes).
    pub type NeonI8 = Vector<I8, Neon>;
    /// Concrete NEON i16 vector register (8 lanes).
    pub type NeonI16 = Vector<I16, Neon>;
    /// Concrete NEON i32 vector register (4 lanes).
    pub type NeonI32 = Vector<I32, Neon>;

    // NEON Mask types
    /// Concrete NEON f32 mask register (4 lanes).
    pub type NeonMaskF32 = Mask<F32, Neon>;
    /// Concrete NEON f64 mask register (2 lanes).
    pub type NeonMaskF64 = Mask<F64, Neon>;
    /// Concrete NEON f16 mask register (8 lanes).
    pub type NeonMaskF16 = Mask<F16, Neon>;
    /// Concrete NEON bfloat16 mask register (8 lanes).
    pub type NeonMaskBf16 = Mask<Bf16, Neon>;
}
