//! Hardware intrinsics and backend-specific implementations of SIMD kernels.
//!
//! This crate provides concrete [`SimdKernel`](hermes_simd_core::kernel::SimdKernel)
//! implementations for every
//! supported architecture marker:
//!
//! | Marker          | ISA          | f32 lanes | f64 lanes |
//! |-----------------|--------------|-----------|-----------|
//! | [`Scalar`]      | scalar loop  | 4         | 2         |
//! | [`Avx2`]        | x86 AVX2     | 8         | 4         |
//! | [`Avx512`]      | x86 AVX-512F | 16        | 8         |
//! | [`Neon`]        | AArch64 NEON | 4         | 2         |
//! | [`SveArch`]    | AArch64 SVE shape, emulated | 16 | 8 |
//!
//! Optional crate-feature backends:
//! - `wide` — wraps the [`wide`](https://docs.rs/wide) crate.
//! - `portable-simd` — wraps nightly `std::simd`.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![allow(
    clippy::needless_range_loop,
    clippy::missing_safety_doc,
    clippy::new_without_default,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::missing_const_for_thread_local
)]
extern crate alloc;

use hermes_simd_core::arch::SimdArch;

/// Implements `SimdKernel<$t>` for `$arch` as a lane-emulated `[T; N]` backend.
///
/// Used for the `Scalar` marker and for `(type, arch)` pairs without native
/// register support; each method is a per-lane loop the optimizer is free to
/// auto-vectorize.
#[macro_export]
macro_rules! impl_emulated_kernel {
    ($arch:ty, $t:ty, $lanes:expr, $cfg:meta) => {
        #[$cfg]
        impl hermes_simd_core::kernel::SimdKernel<$t> for $arch {
            type Vector = [$t; $lanes];
            type Mask = [bool; $lanes];
            type IndexVector = [i32; $lanes];
            const LANE_COUNT: usize = $lanes;
            const UNROLL_FACTOR: usize = 4;

            #[inline(always)]
            unsafe fn load_aligned(ptr: *const $t) -> Self::Vector {
                let mut v = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
                core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), $lanes);
                v
            }

            #[inline(always)]
            unsafe fn load_unaligned(ptr: *const $t) -> Self::Vector {
                let mut v = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
                core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), $lanes);
                v
            }

            #[inline(always)]
            unsafe fn store_aligned(ptr: *mut $t, val: Self::Vector) {
                core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, $lanes);
            }

            #[inline(always)]
            unsafe fn store_unaligned(ptr: *mut $t, val: Self::Vector) {
                core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, $lanes);
            }

            #[inline(always)]
            unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i] + b[i])
            }

            #[inline(always)]
            unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i] * b[i])
            }

            #[inline(always)]
            unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i] - b[i])
            }

            #[inline(always)]
            unsafe fn neg(a: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| -a[i])
            }

            #[inline(always)]
            unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| {
                    <$t as hermes_simd_core::scalar::NumericElement>::scalar_fmadd(a[i], b[i], c[i])
                })
            }

            #[inline(always)]
            unsafe fn sum_reduce(v: Self::Vector) -> $t {
                v.iter().copied().fold(
                    <$t as hermes_simd_core::scalar::NumericElement>::ZERO,
                    |acc, x| acc + x,
                )
            }

            // masked_load_unaligned / masked_store_unaligned / masked_add /
            // masked_mul / masked_fmadd / masked_sum_reduce are inherited from the
            // `SimdKernel` scalar-emulated defaults (blend / generic_masked_*),
            // which are bit-identical to the per-element loops they replaced.

            #[inline(always)]
            unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
                let mut out = [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes];
                let mut k = 0;
                for i in 0..$lanes {
                    if mask[i] {
                        out[k] = src[i];
                        k += 1;
                    }
                }
                out
            }

            #[inline(always)]
            unsafe fn expand(
                src: Self::Vector,
                mask: Self::Mask,
                fill: Self::Vector,
            ) -> Self::Vector {
                let mut out = fill;
                let mut k = 0;
                for i in 0..$lanes {
                    if mask[i] {
                        out[i] = src[k];
                        k += 1;
                    }
                }
                out
            }

            #[inline(always)]
            unsafe fn gather(base: *const $t, indices: Self::IndexVector) -> Self::Vector {
                core::array::from_fn(|i| *base.add(indices[i] as usize))
            }

            #[inline(always)]
            unsafe fn gather_masked(
                base: *const $t,
                indices: Self::IndexVector,
                mask: Self::Mask,
                src: Self::Vector,
            ) -> Self::Vector {
                core::array::from_fn(|i| {
                    if mask[i] {
                        *base.add(indices[i] as usize)
                    } else {
                        src[i]
                    }
                })
            }

            #[inline(always)]
            unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
                debug_assert_eq!(bits.len(), $lanes);
                core::array::from_fn(|i| bits[i])
            }

            #[inline(always)]
            unsafe fn leading_k_mask(k: usize) -> Self::Mask {
                core::array::from_fn(|i| i < k)
            }

            #[inline(always)]
            unsafe fn zero() -> Self::Vector {
                [<$t as hermes_simd_core::scalar::NumericElement>::ZERO; $lanes]
            }

            #[inline(always)]
            unsafe fn splat(val: $t) -> Self::Vector {
                [val; $lanes]
            }

            #[inline(always)]
            unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
                let mut m = 0u64;
                for i in 0..$lanes {
                    if mask[i] {
                        m |= 1u64 << i;
                    }
                }
                m
            }

            #[inline(always)]
            unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
                core::array::from_fn(|i| {
                    if mask[i] {
                        <$t as hermes_simd_core::scalar::NumericElement>::ALL_ONES
                    } else {
                        <$t as hermes_simd_core::scalar::NumericElement>::ZERO
                    }
                })
            }
        }
    };
}

pub mod aarch64;
pub mod bitboard;
pub mod scalar;
pub mod x86_64;

// Re-export SVE marker at crate root for ergonomic access.
pub use aarch64::sve::SveArch;

// Re-export bitboard backend markers.
pub use bitboard::hybrid::HybridSwarMagic;
pub use bitboard::hyperbola::Hyperbola;
pub use bitboard::kogge_stone::KoggeStone;
pub use bitboard::magic::Magic;
pub use bitboard::swar::{Swar, SwarUtils};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use x86_64::amx::{AmxBatchSession, AmxBf16, AmxConfig, AmxInt8, AmxSession, AmxSessionError};

// ---------------------------------------------------------------------------
// ZST Architecture Markers
// ---------------------------------------------------------------------------

/// Fallback scalar implementation marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scalar;

/// x86/x86_64 AVX2 instruction set architecture marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Avx2;

/// x86/x86_64 AVX-512F instruction set architecture marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Avx512;

/// x86/x86_64 AVX-VNNI (256-bit VEX-encoded VNNI) instruction set marker.
///
/// Distinct from [`Avx512`]: AVX-VNNI provides `vpdpbusd`/`vpdpwssd` on 256-bit
/// YMM registers without requiring AVX-512, so it accelerates integer dot/GEMM
/// on client CPUs (Alder Lake and newer) that lack AVX-512. The int8 tile kernel
/// uses the unsigned-signed `vpdpbusd` with a `+128` operand bias correction
/// (base AVX-VNNI has no signed-signed `vpdpbssd`; that is the separate
/// `avxvnniint8` feature).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AvxVnni;

/// AArch64 NEON instruction set architecture marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neon;

// ---------------------------------------------------------------------------
// SimdArch impls
// ---------------------------------------------------------------------------

impl SimdArch for Scalar {
    const NAME: &'static str = "scalar";
    const REGISTER_WIDTH_BITS: u32 = 0;
    const ISA_FAMILY: hermes_simd_core::arch::IsaFamily = hermes_simd_core::arch::IsaFamily::Scalar;
    const FMA_THROUGHPUT_HINT: u32 = 1;

    #[inline]
    fn is_runtime_supported() -> bool {
        true
    }
}

impl SimdArch for Avx2 {
    const NAME: &'static str = "avx2";
    const REGISTER_WIDTH_BITS: u32 = 256;
    const ISA_FAMILY: hermes_simd_core::arch::IsaFamily = hermes_simd_core::arch::IsaFamily::X86;
    const FMA_THROUGHPUT_HINT: u32 = 4;

    #[inline]
    fn is_runtime_supported() -> bool {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
        {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "std")))]
        {
            cfg!(target_feature = "avx2") && cfg!(target_feature = "fma")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
}

impl SimdArch for Avx512 {
    const NAME: &'static str = "avx512";
    const REGISTER_WIDTH_BITS: u32 = 512;
    const ISA_FAMILY: hermes_simd_core::arch::IsaFamily = hermes_simd_core::arch::IsaFamily::X86;
    const FMA_THROUGHPUT_HINT: u32 = 8;

    #[inline]
    fn is_runtime_supported() -> bool {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
        {
            std::is_x86_feature_detected!("avx512f")
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "std")))]
        {
            cfg!(target_feature = "avx512f")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
}

impl SimdArch for AvxVnni {
    const NAME: &'static str = "avx_vnni";
    const REGISTER_WIDTH_BITS: u32 = 256;
    const ISA_FAMILY: hermes_simd_core::arch::IsaFamily = hermes_simd_core::arch::IsaFamily::X86;
    // Two 256-bit `vpdpbusd` per cycle on current client cores (Golden Cove / Zen 4).
    const FMA_THROUGHPUT_HINT: u32 = 4;

    #[inline]
    fn is_runtime_supported() -> bool {
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "std"))]
        {
            std::is_x86_feature_detected!("avxvnni")
        }
        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(feature = "std")))]
        {
            false
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
}

impl SimdArch for Neon {
    const NAME: &'static str = "neon";
    const REGISTER_WIDTH_BITS: u32 = 128;
    const ISA_FAMILY: hermes_simd_core::arch::IsaFamily =
        hermes_simd_core::arch::IsaFamily::AArch64;
    const FMA_THROUGHPUT_HINT: u32 = 4;

    #[inline]
    fn is_runtime_supported() -> bool {
        cfg!(target_arch = "aarch64")
    }
}

impl hermes_simd_core::private::Sealed for Scalar {}
impl hermes_simd_core::private::Sealed for Avx2 {}
impl hermes_simd_core::private::Sealed for Avx512 {}
impl hermes_simd_core::private::Sealed for AvxVnni {}
impl hermes_simd_core::private::Sealed for Neon {}
impl hermes_simd_core::private::Sealed for SveArch {}
