//! AArch64 SVE (Scalable Vector Extension) marker stub.
//!
//! SVE provides hardware predicate registers and scalable vector widths
//! (128–2048 bits in 128-bit increments, determined at runtime via the
//! `rdvl` instruction). This module provides the [`SveArch`] ZST marker
//! and skeleton [`SimdKernel`] implementations.
//!
//! Full SVE intrinsic implementations require the `+sve` target feature and
//! nightly compiler support for `core::arch::aarch64::svfloat32_t` etc.
//!
//! # Status
//!
//! **Stub only.** All methods `unimplemented!()` at runtime. Replace with real
//! SVE intrinsics when targeting SVE-capable hardware:
//! - Cortex-A510, A710, A720 (ARMv9-A)
//! - Neoverse N2, V1, V2
//! - Apple M4 and later (SVE2)
//!
//! Enable with `RUSTFLAGS="-C target-feature=+sve"`.

use hermes_simd_core::{arch::SimdArch, kernel::SimdKernel};

/// AArch64 SVE architecture ZST marker.
///
/// Represents the Scalable Vector Extension instruction set.
/// Carries no data and is resolved to a compile-time constant by the
/// monomorphizer — zero runtime overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SveArch;

impl SimdArch for SveArch {
    const NAME: &'static str = "sve";
    const REGISTER_WIDTH_BITS: u32 = 0;  // SVE is variable-width; 0 = runtime-determined
    const ISA_FAMILY: hermes_simd_core::arch::IsaFamily = hermes_simd_core::arch::IsaFamily::AArch64;
    const FMA_THROUGHPUT_HINT: u32 = 4;
}

// ---------------------------------------------------------------------------
// SVE f32 stub
//
// `LANE_COUNT` is a compile-time approximation: SVE vectors are runtime-
// scalable. The actual lane count is `svcntw()` at runtime. This stub uses
// 16 as a reasonable upper-bound placeholder (512-bit vector length).
// A production implementation should query the lane count at runtime and
// use `svbool_t` / `svfloat32_t` throughout.
// ---------------------------------------------------------------------------

/// Opaque SVE f32 vector placeholder.
///
/// In a real implementation this would be `svfloat32_t`, which is a
/// scalable-vector type requiring nightly `target_feature = "sve"`.
#[derive(Copy, Clone)]
pub struct SveF32Vec {
    // 16 × f32 conservative upper-bound storage for the stub.
    // Replace with `svfloat32_t` on a nightly SVE target.
    _storage: [f32; 16],
}

/// Opaque SVE f32 predicate mask placeholder.
///
/// In a real implementation this would be `svbool_t`.
#[derive(Copy, Clone)]
pub struct SveF32Mask {
    _storage: [bool; 16],
}

unsafe impl Send for SveF32Vec {}
unsafe impl Sync for SveF32Vec {}
unsafe impl Send for SveF32Mask {}
unsafe impl Sync for SveF32Mask {}

impl SimdKernel<f32> for SveArch {
    type Vector = SveF32Vec;
    type Mask = SveF32Mask;
    /// Scalar index array fallback. SVE uses `svindex_s32` / `svld1_gather`.
    type IndexVector = [i32; 16];
    /// Compile-time approximation. Runtime: query `svcntw()`.
    const LANE_COUNT: usize = 16;
    const UNROLL_FACTOR: usize = 4;

    unsafe fn load_aligned(_ptr: *const f32) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svld1_f32")
    }
    unsafe fn load_unaligned(_ptr: *const f32) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svld1_f32")
    }
    unsafe fn store_aligned(_ptr: *mut f32, _val: Self::Vector) {
        unimplemented!("SVE f32 stub: replace with svst1_f32")
    }
    unsafe fn store_unaligned(_ptr: *mut f32, _val: Self::Vector) {
        unimplemented!("SVE f32 stub: replace with svst1_f32")
    }
    unsafe fn add(_a: Self::Vector, _b: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svadd_f32_z")
    }
    unsafe fn mul(_a: Self::Vector, _b: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svmul_f32_z")
    }
    unsafe fn fmadd(_a: Self::Vector, _b: Self::Vector, _c: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svmla_f32_z")
    }
    unsafe fn sum_reduce(_v: Self::Vector) -> f32 {
        unimplemented!("SVE f32 stub: replace with svaddv_f32")
    }
    unsafe fn masked_load_unaligned(
        _ptr: *const f32,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svld1_f32 with pg")
    }
    unsafe fn masked_store_unaligned(_ptr: *mut f32, _mask: Self::Mask, _val: Self::Vector) {
        unimplemented!("SVE f32 stub: replace with svst1_f32 with pg")
    }
    unsafe fn masked_add(
        _a: Self::Vector,
        _b: Self::Vector,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svadd_f32_m")
    }
    unsafe fn masked_mul(
        _a: Self::Vector,
        _b: Self::Vector,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svmul_f32_m")
    }
    unsafe fn masked_fmadd(
        _a: Self::Vector,
        _b: Self::Vector,
        _c: Self::Vector,
        _mask: Self::Mask,
    ) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svmla_f32_m")
    }
    unsafe fn masked_sum_reduce(_v: Self::Vector, _mask: Self::Mask) -> f32 {
        unimplemented!("SVE f32 stub: replace with svaddv_f32 with pg")
    }
    unsafe fn compress(_src: Self::Vector, _mask: Self::Mask) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svcompact_f32")
    }
    unsafe fn expand(_src: Self::Vector, _mask: Self::Mask, _fill: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f32 stub: no direct SVE expand — emulate with svsel/scatter")
    }
    unsafe fn gather(_base: *const f32, _indices: Self::IndexVector) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svld1_gather_s32offset_f32")
    }
    unsafe fn gather_masked(
        _base: *const f32,
        _indices: Self::IndexVector,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svld1_gather_s32offset_f32 with pg")
    }
    unsafe fn mask_from_bools(_bits: &[bool]) -> Self::Mask {
        unimplemented!("SVE f32 stub: replace with svwhilelt_b32 / svdupq_n_b32")
    }
    unsafe fn leading_k_mask(_k: usize) -> Self::Mask {
        unimplemented!("SVE f32 stub: replace with svwhilelt_b32(0, k as i64)")
    }
    unsafe fn zero() -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svdup_n_f32(0.0)")
    }
    unsafe fn splat(_val: f32) -> Self::Vector {
        unimplemented!("SVE f32 stub: replace with svdup_n_f32(val)")
    }
    unsafe fn mask_to_bitmask(_mask: Self::Mask) -> u64 {
        unimplemented!("SVE f32 stub")
    }
}

// ---------------------------------------------------------------------------
// SVE f64 stub
// ---------------------------------------------------------------------------

/// Opaque SVE f64 vector placeholder.
#[derive(Copy, Clone)]
pub struct SveF64Vec {
    _storage: [f64; 8],
}

/// Opaque SVE f64 predicate mask placeholder.
#[derive(Copy, Clone)]
pub struct SveF64Mask {
    _storage: [bool; 8],
}

unsafe impl Send for SveF64Vec {}
unsafe impl Sync for SveF64Vec {}
unsafe impl Send for SveF64Mask {}
unsafe impl Sync for SveF64Mask {}

impl SimdKernel<f64> for SveArch {
    type Vector = SveF64Vec;
    type Mask = SveF64Mask;
    type IndexVector = [i32; 8];
    const LANE_COUNT: usize = 8;
    const UNROLL_FACTOR: usize = 4;

    unsafe fn load_aligned(_ptr: *const f64) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svld1_f64")
    }
    unsafe fn load_unaligned(_ptr: *const f64) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svld1_f64")
    }
    unsafe fn store_aligned(_ptr: *mut f64, _val: Self::Vector) {
        unimplemented!("SVE f64 stub: replace with svst1_f64")
    }
    unsafe fn store_unaligned(_ptr: *mut f64, _val: Self::Vector) {
        unimplemented!("SVE f64 stub: replace with svst1_f64")
    }
    unsafe fn add(_a: Self::Vector, _b: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svadd_f64_z")
    }
    unsafe fn mul(_a: Self::Vector, _b: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svmul_f64_z")
    }
    unsafe fn fmadd(_a: Self::Vector, _b: Self::Vector, _c: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svmla_f64_z")
    }
    unsafe fn sum_reduce(_v: Self::Vector) -> f64 {
        unimplemented!("SVE f64 stub: replace with svaddv_f64")
    }
    unsafe fn masked_load_unaligned(
        _ptr: *const f64,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svld1_f64 with pg")
    }
    unsafe fn masked_store_unaligned(_ptr: *mut f64, _mask: Self::Mask, _val: Self::Vector) {
        unimplemented!("SVE f64 stub: replace with svst1_f64 with pg")
    }
    unsafe fn masked_add(
        _a: Self::Vector,
        _b: Self::Vector,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svadd_f64_m")
    }
    unsafe fn masked_mul(
        _a: Self::Vector,
        _b: Self::Vector,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svmul_f64_m")
    }
    unsafe fn masked_fmadd(
        _a: Self::Vector,
        _b: Self::Vector,
        _c: Self::Vector,
        _mask: Self::Mask,
    ) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svmla_f64_m")
    }
    unsafe fn masked_sum_reduce(_v: Self::Vector, _mask: Self::Mask) -> f64 {
        unimplemented!("SVE f64 stub: replace with svaddv_f64 with pg")
    }
    unsafe fn compress(_src: Self::Vector, _mask: Self::Mask) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svcompact_f64")
    }
    unsafe fn expand(_src: Self::Vector, _mask: Self::Mask, _fill: Self::Vector) -> Self::Vector {
        unimplemented!("SVE f64 stub: no direct SVE expand — emulate with svsel/scatter")
    }
    unsafe fn gather(_base: *const f64, _indices: Self::IndexVector) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svld1_gather_s64offset_f64")
    }
    unsafe fn gather_masked(
        _base: *const f64,
        _indices: Self::IndexVector,
        _mask: Self::Mask,
        _src: Self::Vector,
    ) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svld1_gather_s64offset_f64 with pg")
    }
    unsafe fn mask_from_bools(_bits: &[bool]) -> Self::Mask {
        unimplemented!("SVE f64 stub: replace with svwhilelt_b64 / svdupq_n_b64")
    }
    unsafe fn leading_k_mask(_k: usize) -> Self::Mask {
        unimplemented!("SVE f64 stub: replace with svwhilelt_b64(0, k as i64)")
    }
    unsafe fn zero() -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svdup_n_f64(0.0)")
    }
    unsafe fn splat(_val: f64) -> Self::Vector {
        unimplemented!("SVE f64 stub: replace with svdup_n_f64(val)")
    }
    unsafe fn mask_to_bitmask(_mask: Self::Mask) -> u64 {
        unimplemented!("SVE f64 stub")
    }
}
