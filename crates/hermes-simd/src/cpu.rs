/// Trait for querying runtime Intel AMX support for specific element types.
pub trait AmxSupport {
    /// Returns true if the current CPU supports AMX for this type.
    fn has_amx() -> bool;
}

/// Trait for querying runtime AVX-512 support for specific element types.
pub trait Avx512Support {
    /// Returns true if the current CPU supports AVX-512 extensions/fallbacks for this type.
    fn has_avx512() -> bool;
}

/// Trait for querying runtime FMA (Fused Multiply-Add) support.
///
/// FMA (`vfmadd*` family) is available on Intel Haswell+ and AMD Piledriver+.
/// It performs `a * b + c` in a single instruction with one rounding for
/// floating-point kernels that select an FMA-capable implementation.
pub trait FmaSupport {
    /// Returns `true` if the current CPU supports FMA3 instructions.
    fn has_fma() -> bool;
}

/// Global OnceLock-cached FMA3 probe.
///
/// Separate from the per-type traits above because FMA availability is
/// processor-wide; scalar type implementations only expose whether their
/// operation family can use that host capability.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_fma3() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::is_x86_feature_detected!("fma"))
}

/// FMA3 is an x86_64-specific extension; every other architecture reports
/// no support (aarch64/NEON has its own always-available fused multiply-add
/// instructions, dispatched separately from this x86-specific probe).
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn has_fma3() -> bool {
    false
}

impl FmaSupport for f32 {
    #[inline]
    fn has_fma() -> bool {
        has_fma3()
    }
}

impl FmaSupport for f64 {
    #[inline]
    fn has_fma() -> bool {
        has_fma3()
    }
}

impl FmaSupport for Bf16 {
    #[inline]
    fn has_fma() -> bool {
        has_fma3()
    }
}

impl AmxSupport for Bf16 {
    #[inline]
    fn has_amx() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            has_amx_bf16()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

impl AmxSupport for i8 {
    #[inline]
    fn has_amx() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            has_amx_int8()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

impl Avx512Support for Bf16 {
    #[inline]
    fn has_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            has_avx512_bf16_tile()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

impl Avx512Support for i8 {
    #[inline]
    fn has_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            has_avx512_vnni_tile()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

// AMX / AVX-512 tile-kernel capability probes.
//
// The AMX probes delegate to `hermes-simd-intrinsics`, which owns the capability
// SSOT and caches its result. Raw CPUID alone is unsound: it misses both the
// XCR0 OS-enablement state and the per-process XTILEDATA permission that Linux
// and Windows each gate behind their own protocol, and Rust's stable feature
// macro does not accept AMX feature strings on this toolchain. See
// `hermes_simd_intrinsics::x86_64::amx::probe` for the full chain.

/// AMX bf16 tile GEMM (`tdpbf16ps` inline asm) requires `amx-tile` + `amx-bf16`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_amx_bf16() -> bool {
    hermes_simd_intrinsics::has_amx_bf16()
}

/// AMX int8 tile GEMM (`tdpbssd` inline asm) requires `amx-tile` + `amx-int8`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_amx_int8() -> bool {
    hermes_simd_intrinsics::has_amx_int8()
}

/// The AVX-512 BF16 conversion/FMA tile fallback enables
/// `avx512f,avx512bw,avx512vl` — it widens BF16 to f32 and performs f32 FMA.
/// This is intentionally separate from [`has_avx512_bf16`], which reports the
/// native `DPBF16PS` instruction capability.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512_bf16_tile() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
    })
}

/// Returns whether native AVX-512 BF16 dot-product instructions are usable.
///
/// This probe is distinct from [`Avx512Support::has_avx512`]: the latter also
/// admits the conversion/FMA fallback on AVX-512F/BW/VL hosts. Native callers
/// must use this exact capability before entering `DPBF16PS` code.
#[inline]
pub fn has_avx512_bf16() -> bool {
    hermes_simd_intrinsics::has_avx512_bf16()
}

/// 256-bit VEX-encoded AVX-VNNI (`vpdpbusd`/`vpdpwssd` on YMM without AVX-512).
///
/// Present on Intel Alder Lake+ client parts and AMD Zen 5 — hardware that has
/// no AVX-512. The int8 tile kernel gated on this probe requires only the
/// `avxvnni` feature (the signed-signed `vpdpbssd` from `avxvnniint8` is NOT
/// assumed; the kernel bias-corrects `vpdpbusd` instead). The macro handles
/// XCR0/OSXSAVE and the CPUID max-leaf internally.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx_vnni() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::is_x86_feature_detected!("avxvnni"))
}

/// The AVX-512 int8 tile kernel enables `avx512f,avx512vnni` and implements
/// signed-byte products through `VPDPBUSD` bias correction.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512_vnni_tile() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vnni")
    })
}
use eunomia::Bf16;
