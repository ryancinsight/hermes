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

impl FmaSupport for half::bf16 {
    #[inline]
    fn has_fma() -> bool {
        has_fma3()
    }
}

impl AmxSupport for half::bf16 {
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

impl Avx512Support for half::bf16 {
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
// AMX dispatch is disabled until Hermes has a stable, permission-aware probe
// that verifies hardware support, XCR0 OS state, and Linux XTILEDATA process
// permission before reporting true. Raw CPUID is insufficient because it misses
// OS enablement and can alias unsupported leaves; the stable Rust feature macro
// does not currently accept AMX feature strings on this toolchain. Returning
// false preserves the safe-dispatch contract instead of risking a #UD/#NM fault.

/// AMX bf16 tile GEMM (`tdpbf16ps` inline asm) requires `amx-tile` + `amx-bf16`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_amx_bf16() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| false)
}

/// AMX int8 tile GEMM (`tdpbssd` inline asm) requires `amx-tile` + `amx-int8`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_amx_int8() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| false)
}

/// The AVX-512 bf16 tile kernels (`avx512_tiling`) enable
/// `avx512f,avx512bw,avx512vl` — they widen bf16 to f32 and FMA in f32, so they
/// need the base 512-bit + byte/word + 128/256-bit-lane extensions, **not** the
/// `avx512bf16` dot-product ISA. Detecting the exact enabled set (rather than the
/// old, mismatched `avx512bf16` bit) both closes the `#UD` window and stops
/// falsely skipping the kernel on capable non-bf16 parts.
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

/// The AVX-512 int8 tile kernels enable `avx512f,avx512vnni,avx512vl`.
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512_vnni_tile() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vnni")
            && std::is_x86_feature_detected!("avx512vl")
    })
}
