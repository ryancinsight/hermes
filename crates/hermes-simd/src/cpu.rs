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
            // Cache the result — CPU capabilities are immutable at runtime and
            // `cpuid` is a serializing instruction (~50-200 cycles).  A one-time
            // `OnceLock` init pays the cost once per process lifetime; steady-state
            // calls pay a single relaxed-atomic load.
            static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| {
                let res = core::arch::x86_64::__cpuid_count(7, 0);
                let amx_tile = (res.edx & (1 << 24)) != 0;
                let amx_bf16 = (res.edx & (1 << 22)) != 0;
                amx_tile && amx_bf16
            })
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
            static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| {
                let res = core::arch::x86_64::__cpuid_count(7, 0);
                let amx_tile = (res.edx & (1 << 24)) != 0;
                let amx_int8 = (res.edx & (1 << 25)) != 0;
                amx_tile && amx_int8
            })
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
            static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| {
                let res = core::arch::x86_64::__cpuid_count(7, 1);
                (res.eax & (1 << 5)) != 0
            })
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
            static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| {
                let res = core::arch::x86_64::__cpuid_count(7, 0);
                (res.ecx & (1 << 11)) != 0
            })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}
