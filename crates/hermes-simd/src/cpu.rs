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

impl AmxSupport for half::bf16 {
    #[inline]
    fn has_amx() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let res = core::arch::x86_64::__cpuid_count(7, 0);
            let amx_tile = (res.edx & (1 << 24)) != 0;
            let amx_bf16 = (res.edx & (1 << 22)) != 0;
            amx_tile && amx_bf16
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
            let res = core::arch::x86_64::__cpuid_count(7, 0);
            let amx_tile = (res.edx & (1 << 24)) != 0;
            let amx_int8 = (res.edx & (1 << 25)) != 0;
            amx_tile && amx_int8
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
            let res = core::arch::x86_64::__cpuid_count(7, 1);
            (res.eax & (1 << 5)) != 0
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
            let res = core::arch::x86_64::__cpuid_count(7, 0);
            (res.ecx & (1 << 11)) != 0
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}
