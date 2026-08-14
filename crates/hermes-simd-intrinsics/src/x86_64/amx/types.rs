use hermes_simd_core::arch::{IsaFamily, SimdArch};

/// `x86/x86_64` AMX BF16 matrix multiply backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxBf16;

/// `x86/x86_64` AMX INT8 matrix multiply backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxInt8;

impl SimdArch for AmxBf16 {
    const NAME: &'static str = "amx_bf16";
    const REGISTER_WIDTH_BITS: u32 = 8192; // AMX tile registers are 1024 bytes (8192 bits) each
    const ISA_FAMILY: IsaFamily = IsaFamily::X86;
    const FMA_THROUGHPUT_HINT: u32 = 16;

    #[inline]
    fn is_runtime_supported() -> bool {
        super::amx_runtime_supported()
    }
}

impl SimdArch for AmxInt8 {
    const NAME: &'static str = "amx_int8";
    const REGISTER_WIDTH_BITS: u32 = 8192;
    const ISA_FAMILY: IsaFamily = IsaFamily::X86;
    const FMA_THROUGHPUT_HINT: u32 = 16;

    #[inline]
    fn is_runtime_supported() -> bool {
        super::amx_runtime_supported()
    }
}

impl hermes_simd_core::private::Sealed for AmxBf16 {}
impl hermes_simd_core::private::Sealed for AmxInt8 {}
