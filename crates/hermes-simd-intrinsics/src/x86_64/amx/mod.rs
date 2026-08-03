//! Intel AMX (Advanced Matrix Extensions) backend for BF16 and INT8 matrix multiplication.

mod config;
mod session;
mod types;

pub use config::AmxConfig;
pub(crate) use config::ACTIVE_CONFIG;
pub use session::{AmxBatchSession, AmxSession, AmxSessionError};
pub use types::{AmxBf16, AmxInt8};

#[inline]
fn amx_runtime_supported() -> bool {
    #[cfg(miri)]
    {
        true
    }
    #[cfg(not(miri))]
    {
        false
    }
}

/// AMX instruction wrappers using inline assembly.
pub mod raw {
    use super::AmxConfig;

    /// Load tile configuration from memory.
    #[inline(always)]
    pub unsafe fn ldtilecfg(config: &AmxConfig) {
        let _ = config;
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "ldtilecfg [{ptr}]",
                ptr = in(reg) config,
            );
        }
    }

    /// Store tile configuration to memory.
    #[inline(always)]
    pub unsafe fn sttilecfg(config: &mut AmxConfig) {
        let _ = config;
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "sttilecfg [{ptr}]",
                ptr = in(reg) config,
            );
        }
    }

    /// Release AMX tile configuration (returns tile state to initialized).
    #[inline(always)]
    pub unsafe fn tilerelease() {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!("tilerelease");
        }
    }

    /// Zero out a tile register.
    #[inline(always)]
    pub unsafe fn tilezero(tile: u8) {
        #[cfg(miri)]
        {
            let _ = tile;
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            match tile {
                0 => core::arch::asm!("tilezero tmm0"),
                1 => core::arch::asm!("tilezero tmm1"),
                2 => core::arch::asm!("tilezero tmm2"),
                3 => core::arch::asm!("tilezero tmm3"),
                4 => core::arch::asm!("tilezero tmm4"),
                5 => core::arch::asm!("tilezero tmm5"),
                6 => core::arch::asm!("tilezero tmm6"),
                7 => core::arch::asm!("tilezero tmm7"),
                _ => unreachable!("AMX tile index out of range (valid: tmm0-tmm7)"),
            }
        }
    }

    /// Load 2D data from memory into a tile register.
    #[inline(always)]
    pub unsafe fn tileloadd(tile: u8, base: *const core::ffi::c_void, stride: isize) {
        #[cfg(miri)]
        {
            let _ = (tile, base, stride);
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            match tile {
                0 => {
                    core::arch::asm!("tileloadd tmm0, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                1 => {
                    core::arch::asm!("tileloadd tmm1, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                2 => {
                    core::arch::asm!("tileloadd tmm2, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                3 => {
                    core::arch::asm!("tileloadd tmm3, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                4 => {
                    core::arch::asm!("tileloadd tmm4, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                5 => {
                    core::arch::asm!("tileloadd tmm5, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                6 => {
                    core::arch::asm!("tileloadd tmm6, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                7 => {
                    core::arch::asm!("tileloadd tmm7, [{base} + {stride}]", base = in(reg) base, stride = in(reg) stride)
                }
                _ => unreachable!("AMX tile index out of range (valid: tmm0-tmm7)"),
            }
        }
    }

    /// Store 2D data from a tile register into memory.
    #[inline(always)]
    pub unsafe fn tilestored(tile: u8, base: *mut core::ffi::c_void, stride: isize) {
        #[cfg(miri)]
        {
            let _ = (tile, base, stride);
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            match tile {
                0 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm0", base = in(reg) base, stride = in(reg) stride)
                }
                1 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm1", base = in(reg) base, stride = in(reg) stride)
                }
                2 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm2", base = in(reg) base, stride = in(reg) stride)
                }
                3 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm3", base = in(reg) base, stride = in(reg) stride)
                }
                4 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm4", base = in(reg) base, stride = in(reg) stride)
                }
                5 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm5", base = in(reg) base, stride = in(reg) stride)
                }
                6 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm6", base = in(reg) base, stride = in(reg) stride)
                }
                7 => {
                    core::arch::asm!("tilestored [{base} + {stride}], tmm7", base = in(reg) base, stride = in(reg) stride)
                }
                _ => unreachable!("AMX tile index out of range (valid: tmm0-tmm7)"),
            }
        }
    }

    /// Compute F32 dot product of BF16 elements: dst += src1 * src2
    #[inline(always)]
    pub unsafe fn tdpbf16ps(dst: u8, src1: u8, src2: u8) {
        #[cfg(miri)]
        {
            let _ = (dst, src1, src2);
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            match (dst, src1, src2) {
                (2, 0, 6) => core::arch::asm!("tdpbf16ps tmm2, tmm0, tmm6"),
                (3, 0, 7) => core::arch::asm!("tdpbf16ps tmm3, tmm0, tmm7"),
                (4, 1, 6) => core::arch::asm!("tdpbf16ps tmm4, tmm1, tmm6"),
                (5, 1, 7) => core::arch::asm!("tdpbf16ps tmm5, tmm1, tmm7"),
                (5, 3, 4) => core::arch::asm!("tdpbf16ps tmm5, tmm3, tmm4"),
                (4, 0, 2) => core::arch::asm!("tdpbf16ps tmm4, tmm0, tmm2"),
                (5, 0, 3) => core::arch::asm!("tdpbf16ps tmm5, tmm0, tmm3"),
                (6, 1, 2) => core::arch::asm!("tdpbf16ps tmm6, tmm1, tmm2"),
                (7, 1, 3) => core::arch::asm!("tdpbf16ps tmm7, tmm1, tmm3"),
                (0, 1, 2) => core::arch::asm!("tdpbf16ps tmm0, tmm1, tmm2"),
                (2, 0, 1) => core::arch::asm!("tdpbf16ps tmm2, tmm0, tmm1"),
                _ => unreachable!("AMX tile index out of range (valid: tmm0-tmm7)"),
            }
        }
    }

    /// Compute INT32 dot product of INT8 elements: dst += src1 * src2
    #[inline(always)]
    pub unsafe fn tdpbssd(dst: u8, src1: u8, src2: u8) {
        #[cfg(miri)]
        {
            let _ = (dst, src1, src2);
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            match (dst, src1, src2) {
                (2, 0, 6) => core::arch::asm!("tdpbssd tmm2, tmm0, tmm6"),
                (3, 0, 7) => core::arch::asm!("tdpbssd tmm3, tmm0, tmm7"),
                (4, 1, 6) => core::arch::asm!("tdpbssd tmm4, tmm1, tmm6"),
                (5, 1, 7) => core::arch::asm!("tdpbssd tmm5, tmm1, tmm7"),
                (5, 3, 4) => core::arch::asm!("tdpbssd tmm5, tmm3, tmm4"),
                (4, 0, 2) => core::arch::asm!("tdpbssd tmm4, tmm0, tmm2"),
                (5, 0, 3) => core::arch::asm!("tdpbssd tmm5, tmm0, tmm3"),
                (6, 1, 2) => core::arch::asm!("tdpbssd tmm6, tmm1, tmm2"),
                (7, 1, 3) => core::arch::asm!("tdpbssd tmm7, tmm1, tmm3"),
                (0, 1, 2) => core::arch::asm!("tdpbssd tmm0, tmm1, tmm2"),
                (2, 0, 1) => core::arch::asm!("tdpbssd tmm2, tmm0, tmm1"),
                _ => unreachable!("AMX tile index out of range (valid: tmm0-tmm7)"),
            }
        }
    }
}

/// Trait for performing blocked matrix multiplication using Intel AMX.
pub trait AmxGemm<TA, TB, TC> {
    /// Perform matrix multiplication `c += a * b` using Intel AMX block execution.
    ///
    /// # Safety
    /// - Pointers must be valid and aligned as per backend requirements.
    /// - AMX must be available and enabled before calling: the CPU must support
    ///   the relevant AMX features (`amx-tile` plus `amx-bf16`/`amx-int8`) and,
    ///   on OSes that gate tile state (e.g. Linux `arch_prctl(ARCH_REQ_XCOMP_PERM,
    ///   XFEATURE_XTILEDATA)`), the process must have requested permission.
    ///   Callers gate this behind a runtime probe (the dispatcher reaches this
    ///   only via `DispatchDecision::Amx`, taken when `AmxSupport::has_amx()` is
    ///   true); invoking it without that guarantee is a `#UD`/`#NM` fault.
    unsafe fn amx_gemm(
        m: usize,
        n: usize,
        k: usize,
        a: *const TA,
        a_stride: usize,
        b: *const TB,
        b_stride: usize,
        c: *mut TC,
        c_stride: usize,
    );
}

/// AMX tile GEMM over brain-float-16 inputs (`tdpbf16ps`).
pub mod bf16;
/// AMX tile GEMM over signed 8-bit integer inputs (`tdpbssd`).
pub mod int8;

#[cfg(all(test, miri))]
mod tests {
    use super::{AmxBatchSession, AmxConfig, AmxSession};

    #[test]
    fn miri_session_nesting_preserves_active_config_until_outer_drop() {
        let config = AmxConfig::new_uniform(16, 64);
        assert!(!AmxSession::is_active());

        {
            let _outer = AmxSession::new(&config).unwrap();
            assert!(AmxSession::is_active());
            {
                let _inner = AmxSession::new(&config).unwrap();
                assert!(AmxSession::is_active());
            }
            assert!(AmxSession::is_active());
        }

        assert!(!AmxSession::is_active());
    }

    #[test]
    fn miri_explicit_release_resets_session_state() {
        let config = AmxConfig::new_uniform(16, 64);
        let _session = AmxSession::new(&config).unwrap();
        assert!(AmxSession::is_active());

        AmxSession::release();
        assert!(!AmxSession::is_active());
    }

    #[test]
    fn miri_batch_session_drop_resets_session_state() {
        let config = AmxConfig::new_uniform(16, 64);
        {
            let _session = AmxBatchSession::begin(&config).unwrap();
            assert!(AmxSession::is_active());
        }
        assert!(!AmxSession::is_active());
    }
}
