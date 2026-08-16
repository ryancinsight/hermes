//! Intel AMX (Advanced Matrix Extensions) backend for BF16 and INT8 matrix multiplication.

mod config;
mod pack;
pub mod probe;
mod session;
mod types;

pub use config::AmxConfig;
#[cfg(feature = "std")]
pub(crate) use config::ACTIVE_CONFIG;
pub use probe::{has_amx_bf16, has_amx_int8, has_amx_tile};
pub use session::{AmxBatchSession, AmxSession, AmxSessionError};
pub use types::{AmxBf16, AmxInt8};

/// Whether this process may execute AMX tile instructions.
///
/// Delegates to [`probe::has_amx_tile`], which owns the CPUID / `XCR0` /
/// OS-permission chain. Under Miri no instruction actually executes — the
/// `raw` wrappers are compiled out and the session tests exercise only the
/// configuration state machine — so the probe is bypassed to keep that
/// coverage reachable.
#[inline]
fn amx_runtime_supported() -> bool {
    #[cfg(miri)]
    {
        true
    }
    #[cfg(not(miri))]
    {
        #[cfg(feature = "std")]
        {
            probe::has_amx_tile()
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }
}

/// AMX instruction wrappers using inline assembly.
///
/// # `asm!` option policy
///
/// Every block here declares the strongest options that are *literally* true
/// of its instruction, because the default (no options) makes the compiler
/// assume an arbitrary memory clobber and forces spills around each tile op in
/// the GEMM inner loop. The three axes:
///
/// - `nomem` where the instruction touches only tile registers (`tilezero`,
///   `tilerelease`, `tdpbf16ps`, `tdpbssd`); `readonly` where it reads memory
///   but never writes it (`ldtilecfg`, `tileloadd`). `sttilecfg` and
///   `tilestored` write through their pointer, so they claim neither and keep
///   the conservative default.
/// - `nostack`: no AMX instruction pushes, pops, or uses the red zone. A
///   pointer operand that happens to address the stack is unaffected — the
///   option constrains what the asm itself does with the stack pointer.
/// - `preserves_flags`: no AMX instruction writes `EFLAGS`.
///
/// `pure` is deliberately absent everywhere, including from the `nomem`
/// blocks. Every one of these mutates tile-register state that LLVM does not
/// model, so the implicit "has side effects" of a non-`pure` block is the only
/// thing keeping `ldtilecfg` ahead of the `tileloadd`/`tdp*` sequence that
/// depends on it, and keeping a reused accumulator's ops in order. Marking any
/// of them `pure` would license reordering or elision and silently corrupt the
/// GEMM.
pub mod raw {
    use super::AmxConfig;

    /// Load tile configuration from memory.
    ///
    /// # Safety
    /// The current thread must have AMX tile permission, and `config` must be
    /// a valid, 64-byte-aligned `TILECFG` value for the executing processor.
    #[inline(always)]
    pub unsafe fn ldtilecfg(config: &AmxConfig) {
        let _ = config;
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "ldtilecfg [{ptr}]",
                ptr = in(reg) config,
                options(readonly, nostack, preserves_flags),
            );
        }
    }

    /// Store tile configuration to memory.
    ///
    /// # Safety
    /// The current thread must have AMX tile permission, and `config` must be
    /// valid writable, 64-byte-aligned storage for the processor's `TILECFG`
    /// representation.
    #[inline(always)]
    pub unsafe fn sttilecfg(config: &mut AmxConfig) {
        let _ = config;
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "sttilecfg [{ptr}]",
                ptr = in(reg) config,
                options(nostack, preserves_flags),
            );
        }
    }

    /// Release AMX tile configuration (returns tile state to initialized).
    ///
    /// # Safety
    /// The current thread must have AMX tile permission. The caller must not
    /// use the tile registers after this call until a valid configuration is
    /// loaded again.
    #[inline(always)]
    pub unsafe fn tilerelease() {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!("tilerelease", options(nomem, nostack, preserves_flags));
        }
    }

    /// Zero out a tile register.
    ///
    /// The tile index is a compile-time constant: `TILE` must name a valid
    /// register (`0..8`), and the assembler rejects out-of-range indices at
    /// compile time.
    ///
    /// # Safety
    /// The current thread must have an active AMX tile configuration.
    #[inline(always)]
    pub unsafe fn tilezero<const TILE: u8>() {
        #[cfg(miri)]
        {
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "tilezero tmm{TILE}",
                TILE = const TILE,
                options(nomem, nostack, preserves_flags),
            );
        }
    }

    /// Load 2D data from memory into a tile register.
    ///
    /// The tile index is a compile-time constant: `TILE` must name a valid
    /// register (`0..8`), and the assembler rejects out-of-range indices at
    /// compile time.
    ///
    /// # Safety
    /// The current thread must have an active AMX tile configuration. `base`
    /// and `stride` must describe readable rows whose lengths and spacing
    /// satisfy the loaded tile's configured shape; every address accessed by
    /// the instruction must be valid for the duration of the call.
    #[inline(always)]
    pub unsafe fn tileloadd<const TILE: u8>(base: *const core::ffi::c_void, stride: isize) {
        #[cfg(miri)]
        {
            let _ = (base, stride);
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "tileloadd tmm{TILE}, [{base} + {stride}]",
                TILE = const TILE,
                base = in(reg) base,
                stride = in(reg) stride,
                options(readonly, nostack, preserves_flags),
            );
        }
    }

    /// Store 2D data from a tile register into memory.
    ///
    /// The tile index is a compile-time constant: `TILE` must name a valid
    /// register (`0..8`), and the assembler rejects out-of-range indices at
    /// compile time.
    ///
    /// # Safety
    /// The current thread must have an active AMX tile configuration. `base`
    /// and `stride` must describe writable rows whose lengths and spacing
    /// satisfy the stored tile's configured shape; every address accessed by
    /// the instruction must be valid for the duration of the call.
    #[inline(always)]
    pub unsafe fn tilestored<const TILE: u8>(base: *mut core::ffi::c_void, stride: isize) {
        #[cfg(miri)]
        {
            let _ = (base, stride);
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "tilestored [{base} + {stride}], tmm{TILE}",
                TILE = const TILE,
                base = in(reg) base,
                stride = in(reg) stride,
                options(nostack, preserves_flags),
            );
        }
    }

    /// Compute F32 dot product of BF16 elements: dst += src1 * src2
    ///
    /// The tile indices are compile-time constants: `DST`, `SRC1`, and `SRC2`
    /// must name valid registers (`0..8`), and the assembler rejects
    /// out-of-range indices at compile time.
    ///
    /// # Safety
    /// The current thread must have an active AMX tile configuration, and the
    /// tiles must have configured shapes and BF16 dot-product layout accepted
    /// by `TDPBF16PS`. The destination tile must be initialized for
    /// accumulation.
    #[inline(always)]
    pub unsafe fn tdpbf16ps<const DST: u8, const SRC1: u8, const SRC2: u8>() {
        #[cfg(miri)]
        {
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "tdpbf16ps tmm{DST}, tmm{SRC1}, tmm{SRC2}",
                DST = const DST,
                SRC1 = const SRC1,
                SRC2 = const SRC2,
                options(nomem, nostack, preserves_flags),
            );
        }
    }

    /// Compute INT32 dot product of INT8 elements: dst += src1 * src2
    ///
    /// The tile indices are compile-time constants: `DST`, `SRC1`, and `SRC2`
    /// must name valid registers (`0..8`), and the assembler rejects
    /// out-of-range indices at compile time.
    ///
    /// # Safety
    /// The current thread must have an active AMX tile configuration, and the
    /// tiles must have configured shapes and INT8 VNNI dot-product layout
    /// accepted by `TDPBSSD`. The destination tile must be initialized for
    /// accumulation.
    #[inline(always)]
    pub unsafe fn tdpbssd<const DST: u8, const SRC1: u8, const SRC2: u8>() {
        #[cfg(miri)]
        {
            panic!("AMX tile execution is not available under Miri");
        }
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            core::arch::asm!(
                "tdpbssd tmm{DST}, tmm{SRC1}, tmm{SRC2}",
                DST = const DST,
                SRC1 = const SRC1,
                SRC2 = const SRC2,
                options(nomem, nostack, preserves_flags),
            );
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
