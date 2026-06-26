//! Intel AMX (Advanced Matrix Extensions) backend for BF16 and INT8 matrix multiplication.

use hermes_simd_core::arch::{IsaFamily, SimdArch};

/// x86/x86_64 AMX BF16 matrix multiply backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxBf16;

/// x86/x86_64 AMX INT8 matrix multiply backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxInt8;

impl SimdArch for AmxBf16 {
    const NAME: &'static str = "amx_bf16";
    const REGISTER_WIDTH_BITS: u32 = 8192; // AMX tile registers are 1024 bytes (8192 bits) each
    const ISA_FAMILY: IsaFamily = IsaFamily::X86;
    const FMA_THROUGHPUT_HINT: u32 = 16;
}

impl SimdArch for AmxInt8 {
    const NAME: &'static str = "amx_int8";
    const REGISTER_WIDTH_BITS: u32 = 8192;
    const ISA_FAMILY: IsaFamily = IsaFamily::X86;
    const FMA_THROUGHPUT_HINT: u32 = 16;
}

impl hermes_simd_core::private::Sealed for AmxBf16 {}
impl hermes_simd_core::private::Sealed for AmxInt8 {}

/// 64-byte AMX tile configuration structure (TILECFG).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxConfig {
    /// Palette ID (0 = init, 1 = Palette 1)
    pub palette_id: u8,
    /// Start row for recovery after interrupt / context switch
    pub start_row: u8,
    /// Reserved bytes
    pub reserved: [u8; 14],
    /// Column widths in bytes for each of the 8 tiles
    pub cols_b: [u16; 8],
    /// Reserved bytes
    pub reserved2: [u8; 16],
    /// Number of rows for each of the 8 tiles
    pub rows: [u8; 8],
    /// Reserved bytes
    pub reserved3: [u8; 8],
}

impl AmxConfig {
    /// Creates a palette 1 configuration where all tiles have the specified rows and byte columns.
    #[inline]
    pub fn new_uniform(r: u8, c_bytes: u16) -> Self {
        Self {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 14],
            cols_b: [c_bytes; 8],
            reserved2: [0; 16],
            rows: [r; 8],
            reserved3: [0; 8],
        }
    }

    /// Creates a custom palette 1 configuration with row/col sizes for each of the 8 tiles.
    #[inline]
    pub fn new_custom(rows: [u8; 8], cols_b: [u16; 8]) -> Self {
        Self {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 14],
            cols_b,
            reserved2: [0; 16],
            rows,
            reserved3: [0; 8],
        }
    }

    /// Generate adaptive tile config based on dynamic matrix dimensions.
    #[inline]
    pub fn for_dimensions(m: usize, n: usize, k: usize, element_size: usize) -> Self {
        let r_a = m.min(16) as u8;
        let c_a_bytes = (k * element_size).min(64) as u16;
        let r_b = k.min(64 / element_size) as u8;
        let c_b_bytes = (n * element_size).min(64) as u16;
        let r_c = m.min(16) as u8;
        let c_c_bytes = (n * 4).min(64) as u16; // Accumulation in 32-bit (F32/I32)

        let mut rows = [0; 8];
        let mut cols_b = [0; 8];

        // Tile 0 (A): M rows x K cols
        rows[0] = r_a;
        cols_b[0] = c_a_bytes;
        // Tile 1 (B): K rows x N cols
        rows[1] = r_b;
        cols_b[1] = c_b_bytes;
        // Tile 2 (C): M rows x N cols
        rows[2] = r_c;
        cols_b[2] = c_c_bytes;

        // Auxiliary registers for register blocking
        rows[3] = r_c;
        cols_b[3] = c_c_bytes;
        rows[4] = r_c;
        cols_b[4] = c_c_bytes;
        rows[5] = r_c;
        cols_b[5] = c_c_bytes;
        rows[6] = r_b;
        cols_b[6] = c_b_bytes;
        rows[7] = r_b;
        cols_b[7] = c_b_bytes;

        Self::new_custom(rows, cols_b)
    }
}

#[cfg(feature = "std")]
thread_local! {
    pub(crate) static ACTIVE_CONFIG: core::cell::Cell<Option<AmxConfig>> = core::cell::Cell::new(None);
    pub(crate) static SESSION_DEPTH: core::cell::Cell<usize> = core::cell::Cell::new(0);
}

#[cfg(not(feature = "std"))]
pub(crate) struct DummyThreadLocal<T> {
    cell: core::cell::Cell<T>,
}

#[cfg(not(feature = "std"))]
impl<T> DummyThreadLocal<T> {
    const fn new(val: T) -> Self {
        Self {
            cell: core::cell::Cell::new(val),
        }
    }
    #[inline(always)]
    fn with<R, F: FnOnce(&core::cell::Cell<T>) -> R>(&self, f: F) -> R {
        f(&self.cell)
    }
}

#[cfg(not(feature = "std"))]
unsafe impl<T> Sync for DummyThreadLocal<T> {}

#[cfg(not(feature = "std"))]
pub(crate) static ACTIVE_CONFIG: DummyThreadLocal<Option<AmxConfig>> = DummyThreadLocal::new(None);

#[cfg(not(feature = "std"))]
pub(crate) static SESSION_DEPTH: DummyThreadLocal<usize> = DummyThreadLocal::new(0);

/// A session guard that manages AMX tile configuration lifecycle on the current thread.
pub struct AmxSession {
    _private: (),
}

impl AmxSession {
    /// Returns true if an AMX session is currently active on the executing thread.
    #[inline]
    pub fn is_active() -> bool {
        ACTIVE_CONFIG.with(|c| c.get().is_some())
    }

    /// Enter a new AMX compute phase with the given configuration.
    #[inline]
    pub fn new(config: &AmxConfig) -> Self {
        let depth = SESSION_DEPTH.with(|d| {
            let val = d.get();
            d.set(val + 1);
            val
        });

        if depth == 0 {
            unsafe {
                raw::ldtilecfg(config);
            }
            ACTIVE_CONFIG.with(|c| c.set(Some(*config)));
        } else {
            let active = ACTIVE_CONFIG.with(|c| c.get());
            if active != Some(*config) {
                unsafe {
                    raw::ldtilecfg(config);
                }
                ACTIVE_CONFIG.with(|c| c.set(Some(*config)));
            }
        }
        Self { _private: () }
    }

    /// Context switch mitigation: release tile registers explicitly.
    #[inline]
    pub fn release() {
        unsafe {
            raw::tilerelease();
        }
        ACTIVE_CONFIG.with(|c| c.set(None));
        SESSION_DEPTH.with(|d| d.set(0));
    }
}

impl Drop for AmxSession {
    #[inline]
    fn drop(&mut self) {
        let depth = SESSION_DEPTH.with(|d| {
            let val = d.get();
            if val > 0 {
                d.set(val - 1);
                val - 1
            } else {
                0
            }
        });

        if depth == 0 {
            unsafe {
                raw::tilerelease();
            }
            ACTIVE_CONFIG.with(|c| c.set(None));
        }
    }
}

/// An RAII guard that encapsulates a complete AMX batch computation.
///
/// Automatically releases the AMX registers (`tilerelease()`) when dropped to prevent
/// context-switch penalties.
pub struct AmxBatchSession;

impl AmxBatchSession {
    /// Begin a new AMX batch computation.
    #[inline]
    pub fn begin(config: &AmxConfig) -> Self {
        unsafe {
            raw::ldtilecfg(config);
        }
        ACTIVE_CONFIG.with(|c| c.set(Some(*config)));
        Self
    }
}

impl Drop for AmxBatchSession {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            raw::tilerelease();
        }
        ACTIVE_CONFIG.with(|c| c.set(None));
        SESSION_DEPTH.with(|d| d.set(0));
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
            let _outer = AmxSession::new(&config);
            assert!(AmxSession::is_active());
            {
                let _inner = AmxSession::new(&config);
                assert!(AmxSession::is_active());
            }
            assert!(AmxSession::is_active());
        }

        assert!(!AmxSession::is_active());
    }

    #[test]
    fn miri_explicit_release_resets_session_state() {
        let config = AmxConfig::new_uniform(16, 64);
        let _session = AmxSession::new(&config);
        assert!(AmxSession::is_active());

        AmxSession::release();
        assert!(!AmxSession::is_active());
    }

    #[test]
    fn miri_batch_session_drop_resets_session_state() {
        let config = AmxConfig::new_uniform(16, 64);
        {
            let _session = AmxBatchSession::begin(&config);
            assert!(AmxSession::is_active());
        }
        assert!(!AmxSession::is_active());
    }
}
