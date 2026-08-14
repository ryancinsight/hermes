//! Runtime probe establishing whether this process may execute Intel AMX tile
//! instructions.
//!
//! # Why CPUID is not enough
//!
//! AMX keeps its tile registers in XSAVE state components 17 (`XTILECFG`) and
//! 18 (`XTILEDATA`). Three independent conditions must hold before a tile
//! instruction can retire, and each one is checked by a different mechanism:
//!
//! 1. **The silicon implements AMX.** `CPUID.(EAX=7,ECX=0).EDX` bit 24
//!    (`amx-tile`) plus the per-kernel bits 22 (`amx-bf16`) and 25
//!    (`amx-int8`).
//! 2. **The OS saves and restores the state.** `XCR0` bits 17 and 18 must both
//!    be set, read through `XGETBV`. If they are clear, a tile instruction
//!    raises `#UD` regardless of what CPUID reports — this is the case on a
//!    kernel too old to know the state components exist.
//! 3. **This process holds permission.** `XTILEDATA` is an *XFD-gated*
//!    (extended-feature-disable) component: it is large (8 KiB), so operating
//!    systems arm `IA32_XFD` to trap first use rather than growing every
//!    process's signal frame. XCR0 advertises the component system-wide while
//!    XFD withholds it per thread, so condition 2 does **not** subsume this
//!    one. Executing a tile instruction without permission raises `#NM`.
//!
//! Condition 3 is what makes the probe platform-specific, and it is why a
//! CPUID-only check is unsound. Note that Rust's own
//! `is_x86_feature_detected!("amx-tile")` implements conditions 1 and 2 only
//! (and is unstable on the pinned toolchain regardless), so it cannot be used
//! here.
//!
//! # Conservative direction
//!
//! Every unresolved condition resolves to `false`. A false negative costs a
//! dispatch to the AVX-512 or scalar tile path; a false positive is a
//! `SIGILL`. Unknown operating systems therefore refuse rather than assume the
//! Linux or Windows model transfers.
//!
//! # Side effect
//!
//! Probing is not read-only. On both supported platforms the only way to learn
//! whether permission is obtainable is to request it, so the first call
//! performs a process-wide, irreversible opt-in that enlarges the XSAVE area
//! for every thread. The result is cached so this happens at most once.

/// `CPUID.(EAX=7,ECX=0).EDX` bit 22 — `amx-bf16` (`TDPBF16PS`).
const CPUID_EDX_AMX_BF16: u32 = 1 << 22;
/// `CPUID.(EAX=7,ECX=0).EDX` bit 24 — `amx-tile` (tile config/load/store).
const CPUID_EDX_AMX_TILE: u32 = 1 << 24;
/// `CPUID.(EAX=7,ECX=0).EDX` bit 25 — `amx-int8` (`TDPBSSD`).
const CPUID_EDX_AMX_INT8: u32 = 1 << 25;
/// `CPUID.(EAX=1).ECX` bit 27 — `OSXSAVE`, i.e. `CR4.OSXSAVE` is set and
/// `XGETBV` is therefore legal to execute.
const CPUID_ECX_OSXSAVE: u32 = 1 << 27;
/// `XCR0` bits 17 and 18 — `XTILECFG` and `XTILEDATA`.
const XCR0_AMX_STATE: u64 = (1 << 17) | (1 << 18);

/// Which AMX capabilities the executing process may actually use.
#[derive(Clone, Copy, Default)]
struct AmxAvailability {
    /// Tile configuration, load, and store are usable: the full chain holds.
    tile: bool,
    /// `TDPBF16PS` is usable (implies `tile`).
    bf16: bool,
    /// `TDPBSSD` is usable (implies `tile`).
    int8: bool,
}

impl AmxAvailability {
    /// Run the full detect chain exactly once.
    fn detect() -> Self {
        let unsupported = Self::default();

        // Miri interprets MIR and executes neither `CPUID` nor `XGETBV` (both
        // compile to inline asm), so the chain below cannot run under it. It is
        // also the honest answer: the interpreter is not a CPU with AMX. Note
        // this does not disable the session tests — `amx_runtime_supported`
        // overrides to `true` under Miri so the tile-configuration state
        // machine stays covered.
        if cfg!(miri) {
            return unsupported;
        }

        // Leaf 7 must exist before it can be queried; on a CPU whose maximum
        // basic leaf is below 7 the instruction returns the highest supported
        // leaf's data instead, which would alias into a false positive.
        if core::arch::x86_64::__get_cpuid_max(0).0 < 7 {
            return unsupported;
        }

        let leaf1 = core::arch::x86_64::__cpuid(1);
        let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);

        if leaf7.edx & CPUID_EDX_AMX_TILE == 0 {
            return unsupported;
        }

        // XGETBV raises #UD unless CR4.OSXSAVE is set, which OSXSAVE reports.
        if leaf1.ecx & CPUID_ECX_OSXSAVE == 0 {
            return unsupported;
        }
        if xcr0() & XCR0_AMX_STATE != XCR0_AMX_STATE {
            return unsupported;
        }

        if !request_tile_data_permission() {
            return unsupported;
        }

        Self {
            tile: true,
            bf16: leaf7.edx & CPUID_EDX_AMX_BF16 != 0,
            int8: leaf7.edx & CPUID_EDX_AMX_INT8 != 0,
        }
    }
}

/// Read `XCR0` (extended control register 0).
///
/// The caller must have confirmed `CPUID.(EAX=1).ECX[27]` (`OSXSAVE`) first;
/// `XGETBV` raises `#UD` when `CR4.OSXSAVE` is clear.
#[inline]
fn xcr0() -> u64 {
    let (low, high): (u32, u32);
    // SAFETY: the caller verified OSXSAVE, so XGETBV is legal here. ECX=0
    // selects XCR0, the only index this reads.
    //
    // Options: XGETBV reads a control register and writes only EDX:EAX, so it
    // touches no memory (`nomem`) and no stack (`nostack`); it is documented to
    // leave RFLAGS untouched (`preserves_flags`).
    unsafe {
        core::arch::asm!(
            "xgetbv",
            in("ecx") 0_u32,
            out("eax") low,
            out("edx") high,
            options(nomem, nostack, preserves_flags),
        );
    }
    (u64::from(high) << 32) | u64::from(low)
}

/// Obtain permission to use the `XTILEDATA` state component, returning whether
/// this process may now execute tile instructions.
///
/// Both implementations verify the grant by re-reading the permission state
/// rather than trusting the request's return value alone.
#[cfg(target_os = "linux")]
fn request_tile_data_permission() -> bool {
    /// `arch_prctl` syscall number on `x86_64` Linux.
    const SYS_ARCH_PRCTL: i64 = 158;
    /// `ARCH_GET_XCOMP_SUPP` — mask of dynamic features the kernel supports.
    const ARCH_GET_XCOMP_SUPP: u64 = 0x1021;
    /// `ARCH_GET_XCOMP_PERM` — mask of features this process has been granted.
    const ARCH_GET_XCOMP_PERM: u64 = 0x1022;
    /// `ARCH_REQ_XCOMP_PERM` — request one dynamic feature by state index.
    const ARCH_REQ_XCOMP_PERM: u64 = 0x1023;
    /// `XFEATURE_XTILEDATA`, the state-component index AMX tile data occupies.
    const XFEATURE_XTILEDATA: u64 = 18;
    /// The same component expressed as a mask, for the two query results.
    const XFEATURE_MASK_XTILEDATA: u64 = 1 << XFEATURE_XTILEDATA;

    /// Issue `arch_prctl(code, arg)`, returning the raw kernel result
    /// (`0` on success, negative errno on failure).
    ///
    /// # Safety
    /// For the `ARCH_GET_*` codes `arg` must be a valid, writable `*mut u64`.
    /// For `ARCH_REQ_XCOMP_PERM` it is a by-value state index.
    #[inline]
    unsafe fn arch_prctl(code: u64, arg: u64) -> i64 {
        let result: i64;
        // SAFETY: the caller guarantees `arg` matches `code`'s contract. The
        // syscall ABI is rax=number, rdi/rsi=arguments, returning in rax and
        // clobbering rcx and r11, all declared below.
        //
        // Options: `nostack` only — the kernel runs on its own stack and never
        // touches the user red zone. `nomem` is deliberately NOT claimed: the
        // ARCH_GET_* codes have the kernel write through `arg`, so the compiler
        // must keep assuming memory is clobbered.
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") SYS_ARCH_PRCTL => result,
                in("rdi") code,
                in("rsi") arg,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
        }
        result
    }

    /// Read one of the two dynamic-feature masks, or `None` if the kernel
    /// rejected the query (a pre-5.16 kernel has no such code).
    fn query_mask(code: u64) -> Option<u64> {
        let mut mask: u64 = 0;
        // SAFETY: `code` is a caller-selected ARCH_GET_* code and the argument
        // is a pointer to a live, writable local `u64`.
        let result = unsafe { arch_prctl(code, core::ptr::addr_of_mut!(mask) as u64) };
        (result == 0).then_some(mask)
    }

    // Ask what the kernel can offer before requesting: on a host without AMX
    // the request would fail with EOPNOTSUPP anyway, and querying first keeps
    // the failure attributable.
    let Some(supported) = query_mask(ARCH_GET_XCOMP_SUPP) else {
        return false;
    };
    if supported & XFEATURE_MASK_XTILEDATA == 0 {
        return false;
    }

    // SAFETY: ARCH_REQ_XCOMP_PERM takes the state index by value, not a
    // pointer, so there is no memory for the kernel to write through.
    if unsafe { arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) } != 0 {
        return false;
    }

    // Confirm the grant landed rather than trusting the return code alone.
    query_mask(ARCH_GET_XCOMP_PERM).is_some_and(|granted| granted & XFEATURE_MASK_XTILEDATA != 0)
}

/// Obtain permission to use the `XTILEDATA` state component on Windows.
///
/// Windows models AMX tile data as an *optional `XState` feature*: threads start
/// with it XFD-disabled and a process opts in through
/// `EnableProcessOptionalXStateFeatures`, which covers existing and future
/// threads alike. That entry point and its per-thread query counterpart arrived
/// in Windows 11 / Server 2022 and are absent from Windows 10, so they are
/// resolved dynamically — a static import would make the whole binary fail to
/// load on an older host.
#[cfg(target_os = "windows")]
fn request_tile_data_permission() -> bool {
    use core::ffi::c_void;

    /// `XSTATE_MASK_AMX_TILE_CONFIG | XSTATE_MASK_AMX_TILE_DATA` from `winnt.h`.
    const XSTATE_MASK_AMX: u64 = (1 << 17) | (1 << 18);
    /// `XSTATE_MASK_AMX_TILE_DATA` alone — the opt-in operates on the data
    /// component, the only XFD-gated one.
    const XSTATE_MASK_AMX_TILE_DATA: u64 = 1 << 18;

    /// `EnableProcessOptionalXStateFeatures` — returns a Win32 `BOOL`.
    type EnableOptionalFeatures = unsafe extern "system" fn(u64) -> i32;
    /// `GetThreadEnabledXStateFeatures` — mask enabled for the calling thread.
    type ThreadEnabledFeatures = unsafe extern "system" fn() -> u64;

    #[link(name = "kernel32")]
    extern "system" {
        /// Mask of `XState` features the *system* has enabled. Present since
        /// Windows 7, so this one may be imported statically.
        fn GetEnabledXStateFeatures() -> u64;
        fn GetModuleHandleA(module_name: *const u8) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, proc_name: *const u8) -> *mut c_void;
    }

    // The system mask must already carry both components. Requesting a feature
    // the system does not support fails with ERROR_INVALID_PARAMETER, so this
    // check keeps the opt-in attributable and matches the documented contract.
    // SAFETY: no arguments, no pointers; available on every supported Windows.
    if unsafe { GetEnabledXStateFeatures() } & XSTATE_MASK_AMX != XSTATE_MASK_AMX {
        return false;
    }

    // SAFETY: `kernel32.dll` is loaded into every Win32 process, and the name
    // is a NUL-terminated literal. A null return is handled below.
    let kernel32 = unsafe { GetModuleHandleA(c"kernel32.dll".as_ptr().cast()) };
    if kernel32.is_null() {
        return false;
    }

    // SAFETY: `kernel32` is a live module handle and both names are
    // NUL-terminated literals. Absent exports return null, handled below.
    let enable = unsafe {
        GetProcAddress(
            kernel32,
            c"EnableProcessOptionalXStateFeatures".as_ptr().cast(),
        )
    };
    // SAFETY: as above.
    let thread_enabled =
        unsafe { GetProcAddress(kernel32, c"GetThreadEnabledXStateFeatures".as_ptr().cast()) };
    if enable.is_null() || thread_enabled.is_null() {
        // Pre-Windows-11: no optional-XState opt-in exists, so neither does
        // usable AMX.
        return false;
    }

    // SAFETY: both pointers are non-null exports of kernel32 resolved by their
    // documented names, transmuted to the signatures Microsoft documents for
    // them. Calling `enable` for a system-supported feature is defined, and
    // `thread_enabled` takes no arguments.
    unsafe {
        let enable = core::mem::transmute::<*mut c_void, EnableOptionalFeatures>(enable);
        let thread_enabled =
            core::mem::transmute::<*mut c_void, ThreadEnabledFeatures>(thread_enabled);

        if enable(XSTATE_MASK_AMX_TILE_DATA) == 0 {
            return false;
        }
        // Verify the calling thread really carries the component now.
        thread_enabled() & XSTATE_MASK_AMX_TILE_DATA != 0
    }
}

/// Refuse on every other target.
///
/// AMX permission is an OS-specific protocol, and no other platform that runs
/// on AMX-capable silicon has a known one. Guessing here would trade a missed
/// optimization for a `SIGILL`.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn request_tile_data_permission() -> bool {
    false
}

/// Cached probe state: `0` not yet run, otherwise `1 | (tile << 1) |
/// (bf16 << 2) | (int8 << 3)`.
static CACHED: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Bit marking the cache as populated.
const CACHE_READY: u8 = 1 << 0;
/// Bit carrying [`AmxAvailability::tile`].
const CACHE_TILE: u8 = 1 << 1;
/// Bit carrying [`AmxAvailability::bf16`].
const CACHE_BF16: u8 = 1 << 2;
/// Bit carrying [`AmxAvailability::int8`].
const CACHE_INT8: u8 = 1 << 3;

/// Return the cached availability, running the detect chain on first use.
///
/// A race between two first callers is benign: both run the chain, the
/// permission request is idempotent, and both store the same value.
#[inline]
fn availability() -> AmxAvailability {
    use core::sync::atomic::Ordering;

    let cached = CACHED.load(Ordering::Relaxed);
    let bits = if cached & CACHE_READY == 0 {
        let detected = AmxAvailability::detect();
        let bits = CACHE_READY
            | if detected.tile { CACHE_TILE } else { 0 }
            | if detected.bf16 { CACHE_BF16 } else { 0 }
            | if detected.int8 { CACHE_INT8 } else { 0 };
        CACHED.store(bits, Ordering::Relaxed);
        bits
    } else {
        cached
    };

    AmxAvailability {
        tile: bits & CACHE_TILE != 0,
        bf16: bits & CACHE_BF16 != 0,
        int8: bits & CACHE_INT8 != 0,
    }
}

/// Returns whether this process may configure and use AMX tile registers.
///
/// True only when the silicon reports `amx-tile`, the OS has enabled the
/// `XTILECFG`/`XTILEDATA` state components in `XCR0`, and this process has been
/// granted the XFD-gated tile-data component. See the module documentation for
/// why all three are required and for the one-time opt-in side effect.
#[inline]
#[must_use]
pub fn has_amx_tile() -> bool {
    availability().tile
}

/// Returns whether this process may execute `TDPBF16PS` (AMX BF16 tile GEMM).
///
/// Implies [`has_amx_tile`].
#[inline]
#[must_use]
pub fn has_amx_bf16() -> bool {
    availability().bf16
}

/// Returns whether this process may execute `TDPBSSD` (AMX INT8 tile GEMM).
///
/// Implies [`has_amx_tile`].
#[inline]
#[must_use]
pub fn has_amx_int8() -> bool {
    availability().int8
}

#[cfg(test)]
mod tests {
    use super::{has_amx_bf16, has_amx_int8, has_amx_tile};

    /// The per-kernel capabilities are refinements of the tile capability, so
    /// neither can be reported without it. This holds on every host: on one
    /// without AMX all three are false, and the implication is vacuous.
    #[test]
    fn kernel_capabilities_imply_tile_capability() {
        if has_amx_bf16() || has_amx_int8() {
            assert!(
                has_amx_tile(),
                "a tile GEMM kernel was reported usable without tile configuration"
            );
        }
    }

    /// The probe caches its result, so repeated calls must agree — a differing
    /// answer would mean the one-shot permission request had run twice with
    /// different outcomes.
    #[test]
    fn probe_is_stable_across_calls() {
        assert_eq!(has_amx_tile(), has_amx_tile());
        assert_eq!(has_amx_bf16(), has_amx_bf16());
        assert_eq!(has_amx_int8(), has_amx_int8());
    }

    /// AMX exists only on `x86_64`, and only Linux and Windows expose a
    /// permission protocol for it. Everywhere else the probe must refuse.
    #[test]
    fn unsupported_platforms_refuse() {
        if !cfg!(any(target_os = "linux", target_os = "windows")) {
            assert!(
                !has_amx_tile(),
                "AMX reported usable on a platform with no known permission protocol"
            );
        }
    }
}
