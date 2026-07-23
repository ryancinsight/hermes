//! Target architecture definition trait with architecture-level constants.
//!
//! `SimdArch` is a sealed-by-convention trait implemented by ZST markers.
//! Constants here are architecture-wide and do not require a scalar type `T` binding,
//! unlike `SimdKernel<T>` which is type-parameterized.

/// ISA family classification for architecture markers.
///
/// Used for compile-time routing and documentation. Does not affect code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsaFamily {
    /// x86 and x86_64 (SSE, AVX, AVX-512 families).
    X86,
    /// AArch64 (NEON, SVE, SME families).
    AArch64,
    /// RISC-V (V extension).
    RiscV,
    /// No vector ISA; pure scalar fallback.
    Scalar,
    /// Other or experimental architecture.
    Other,
}

/// Trait representing a SIMD instruction set architecture.
///
/// Implemented by Zero-Sized Types (ZSTs). Constants are queryable without a scalar
/// type parameter — contrast with `SimdKernel<T>` which requires `T` to be known.
pub trait SimdArch: crate::private::Sealed + Send + Sync + 'static + Copy + Clone {
    /// Human-readable ISA name (`"avx2"`, `"avx512"`, `"neon"`, `"scalar"`).
    const NAME: &'static str;

    /// Width of vector registers in bits.
    ///
    /// | Architecture | Width |
    /// |---|---|
    /// | `Scalar` | `0` |
    /// | `Neon` | `128` |
    /// | `Avx2` | `256` |
    /// | `Avx512` | `512` |
    const REGISTER_WIDTH_BITS: u32;

    /// ISA family for this architecture.
    const ISA_FAMILY: IsaFamily;

    /// Suggested `TILE_M` value for `tiled_dot` to saturate FMA throughput.
    ///
    /// This is a hint, not a constraint. Optimal values:
    /// - `Scalar`: 1 (no tiling benefit)
    /// - `Neon`: 4 (4 NEON regs × 4-cycle FMA latency)
    /// - `Avx2`: 4 (16 YMM / TILE_M=4 leaves headroom for loop overhead)
    /// - `Avx512`: 8 (32 ZMM / TILE_M=8 saturates two FMA ports)
    const FMA_THROUGHPUT_HINT: u32;

    /// Returns true when the current host may execute this architecture's
    /// native instructions from safe wrappers.
    ///
    /// Emulated backends return `true`; native ISA backends must include the
    /// OS-enabled register-state checks covered by the platform feature probe.
    fn is_runtime_supported() -> bool;
}

/// Panics unless `Arch` can execute on this host.
///
/// Kernel methods are `#[target_feature]`-gated, so invoking one on a host
/// lacking those features is undefined behavior. Types parameterized by `Arch`
/// call this where they are built, which turns "the host supports `Arch`" into
/// an invariant of holding the value — the discharge every `unsafe` kernel call
/// downstream relies on. Constructors that can report failure return `None`
/// instead of calling this.
///
/// The probe caches its CPUID result, so repeated calls are a relaxed load.
///
/// # Panics
/// If `Arch::is_runtime_supported()` is false.
#[inline]
pub(crate) fn assert_arch_executable<Arch: SimdArch>() {
    assert!(
        Arch::is_runtime_supported(),
        "SIMD target {} is not supported or enabled on this host",
        Arch::NAME
    );
}
