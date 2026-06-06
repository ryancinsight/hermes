//! Execution-mode markers for SIMD operations.
//!
//! Two ZST markers define whether a [`crate::view::SimdView`] operates on all lanes (dense)
//! or a hardware-predicated subset (masked). The sealed `ExecutionMode` trait prevents
//! external implementations and enables the compiler to eliminate dead mode branches
//! via DCE during monomorphization.
//!
//! # Design
//! - `Unmasked` — default; maps to unconditional arithmetic. Zero overhead vs. not
//!   parameterizing by mode at all.
//! - `Masked` — activates predicated methods on `SimdView`. Maps to AVX-512 mask registers
//!   (`__mmask16`/`__mmask8`), AVX2 blend masks (`__m256`), SVE predicates, or scalar
//!   `[bool; N]` arrays depending on the bound `SimdKernel` implementation.

/// Private module that seals `ExecutionMode`.
mod sealed {
    pub trait Sealed {}
}

/// Marker trait for SIMD execution modes.
///
/// Sealed to prevent external implementations. Only `Unmasked` and `Masked` satisfy this.
pub trait ExecutionMode: sealed::Sealed + Send + Sync + 'static + Copy + Clone {
    /// Whether this mode requires mask operands on hot-path operations.
    const IS_MASKED: bool;
}

/// Dense execution — all lanes are active. Default mode for [`crate::view::SimdView`].
///
/// Monomorphization eliminates all masking overhead entirely; the compiler sees no
/// conditional paths through the `ExecutionMode` bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Unmasked;

/// Predicated execution — a hardware mask selects active lanes.
///
/// Enables the `dot_masked`, `sum_masked`, and `elementwise_add_masked` methods on
/// `SimdView`. Each method accepts an architecture-native `Arch::Mask` operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Masked;

impl sealed::Sealed for Unmasked {}
impl sealed::Sealed for Masked {}

impl ExecutionMode for Unmasked {
    const IS_MASKED: bool = false;
}

impl ExecutionMode for Masked {
    const IS_MASKED: bool = true;
}
