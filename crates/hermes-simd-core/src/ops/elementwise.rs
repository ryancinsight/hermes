//! Pairwise lane-wise elementwise operation strategies.
//!
//! `ElementOp<T>` is a sealed ZST trait used by `zip_reduce`, `zip_cow`, and
//! `transform_in_place` to parameterize binary vector operations without code
//! duplication. All `apply` impls are `#[inline(always)]` — DCE removes unused strategies.

use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

// ---------------------------------------------------------------------------
// ElementOp — pairwise lane-wise operation
// ---------------------------------------------------------------------------

/// Sealed ZST trait for pairwise SIMD elementwise operations.
///
/// Used by `zip_reduce`, `zip_cow`, and `transform_in_place` to parameterize binary
/// vector operations without code duplication.
///
/// # Scalar Tail
///
/// `apply_scalar(a, b)` handles elements that do not fill a complete SIMD vector.
/// Implementations use direct `T: Scalar` arithmetic so no vector load/store
/// boundary conditions apply. The default does NOT exist — every impl must provide
/// both `apply` (vector) and `apply_scalar` (scalar element).
pub trait ElementOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Apply the operation to two vectors lane-wise.
    ///
    /// Takes `self` by value — for ZSTs this is free; for `Clamp` it captures the bounds.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector;

    /// Apply the operation to two individual scalar elements.
    ///
    /// Used for the SIMD tail (elements that do not fill a complete vector).
    /// Takes `self` by value — for ZSTs this is free; for `Clamp` it captures the bounds.
    fn apply_scalar(self, a: T, b: T) -> T;
}

// ---------------------------------------------------------------------------
// Concrete elementwise ZSTs
// ---------------------------------------------------------------------------

/// Elementwise multiplication: `a[i] * b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mul;

/// Elementwise addition: `a[i] + b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Add;

/// Elementwise subtraction: `a[i] - b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sub;

/// Elementwise division: `a[i] / b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Div;

/// Elementwise bitwise AND: `a[i] & b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitAnd;

/// Elementwise bitwise OR: `a[i] | b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitOr;

/// Elementwise bitwise XOR: `a[i] ^ b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitXor;

/// Fused-multiply-add elementwise operation: `out[i] = a[i] * b[i] + a[i]` (binary form).
///
/// As an `ElementOp`, this interprets the two operand vectors as `a` and `b`, and computes
/// `fmadd(a, b, zero)` — i.e. `a * b` with hardware FMA precision, accumulating into zero.
/// To use as a ternary `a*b + c` accumulation, call `Arch::fmadd` directly.
///
/// # Zero-Cost Guarantee
///
/// `size_of::<FmaAdd>() == 0`. Monomorphization over `Arch` eliminates the ZST entirely;
/// the call site reduces to a direct `Arch::fmadd` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmaAdd;

/// Elementwise clamp: `min(max(a[i], lo), hi)`.
///
/// Unlike the other strategies, `Clamp` carries its bounds as value fields — the
/// bounds are `T`-typed and cannot be encoded as const generics for a generic `T`.
/// The struct is `Copy` and 2×`size_of::<T>()` bytes; the compiler monomorphizes per
/// `(T, Arch)` pair.
///
/// # Usage
///
/// Elementwise clamp: `a[i].clamp(lo, hi)`.
///
/// For binary elementwise use via `zip_cow`, the second operand is ignored —
/// bounds are carried in the struct and broadcast-splat at each SIMD iteration.
/// For unary use via `map_unary`, pass `Clamp { lo, hi }` directly.
///
/// Inlined by the optimizer at each monomorphization site: `Clamp<T>` bounds
/// become register-constant splat values with no indirect reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clamp<T: Copy> {
    /// Lower bound (inclusive).
    pub lo: T,
    /// Upper bound (inclusive).
    pub hi: T,
}

impl<T: Copy> Clamp<T> {
    /// Construct a new `Clamp` strategy with the given bounds.
    #[inline(always)]
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }
}

// ---------------------------------------------------------------------------
// Sealing impls
// ---------------------------------------------------------------------------

impl crate::private::Sealed for Mul {}
impl crate::private::Sealed for Add {}
impl crate::private::Sealed for Sub {}
impl crate::private::Sealed for Div {}
impl crate::private::Sealed for BitAnd {}
impl crate::private::Sealed for BitOr {}
impl crate::private::Sealed for BitXor {}
impl crate::private::Sealed for FmaAdd {}
impl<T: Copy + 'static> crate::private::Sealed for Clamp<T> {}

// ---------------------------------------------------------------------------
// ElementOp impls
// ---------------------------------------------------------------------------

impl<T: Scalar> ElementOp<T> for Mul {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::mul(a, b)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        a * b
    }
}

impl<T: Scalar> ElementOp<T> for Add {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::add(a, b)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        a + b
    }
}

impl<T: Scalar> ElementOp<T> for Sub {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::sub(a, b)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        a - b
    }
}

impl<T: Scalar> ElementOp<T> for Div {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::div(a, b)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        a / b
    }
}

impl<T: Scalar> ElementOp<T> for BitAnd {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::bitand(a, b)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        a.bitand(b)
    }
}

impl<T: Scalar> ElementOp<T> for BitOr {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::bitor(a, b)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        a.bitor(b)
    }
}

impl<T: Scalar> ElementOp<T> for BitXor {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        Arch::bitxor(a, b)
    }
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        a.bitxor(b)
    }
}

impl<T: Scalar> ElementOp<T> for FmaAdd {
    /// `fmadd(a, b, zero)` — uses hardware FMA where available.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, b: Arch::Vector) -> Arch::Vector {
        // Accumulate into a zero register: a[i] * b[i] + 0.
        let zero = Arch::zero();
        Arch::fmadd(a, b, zero)
    }

    /// Scalar tail: `a * b` (scalar `mul`; the addend is the implicit zero).
    #[inline(always)]
    fn apply_scalar(self, a: T, b: T) -> T {
        // scalar_fmadd(a, b, 0) — uses T's scalar FMA implementation.
        a.scalar_fmadd(b, T::ZERO)
    }
}

impl<T: Scalar + Copy> ElementOp<T> for Clamp<T> {
    /// Clamp lanes: `min(max(a_lane, lo_splat), hi_splat)`.
    ///
    /// The second operand `_b` is unused — `Clamp` is a unary operation whose
    /// bounds are carried in the struct. Use `clamp_cow` or `transform_with_clamp`
    /// to apply this as a true single-operand transform.
    ///
    /// The compiler hoists the `splat(lo)` and `splat(hi)` outside the vectorized
    /// loop because `self` is captured by value in each `zip_cow` iteration.
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, a: Arch::Vector, _b: Arch::Vector) -> Arch::Vector {
        let lo_vec = Arch::splat(self.lo);
        let hi_vec = Arch::splat(self.hi);
        Arch::min(Arch::max(a, lo_vec), hi_vec)
    }

    #[inline(always)]
    fn apply_scalar(self, a: T, _b: T) -> T {
        // min(max(a, lo), hi) — scalar path for the vector tail.
        a.max_scalar(self.lo).min_scalar(self.hi)
    }
}
