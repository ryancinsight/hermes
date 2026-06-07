//! Zero-cost operation strategy markers for SIMD reductions and elementwise transforms.
//!
//! `ReductionOp<T>`, `ElementOp<T>`, and `UnaryOp<T>` are sealed ZST traits parameterized
//! by the scalar type `T: Scalar`. Concrete strategies (`Sum`, `Dot`, `Mul`, `Add`, `Sub`,
//! `Abs`, `Neg`, `Sqrt`) implement these traits and are passed as ZST values — they carry no
//! runtime data and the compiler eliminates all abstraction overhead via monomorphization.
//!
//! # Usage
//!
//! ```rust,ignore
//! let total: f32 = view.reduce(ops::Sum);
//! let dot: f32 = view.zip_reduce(&other, ops::Dot)?;
//! ```
//!
//! # Zero-Cost Guarantee
//!
//! Each `unsafe fn accumulate` / `unsafe fn apply` call site is a direct call to
//! an `#[inline(always)]` function that the compiler inlines into the surrounding loop.
//! The ZST parameter is erased entirely — `size_of::<Sum>() == 0`.
//!
//! # Scalar Tail Handling
//!
//! `ElementOp<T>` provides `apply_scalar(a, b) -> T` for processing tail elements that
//! do not fill a complete SIMD vector. This is a pure scalar operation using `T: Scalar`
//! arithmetic operators, eliminating all boundary-condition UB from vector load/store.

use crate::{
    kernel::SimdKernel,
    scalar::{Scalar, NumericElement},
};

// ---------------------------------------------------------------------------
// Sealing
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ReductionOp — single-operand fold across lanes
// ---------------------------------------------------------------------------

/// Sealed ZST trait for SIMD horizontal reduction strategies.
///
/// Implementors define how a vector accumulator is updated (`accumulate`) and how
/// the final scalar result is extracted (`finalize`). Both methods are `#[inline(always)]`
/// and carry no branching — DCE eliminates unused strategies entirely.
///
/// # Identity Element
///
/// `identity_scalar()` returns the reduction identity (0 for Sum, `T::MAX_VALUE` for Min,
/// `T::MIN_VALUE` for Max). It is used for empty-slice fast paths and for combining the
/// scalar tail with the SIMD result via `scalar_combine`.
pub trait ReductionOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Merge a new data vector `v` into accumulator `acc`.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector;

    /// Reduce the final accumulator to a scalar.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T;

    /// The identity element for this reduction as a scalar.
    ///
    /// Default: must be overridden. `Sum` returns `T::ZERO`, `Min` returns `T::MAX_VALUE`,
    /// `Max` returns `T::MIN_VALUE`.
    fn identity_scalar() -> T;

    /// Combine two scalar partial results using this reduction.
    ///
    /// For `Sum`: addition. For `Min`: `min_scalar`. For `Max`: `max_scalar`.
    fn scalar_combine(a: T, b: T) -> T;

    /// Splat the identity element into a vector register.
    ///
    /// Default: `Arch::splat(Self::identity_scalar())`. Backends may override.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn identity_vector<Arch: SimdKernel<T>>() -> Arch::Vector {
        Arch::splat(Self::identity_scalar())
    }
}

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
// Concrete strategy ZSTs
// ---------------------------------------------------------------------------

/// Sum reduction: accumulate by adding vectors, finalize with `sum_reduce`.
///
/// `view.reduce(Sum)` is equivalent to `view.sum()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sum;

/// Dot-product pairwise operation: multiply two vectors lane-wise.
///
/// Use with `zip_reduce`: `a.zip_reduce(&b, Dot)` equals `a.dot(&b)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dot;

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

/// Horizontal minimum reduction: returns the smallest element.
///
/// Identity element: `T::MAX_VALUE` (positive infinity for floats, `i32::MAX` for integers).
/// Use with `view.reduce(Min)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Min;

/// Horizontal maximum reduction: returns the largest element.
///
/// Identity element: `T::MIN_VALUE` (negative infinity for floats, `i32::MIN` for integers).
/// Use with `view.reduce(Max)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Max;

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
// UnaryOp — single-operand elementwise strategy
// ---------------------------------------------------------------------------

/// Sealed ZST trait for single-operand SIMD elementwise operations.
///
/// Implementors define how a single vector is transformed (`apply`) and how
/// a single scalar element is transformed (`apply_scalar`). Both paths are
/// `#[inline(always)]` — DCE eliminates unused strategies entirely.
///
/// # Zero-Cost Guarantee
///
/// Every `impl UnaryOp<T>` passes through to an `#[inline(always)]
/// SimdKernel<T>` method. The ZST strategy parameter is erased at every
/// monomorphization site: `size_of::<Abs>() == 0`.
pub trait UnaryOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Apply the operation to a vector: `self.apply::<Arch>(v) -> result`.
    ///
    /// Takes `self` by value so `Clamp<T>` can access its bounds; for true ZST
    /// strategies (`Abs`, `Neg`, `Sqrt`), `self` has size zero and the compiler
    /// removes it entirely from the generated code.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector;

    /// Apply the operation to a single scalar element.
    ///
    /// Used for the SIMD tail (elements that do not fill a complete vector).
    /// Requires only `T: Scalar` — no unsafe, no vector loads or stores.
    fn apply_scalar(self, a: T) -> T;
}

/// Elementwise absolute value: `|a[i]|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Abs;

/// Elementwise negation: `-a[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Neg;

/// Elementwise square root: `sqrt(a[i])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sqrt;

// ---------------------------------------------------------------------------
// Sealing impls
// ---------------------------------------------------------------------------

impl crate::private::Sealed for Sum {}
impl crate::private::Sealed for Dot {}
impl crate::private::Sealed for Mul {}
impl crate::private::Sealed for Add {}
impl crate::private::Sealed for Sub {}
impl crate::private::Sealed for Div {}
impl crate::private::Sealed for BitAnd {}
impl crate::private::Sealed for BitOr {}
impl crate::private::Sealed for BitXor {}
impl crate::private::Sealed for Min {}
impl crate::private::Sealed for Max {}
impl crate::private::Sealed for Abs {}
impl crate::private::Sealed for Neg {}
impl crate::private::Sealed for Sqrt {}
impl<T: Copy + 'static> crate::private::Sealed for Clamp<T> {}

// ---------------------------------------------------------------------------
// ReductionOp impls
// ---------------------------------------------------------------------------

impl<T: Scalar> ReductionOp<T> for Sum {
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        Arch::add(acc, v)
    }
    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::sum_reduce(acc)
    }
    #[inline(always)]
    fn identity_scalar() -> T { T::ZERO }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T { a + b }
}

impl<T: Scalar> ReductionOp<T> for Dot {
    /// Dot accumulation: `acc = fmadd(a, b, acc)` — called with the pairwise product vector.
    ///
    /// The `zip_reduce` loop computes `v = mul(a_chunk, b_chunk)` then calls `accumulate(acc, v)`.
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        // v already holds a[i]*b[i] product from the zip loop; just add to accumulator.
        Arch::add(acc, v)
    }
    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::sum_reduce(acc)
    }
    #[inline(always)]
    fn identity_scalar() -> T { T::ZERO }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T { a + b }
}

impl<T: Scalar> ReductionOp<T> for Min {
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        Arch::min(acc, v)
    }
    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::min_reduce(acc)
    }
    #[inline(always)]
    fn identity_scalar() -> T { T::MAX_VALUE }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T { a.min_scalar(b) }
}

impl<T: Scalar> ReductionOp<T> for Max {
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        Arch::max(acc, v)
    }
    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::max_reduce(acc)
    }
    #[inline(always)]
    fn identity_scalar() -> T { T::MIN_VALUE }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T { a.max_scalar(b) }
}

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

// ---------------------------------------------------------------------------
// UnaryOp impls
// ---------------------------------------------------------------------------

impl<T: Scalar> UnaryOp<T> for Abs {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        Arch::abs(v)
    }

    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        a.abs()
    }
}

impl<T: Scalar> UnaryOp<T> for Neg {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        Arch::neg(v)
    }

    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        T::ZERO - a
    }
}

impl<T: Scalar> UnaryOp<T> for Sqrt {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        Arch::sqrt(v)
    }

    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        a.sqrt()
    }
}

impl<T: Scalar + PartialOrd + NumericElement> UnaryOp<T> for Clamp<T> {
    #[inline(always)]
    unsafe fn apply<Arch: SimdKernel<T>>(self, v: Arch::Vector) -> Arch::Vector {
        // clamp(v, lo, hi) = max(lo, min(v, hi))
        let lo_vec = Arch::splat(self.lo);
        let hi_vec = Arch::splat(self.hi);
        let clamped_hi = Arch::min(v, hi_vec);
        Arch::max(clamped_hi, lo_vec)
    }

    #[inline(always)]
    fn apply_scalar(self, a: T) -> T {
        a.min_scalar(self.hi).max_scalar(self.lo)
    }
}

// ---------------------------------------------------------------------------
// ScanOp — associative binary prefix-scan operation
// ---------------------------------------------------------------------------

/// Sealed ZST trait for prefix scan operations.
pub trait ScanOp<T: Scalar>: crate::private::Sealed + Copy + 'static {
    /// Returns the identity element of the operation.
    fn identity() -> T;
    /// Combine two values using the operation: `a op b`.
    fn combine(a: T, b: T) -> T;
}

/// Addition scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanAdd;

/// Multiplication scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanMul;

/// Minimum scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanMin;

/// Maximum scan strategy ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanMax;

impl crate::private::Sealed for ScanAdd {}
impl crate::private::Sealed for ScanMul {}
impl crate::private::Sealed for ScanMin {}
impl crate::private::Sealed for ScanMax {}

impl<T: Scalar> ScanOp<T> for ScanAdd {
    #[inline(always)]
    fn identity() -> T { T::ZERO }
    #[inline(always)]
    fn combine(a: T, b: T) -> T { a + b }
}

impl<T: Scalar> ScanOp<T> for ScanMul {
    #[inline(always)]
    fn identity() -> T { T::ONE }
    #[inline(always)]
    fn combine(a: T, b: T) -> T { a * b }
}

impl<T: Scalar> ScanOp<T> for ScanMin {
    #[inline(always)]
    fn identity() -> T { T::MAX_VALUE }
    #[inline(always)]
    fn combine(a: T, b: T) -> T { a.min_scalar(b) }
}

impl<T: Scalar> ScanOp<T> for ScanMax {
    #[inline(always)]
    fn identity() -> T { T::MIN_VALUE }
    #[inline(always)]
    fn combine(a: T, b: T) -> T { a.max_scalar(b) }
}

// ---------------------------------------------------------------------------
// ScanMode — prefix-scan inclusion mode
// ---------------------------------------------------------------------------

/// Sealed ZST trait for prefix scan inclusion modes.
pub trait ScanMode: crate::private::Sealed + Copy + 'static {
    /// Whether the scan is inclusive of the current element.
    const IS_INCLUSIVE: bool;
}

/// Inclusive scan ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inclusive;

/// Exclusive scan ZST marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exclusive;

impl crate::private::Sealed for Inclusive {}
impl crate::private::Sealed for Exclusive {}

impl ScanMode for Inclusive {
    const IS_INCLUSIVE: bool = true;
}

impl ScanMode for Exclusive {
    const IS_INCLUSIVE: bool = false;
}