//! Horizontal reduction operation strategies.
//!
//! `ReductionOp<T>` is a sealed ZST trait; implementors define how a vector accumulator
//! is updated and how the final scalar is extracted. All methods are `#[inline(always)]`
//! and carry no branching — DCE eliminates unused strategies entirely.

use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

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

    /// Accumulate a single scalar element `elem` into a scalar accumulator `acc`.
    ///
    /// Default: delegates to `Self::scalar_combine(acc, elem)`.
    /// Override this for reductions whose SIMD `accumulate` applies a per-element transform
    /// (e.g. `SquaredSum` applies `elem * elem` before adding). The scalar tail path uses this
    /// method instead of `scalar_combine` to maintain correctness for slices shorter than
    /// `Arch::LANE_COUNT`.
    #[inline(always)]
    fn scalar_accumulate(acc: T, elem: T) -> T {
        Self::scalar_combine(acc, elem)
    }

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

    /// Per-element lane transform applied before combining (identity by default).
    ///
    /// Reductions with a per-element transform (`AbsSum` applies `abs`) override
    /// this so the reduce loop can seed unrolled accumulators with
    /// `transform_vector(load(...))` instead of raw loads.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn transform_vector<Arch: SimdKernel<T>>(v: Arch::Vector) -> Arch::Vector {
        v
    }

    /// Merge two partial accumulators WITHOUT the per-element transform.
    ///
    /// `accumulate` is `combine_vectors(acc, transform_vector(v))`; the reduce
    /// loop's cross-accumulator merge must use this method, because the
    /// partials are already transformed. Default delegates to `accumulate`,
    /// which is correct exactly when `transform_vector` is the identity —
    /// transform-bearing ops must override both.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn combine_vectors<Arch: SimdKernel<T>>(
        a: Arch::Vector,
        b: Arch::Vector,
    ) -> Arch::Vector {
        Self::accumulate::<Arch>(a, b)
    }
}

// ---------------------------------------------------------------------------
// Concrete reduction ZSTs
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

/// Absolute-sum reduction: computes `Σ |data[i]|` (the L1 norm accumulator).
///
/// Identity element is `T::ZERO`. The per-element transform is `abs`, applied
/// lane-wise before the additive fold (`scalar_accumulate` mirrors it on the
/// tail). Signed-integer `abs` follows `T::abs` semantics, including its
/// behavior at `T::MIN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsSum;

/// Absolute-max reduction: computes `max |data[i]|` (the ∞-norm accumulator).
///
/// Identity element is `T::ZERO`, which is also the mathematically correct
/// result for an empty slice since every magnitude is non-negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsMax;

/// Multiplicative reduction: computes `∏ data[i]`.
///
/// Identity element is `T::ONE`. Uses SIMD `mul` to accumulate lane products, then
/// reduces horizontally via a scalar lane-extraction loop (no `prod_reduce` on
/// `SimdKernel` — the hardware does not expose one universally).
///
/// # Zero-Cost Guarantee
///
/// `size_of::<Product>() == 0`. All branching over `Product` vs other ops is
/// eliminated via DCE during monomorphization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Product;

// ---------------------------------------------------------------------------
// Sealing impls
// ---------------------------------------------------------------------------

impl crate::private::Sealed for Sum {}
impl crate::private::Sealed for Dot {}
impl crate::private::Sealed for Min {}
impl crate::private::Sealed for Max {}
impl crate::private::Sealed for AbsSum {}
impl crate::private::Sealed for AbsMax {}
impl crate::private::Sealed for Product {}

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
    fn identity_scalar() -> T {
        T::ZERO
    }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a + b
    }
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
    fn identity_scalar() -> T {
        T::ZERO
    }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a + b
    }
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
    fn identity_scalar() -> T {
        T::MAX_VALUE
    }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a.min_scalar(b)
    }
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
    fn identity_scalar() -> T {
        T::MIN_VALUE
    }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a.max_scalar(b)
    }
}

impl<T: Scalar> ReductionOp<T> for AbsSum {
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        Arch::add(acc, Arch::abs(v))
    }
    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::sum_reduce(acc)
    }
    #[inline(always)]
    fn identity_scalar() -> T {
        T::ZERO
    }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a + b
    }
    #[inline(always)]
    fn scalar_accumulate(acc: T, elem: T) -> T {
        acc + elem.abs()
    }
    #[inline(always)]
    unsafe fn transform_vector<Arch: SimdKernel<T>>(v: Arch::Vector) -> Arch::Vector {
        Arch::abs(v)
    }
    #[inline(always)]
    unsafe fn combine_vectors<Arch: SimdKernel<T>>(
        a: Arch::Vector,
        b: Arch::Vector,
    ) -> Arch::Vector {
        Arch::add(a, b)
    }
}

impl<T: Scalar> ReductionOp<T> for AbsMax {
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        Arch::max(acc, Arch::abs(v))
    }
    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::max_reduce(acc)
    }
    #[inline(always)]
    fn identity_scalar() -> T {
        T::ZERO
    }
    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a.max_scalar(b)
    }
    #[inline(always)]
    fn scalar_accumulate(acc: T, elem: T) -> T {
        acc.max_scalar(elem.abs())
    }
    #[inline(always)]
    unsafe fn transform_vector<Arch: SimdKernel<T>>(v: Arch::Vector) -> Arch::Vector {
        Arch::abs(v)
    }
    #[inline(always)]
    unsafe fn combine_vectors<Arch: SimdKernel<T>>(
        a: Arch::Vector,
        b: Arch::Vector,
    ) -> Arch::Vector {
        Arch::max(a, b)
    }
}

impl<T: Scalar> ReductionOp<T> for Product {
    /// Accumulate: `acc = acc * v` (lane-wise multiply).
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(acc: Arch::Vector, v: Arch::Vector) -> Arch::Vector {
        Arch::mul(acc, v)
    }

    /// Finalize: store the accumulated product vector and reduce over lanes.
    ///
    /// There is no universal `prod_reduce` intrinsic, so this falls back to:
    /// 1. Store the `LANE_COUNT` partial products into a local stack array.
    /// 2. Scalar-fold with `*`.
    ///
    /// For Scalar arch this is always a single-element store + identity.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        // Fixed-size stack buffer — LANE_COUNT is a compile-time const, so the
        // compiler stack-allocates exactly `LANE_COUNT * size_of::<T>()` bytes.
        let mut buf = [T::ZERO; 16];
        let lanes = Arch::LANE_COUNT.min(16);
        Arch::store_unaligned(buf.as_mut_ptr(), acc);
        let mut result = T::ONE;
        for i in 0..lanes {
            result = result * buf[i];
        }
        result
    }

    #[inline(always)]
    fn identity_scalar() -> T {
        T::ONE
    }

    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T {
        a * b
    }
}
