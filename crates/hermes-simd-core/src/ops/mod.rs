//! Zero-cost operation strategy markers for SIMD reductions and elementwise transforms.
//!
//! `ReductionOp<T>`, `ElementOp<T>`, and `UnaryOp<T>` are sealed ZST traits parameterized
//! by the scalar type `T: Scalar`. Concrete strategies (`Sum`, `Dot`, `Mul`, `Add`, `Sub`,
//! `Abs`, `Neg`, `Sqrt`) implement these traits and are passed as ZST values — they carry no
//! runtime data and the compiler eliminates all abstraction overhead via monomorphization.
//!
//! # Module Organization
//!
//! | Sub-module | Contents |
//! |---|---|
//! | [`reduction`] | `ReductionOp<T>`, `Sum`, `Dot`, `Min`, `Max`, `Product` |
//! | [`elementwise`] | `ElementOp<T>`, `Mul`, `Add`, `Sub`, `Div`, `BitAnd`, `BitOr`, `BitXor`, `FmaAdd`, `Clamp` |
//! | [`unary`] | `UnaryOp<T>`, `Abs`, `Neg`, `Sqrt`, `Floor`, `Ceil`, `Round`, `Trunc` (and `Clamp` as `UnaryOp`) |
//! | [`scan`] | `ScanOp<T>`, `ScanMode`, `ScanAdd`, `ScanMul`, `ScanMin`, `ScanMax`, `Inclusive`, `Exclusive` |
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

pub mod elementwise;
pub mod reduction;
pub mod scan;
pub mod unary;

// Re-exports for backwards-compatible access at the `ops::*` level.
pub use elementwise::{Add, BitAnd, BitOr, BitXor, Clamp, Div, ElementOp, FmaAdd, Mul, Sub};
pub use reduction::{AbsMax, AbsSum, Dot, Max, Min, Product, ReductionOp, Sum};
pub use scan::{Exclusive, Inclusive, ScanAdd, ScanMax, ScanMin, ScanMode, ScanMul, ScanOp};
pub use unary::{Abs, Ceil, Floor, Neg, Popcount, RecipSqrt, Round, Sqrt, Trunc, UnaryOp};
