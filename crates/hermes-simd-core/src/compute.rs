//! Unified computation view abstraction.
//!
//! `ComputeView` is the top-level sealed abstraction over dense SIMD views,
//! masked views, sparse matrix views, and bitboard views. It provides a
//! universal `len()` query and a blanket `reduce()` extension for dense views.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::bitboard::BitBoardView;
use crate::execution::ExecutionMode;
use crate::sparse::{SparseFormat, SparseView};
use crate::view::SimdView;

/// Top-level trait abstracting over dense, masked, sparse, tiled, and bitboard backends.
///
/// Sealed via the `SimdArch` bound (which requires `crate::private::Sealed`). All
/// implementations in this workspace are registered here; external crates cannot
/// implement `ComputeView` without also implementing `SimdArch` (which is sealed).
///
/// # Examples
///
/// Query the length of a dense `SimdView` through the `ComputeView` facade:
///
/// ```rust
/// use hermes_simd_core::compute::ComputeView;
/// use hermes_simd_core::view::SimdView;
/// use hermes_simd_intrinsics::Scalar;
/// use hermes_simd_core::align::Unaligned;
/// use hermes_simd_core::execution::Unmasked;
///
/// let data = [1.0_f32; 16];
/// let view: SimdView<'_, f32, Scalar, Unaligned, Unmasked, &[f32]> =
///     SimdView::new(&data).unwrap();
/// assert_eq!(view.len(), 16);
/// assert!(!view.is_empty());
/// ```
pub trait ComputeView {
    /// Scalar element type of the view (e.g. `f32`, `f64`, `u64`).
    type Element;

    /// SIMD architecture ZST marker.
    type Arch: SimdArch;

    /// Backend/format/execution mode selection marker.
    /// - Dense `SimdView`: `Mode` (e.g. `Unmasked`, `Masked`)
    /// - `SparseView`: `Format` (e.g. `Csr`, `SellP<4>`)
    /// - `BitBoardView`: `Backend` (e.g. `KoggeStone`)
    type Backend;

    /// Returns the logical size of the view.
    ///
    /// - `SimdView`: element count
    /// - `SparseView`: row count (`nrows()`)
    /// - `BitBoardView`: board element count
    fn len(&self) -> usize;

    /// Returns `true` if the view contains no elements.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T, Arch, Align, Mode, Ref> ComputeView for SimdView<'a, T, Arch, Align, Mode, Ref>
where
    Arch: SimdArch,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: 'a,
{
    type Element = T;
    type Arch = Arch;
    type Backend = Mode;

    #[inline(always)]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl<'a, T, Format, Arch> ComputeView for SparseView<'a, T, Format, Arch>
where
    Format: SparseFormat,
    Arch: SimdArch,
{
    type Element = T;
    type Arch = Arch;
    type Backend = Format;

    #[inline(always)]
    fn len(&self) -> usize {
        self.nrows()
    }
}

impl<'a, Backend, Arch, Ref> ComputeView for BitBoardView<'a, Backend, Arch, Ref>
where
    Arch: SimdArch,
    Ref: 'a,
{
    type Element = u64;
    type Arch = Arch;
    type Backend = Backend;

    #[inline(always)]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

// ---------------------------------------------------------------------------
// ComputeReduce — blanket extension trait for SIMD reduction over ComputeView
// ---------------------------------------------------------------------------

use crate::ops::ReductionOp;
use crate::scalar::Scalar;

/// Extension trait that provides `reduce()` over any [`SimdView`]-backed [`ComputeView`].
///
/// # Design
///
/// `ComputeReduce` is a blanket extension: it is automatically satisfied by any type
/// that implements `ComputeView` with a `SimdView` backend. It does not add a vtable
/// entry — the sole method `reduce` is `#[inline(always)]` and monomorphizes to the
/// same code as calling `view.reduce(op)` directly.
///
/// # Examples
///
/// ```rust
/// use hermes_simd_core::compute::{ComputeView, ComputeReduce};
/// use hermes_simd_core::view::SimdView;
/// use hermes_simd_core::ops::Sum;
/// use hermes_simd_intrinsics::Scalar;
/// use hermes_simd_core::align::Unaligned;
/// use hermes_simd_core::execution::Unmasked;
///
/// let data = [1.0_f32; 8];
/// let view: SimdView<'_, f32, Scalar, Unaligned, Unmasked, &[f32]> =
///     SimdView::new(&data).unwrap();
/// let total: f32 = view.compute_reduce(Sum);
/// assert!((total - 8.0_f32).abs() < 1e-6);
/// ```
pub trait ComputeReduce: ComputeView
where
    Self::Arch: crate::kernel::SimdKernel<Self::Element>,
    Self::Element: Scalar,
{
    /// Reduce all elements to a scalar using the given strategy.
    ///
    /// Delegates to the `SimdView::reduce` implementation, which uses
    /// multi-accumulator unrolled SIMD + scalar tail handling.
    fn compute_reduce<Op: ReductionOp<Self::Element>>(&self, op: Op) -> Self::Element;
}

impl<'a, T, Arch, Align, Mode, Ref> ComputeReduce for SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
    Arch: SimdArch + crate::kernel::SimdKernel<T>,
    Align: Alignment,
    Mode: ExecutionMode,
    Ref: 'a,
{
    #[inline(always)]
    fn compute_reduce<Op: ReductionOp<T>>(&self, op: Op) -> T {
        self.reduce(op)
    }
}
