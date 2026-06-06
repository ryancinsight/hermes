//! Unified computation view abstraction.
//!
//! `ComputeView` is the top-level sealed abstraction over dense SIMD views,
//! masked views, sparse matrix views, and bitboard views. It provides a
//! universal `len()` query and a blanket `reduce()` extension for dense views.

use crate::arch::SimdArch;
use crate::view::SimdView;
use crate::align::Alignment;
use crate::execution::ExecutionMode;
use crate::sparse::{SparseView, SparseFormat};
use crate::bitboard::BitBoardView;

/// Top-level trait abstracting over dense, masked, sparse, tiled, and bitboard backends.
///
/// Sealed via the `SimdArch` bound (which requires `crate::private::Sealed`). All
/// implementations in this workspace are registered here; external crates cannot
/// implement `ComputeView` without also implementing `SimdArch` (which is sealed).
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