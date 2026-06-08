//! Layout marker ZSTs and the sealed `Layout` trait.
//!
//! Only [`RowMajor`] and [`ColMajor`] implement `Layout`; the `crate::private::Sealed`
//! supertrait prevents external implementations.

/// Row-major (C-order) layout marker ZST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowMajor;

/// Column-major (Fortran-order) layout marker ZST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColMajor;

/// Sealed marker trait for tensor layout ZSTs.
///
/// External crates cannot implement this trait — the `crate::private::Sealed` supertrait
/// is `pub(crate)` only. Only `RowMajor` and `ColMajor` satisfy `Layout`.
/// `TensorView<_, _, _, L>` requires `L: Layout`, preventing accidental use of
/// arbitrary unit types as layout parameters.
pub trait Layout: crate::private::Sealed + Copy + 'static {}

impl crate::private::Sealed for RowMajor {}
impl crate::private::Sealed for ColMajor {}
impl Layout for RowMajor {}
impl Layout for ColMajor {}
