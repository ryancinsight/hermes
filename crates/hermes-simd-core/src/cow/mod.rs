//! Clone-on-write container for lazy SIMD memory layout allocation.
//!
//! `SimdCow` provides zero-copy read access to borrowed slices via `SimdView`,
//! deferring heap allocation until mutable or owned access is required.
//!
//! CoW combinators (`zip_cow`, `transform_in_place`) apply elementwise `ElementOp`
//! strategies without unnecessary intermediate allocations:
//! - `zip_cow`: exactly one `AlignedVec` allocation for the output.
//! - `transform_in_place`: promotes to owned (one allocation) if borrowed, then
//!   mutates in-place — subsequent calls are allocation-free.
//!
//! Scalar tail handling uses `ElementOp::apply_scalar` to avoid vector load/store
//! boundary UB at the end of non-multiple-of-LANE_COUNT slices.

use core::ops::Deref;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::ops::{ElementOp, ReductionOp};
use crate::view::{SimdView, SimdError};
use crate::vec::AlignedVec;
use crate::execution::Unmasked;
use crate::scalar::Scalar;

/// Operator overloads for Clone-on-Write SIMD containers.
pub mod ops;
/// Zero-copy serialization support for Clone-on-Write SIMD containers using `rkyv`.
pub mod rkyv;

pub use rkyv::{ArchivedSimdCow, SimdCowResolver};

/// A Clone-on-Write SIMD container.
///
/// Promotes zero-copy operations by borrowing slices as read-only `SimdView`s,
/// only copying to a heap-allocated, guaranteed-alignment `AlignedVec` when
/// mutable access or ownership is explicitly requested.
pub enum SimdCow<'a, T: 'a, Arch: SimdArch, Align: Alignment> {
    /// Borrowed read-only SIMD view.
    Borrowed(SimdView<'a, T, Arch, Align, Unmasked, &'a [T]>),
    /// Owned aligned buffer.
    Owned(AlignedVec<T, Align>),
}

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    /// Creates a borrowed `SimdCow` wrapping a `SimdView`.
    #[inline]
    pub fn borrowed(view: SimdView<'a, T, Arch, Align, Unmasked, &'a [T]>) -> Self {
        Self::Borrowed(view)
    }

    /// Creates an owned `SimdCow` wrapping an `AlignedVec`.
    #[inline]
    pub fn owned(vec: AlignedVec<T, Align>) -> Self {
        Self::Owned(vec)
    }

    /// Obtains a read-only `SimdView` over the data.
    #[inline(always)]
    pub fn view(&self) -> SimdView<'_, T, Arch, Align, Unmasked, &'_ [T]> {
        match self {
            Self::Borrowed(view) => *view,
            Self::Owned(vec) => vec.view(),
        }
    }

    /// Returns the number of elements in the contained slice.
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(view) => view.len(),
            Self::Owned(vec) => vec.as_slice().len(),
        }
    }

    /// Returns `true` if the contained slice is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T, Arch, Align> Clone for SimdCow<'a, T, Arch, Align>
where
    T: Clone,
    Arch: SimdArch,
    Align: Alignment,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(view) => Self::Borrowed(*view),
            Self::Owned(vec) => Self::Owned(vec.clone()),
        }
    }
}

impl<'a, T, Arch, Align> Default for SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    #[inline]
    fn default() -> Self {
        Self::Owned(AlignedVec::default())
    }
}

impl<'a, T, Arch, Align> core::fmt::Debug for SimdCow<'a, T, Arch, Align>
where
    T: core::fmt::Debug,
    Arch: SimdArch,
    Align: Alignment,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Borrowed(view) => f.debug_tuple("Borrowed").field(&view.as_slice()).finish(),
            Self::Owned(vec) => f.debug_tuple("Owned").field(&vec.as_slice()).finish(),
        }
    }
}

impl<'a, 'b, T, Arch1, Arch2, Align1, Align2> PartialEq<SimdCow<'b, T, Arch2, Align2>> for SimdCow<'a, T, Arch1, Align1>
where
    T: PartialEq,
    Arch1: SimdArch,
    Arch2: SimdArch,
    Align1: Alignment,
    Align2: Alignment,
{
    #[inline]
    fn eq(&self, other: &SimdCow<'b, T, Arch2, Align2>) -> bool {
        let slice_self: &[T] = self;
        let slice_other: &[T] = other;
        slice_self == slice_other
    }
}

impl<'a, T, Arch, Align> Eq for SimdCow<'a, T, Arch, Align>
where
    T: Eq,
    Arch: SimdArch,
    Align: Alignment,
{}

impl<'a, T: PartialEq, Arch: SimdArch, Align: Alignment> PartialEq<[T]> for SimdCow<'a, T, Arch, Align> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        let s: &[T] = self;
        s == other
    }
}

impl<'a, T: PartialEq, Arch: SimdArch, Align: Alignment> PartialEq<SimdCow<'a, T, Arch, Align>> for [T] {
    #[inline]
    fn eq(&self, other: &SimdCow<'a, T, Arch, Align>) -> bool {
        let s: &[T] = other;
        self == s
    }
}

impl<T: Copy, Arch: SimdArch, Align: Alignment> SimdCow<'static, T, Arch, Align> {
    /// Construct an owned `SimdCow` from a `&[T]` slice.
    ///
    /// One allocation, one `copy_nonoverlapping` — no intermediate `push` loop.
    /// The returned `SimdCow` is `'static` because it owns its data entirely.
    #[inline]
    pub fn from_slice(src: &[T]) -> Self {
        SimdCow::Owned(AlignedVec::from_slice(src))
    }
}

impl<'a, T: Copy + 'a, Arch: SimdArch, Align: Alignment> SimdCow<'a, T, Arch, Align> {
    /// Zero-copy constructor: borrow `src` as a `Borrowed` variant.
    ///
    /// No allocation. Returns `None` if `src` does not satisfy `Align` constraints
    /// (i.e., the pointer is not correctly aligned for the `Align` typestate).
    ///
    /// For `Unaligned`, this always returns `Some`.
    #[inline]
    pub fn borrow_slice(src: &'a [T]) -> Option<Self> {
        SimdView::new(src).map(Self::Borrowed)
    }
}

impl<'a, T: Copy + 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    /// Upgrades to `Owned` if borrowed, copying elements into an `AlignedVec`.
    ///
    /// Subsequent calls on an already-owned `SimdCow` are free — the data is
    /// already owned and no allocation or copy occurs.
    #[inline]
    pub fn to_mut(&mut self) -> &mut AlignedVec<T, Align> {
        if let Self::Borrowed(view) = *self {
            let owned = AlignedVec::from_slice(view.as_slice());
            *self = Self::Owned(owned);
        }
        match self {
            Self::Owned(ref mut vec) => vec,
            _ => unreachable!(),
        }
    }
}

impl<'a, T: 'a, Arch, Align> Deref for SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(view) => view.as_slice(),
            Self::Owned(vec) => vec.as_slice(),
        }
    }
}

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    /// Sum all elements using `SimdView::sum`.
    #[inline(always)]
    pub fn sum(&self) -> T {
        self.view().sum()
    }

    /// Compute the dot product with another `SimdCow`.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> Result<T, SimdError> {
        self.view().dot(&other.view())
    }

    /// Apply an `ElementOp` pairwise to `self` and `other`, returning a fully-owned
    /// `SimdCow<'static, T, Arch, Align>` backed by a single `AlignedVec` allocation.
    ///
    /// The SIMD vectorized loop processes `floor(len / LANE_COUNT) * LANE_COUNT` elements.
    /// The scalar tail uses `Op::apply_scalar` directly on individual elements — no
    /// vector loads or stores are performed in the tail, avoiding all boundary-condition UB.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if `self.len() != other.len()`.
    pub fn zip_cow<Op: ElementOp<T>>(
        &self,
        other: &SimdCow<'_, T, Arch, Align>,
        _op: Op,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        if self.len() != other.len() {
            return Err(SimdError::LengthMismatch);
        }
        let len = self.len();
        if len == 0 {
            return Ok(SimdCow::Owned(AlignedVec::new()));
        }

        let mut out = AlignedVec::with_capacity(len);
        let out_ptr: *mut T = out.as_mut_ptr();

        let view_self = self.view();
        let view_other = other.view();

        let mut chunks_self = view_self.simd_chunks();
        let mut chunks_other = view_other.simd_chunks();

        let mut i = 0usize;
        for (chunk_self, chunk_other) in (&mut chunks_self).zip(&mut chunks_other) {
            unsafe {
                let va = Arch::load_unaligned(chunk_self.as_ptr());
                let vb = Arch::load_unaligned(chunk_other.as_ptr());
                let vr = Op::apply::<Arch>(va, vb);
                Arch::store_unaligned(out_ptr.add(i), vr);
            }
            i += Arch::LANE_COUNT;
        }

        let remainder_self = chunks_self.remainder();
        let remainder_other = chunks_other.remainder();

        for (&a, &b) in remainder_self.iter().zip(remainder_other.iter()) {
            unsafe {
                core::ptr::write(out_ptr.add(i), Op::apply_scalar(a, b));
            }
            i += 1;
        }

        unsafe {
            out.set_len(len);
        }

        Ok(SimdCow::Owned(out))
    }

    /// Apply an `ElementOp` in-place: `self[i] = op(self[i], other[i])`.
    ///
    /// If `self` is `Borrowed`, promotes to `Owned` first (one allocation).
    /// Subsequent calls on the same already-owned `self` are allocation-free.
    ///
    /// The scalar tail uses `Op::apply_scalar` to avoid vector boundary UB.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if `self.len() != other.len()`.
    pub fn transform_in_place<Op: ElementOp<T>>(
        &mut self,
        other: &SimdCow<'_, T, Arch, Align>,
        _op: Op,
    ) -> Result<(), SimdError> {
        if self.len() != other.len() {
            return Err(SimdError::LengthMismatch);
        }
        // Promote to owned if borrowed (one allocation, amortized over subsequent calls).
        let _ = self.to_mut();

        let other_view = other.view();

        let out_slice = match self {
            Self::Owned(ref mut vec) => vec.as_mut_slice(),
            _ => unreachable!(),
        };

        let self_view: SimdView<'_, T, Arch, Align, Unmasked, &mut [T]> =
            SimdView::new_mut(out_slice).expect("alignment invariant violated");

        let mut chunks_self = self_view.simd_chunks_mut();
        let mut chunks_other = other_view.simd_chunks();

        for (mut chunk_self, chunk_other) in (&mut chunks_self).zip(&mut chunks_other) {
            unsafe {
                let va = Arch::load_unaligned(chunk_self.as_ptr());
                let vb = Arch::load_unaligned(chunk_other.as_ptr());
                let vr = Op::apply::<Arch>(va, vb);
                Arch::store_unaligned(chunk_self.as_mut_ptr(), vr);
            }
        }

        let tail_self = chunks_self.into_remainder();
        let tail_other = chunks_other.remainder();
        for (a, &b) in tail_self.iter_mut().zip(tail_other.iter()) {
            *a = Op::apply_scalar(*a, b);
        }

        Ok(())
    }

    /// Apply a `ReductionOp` to this `SimdCow`, delegating to `SimdView::reduce`.
    ///
    /// Monomorphization is shared with the view path — no duplicate code.
    #[inline(always)]
    pub fn reduce<Op: ReductionOp<T>>(&self, op: Op) -> T {
        self.view().reduce(op)
    }
}
