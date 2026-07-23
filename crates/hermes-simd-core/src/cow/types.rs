//! `SimdCow` type definition, basic accessors, and trait implementations.

use crate::align::Alignment;
use crate::arch::{assert_arch_executable, SimdArch};
use crate::execution::Unmasked;
use crate::vec::AlignedVec;
use crate::view::SimdView;
use core::ops::Deref;

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
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn owned(vec: AlignedVec<T, Align>) -> Self {
        assert_arch_executable::<Arch>();
        Self::Owned(vec)
    }

    /// Returns true when this container is borrowing caller-owned storage.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns true when this container owns aligned storage.
    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
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

    /// Converts this `SimdCow` into an unaligned Cow, stripping the alignment guarantee zero-cost.
    #[inline]
    pub fn into_unaligned(self) -> SimdCow<'a, T, Arch, crate::align::Unaligned> {
        match self {
            Self::Borrowed(view) => SimdCow::Borrowed(view.into_unaligned()),
            Self::Owned(vec) => SimdCow::Owned(vec.into_unaligned()),
        }
    }

    /// Attempts to promote the alignment of this `SimdCow` to boundary `A` bytes.
    /// Returns `Some(SimdCow)` if the start pointer is aligned to `A` bytes, otherwise `None`.
    #[inline]
    pub fn try_into_aligned<const A: usize>(
        self,
    ) -> Option<SimdCow<'a, T, Arch, crate::align::Aligned<A>>> {
        match self {
            Self::Borrowed(view) => view.try_into_aligned::<A>().map(SimdCow::Borrowed),
            Self::Owned(vec) => {
                let addr = vec.as_ptr() as usize;
                if addr % A == 0 {
                    Some(SimdCow::Owned(unsafe { vec.into_alignment_unchecked() }))
                } else {
                    None
                }
            }
        }
    }

    /// Zero-copy sub-slice over a range of indices, returning an unaligned borrowed `SimdCow`.
    #[inline]
    pub fn slice_unaligned(
        &self,
        range: core::ops::Range<usize>,
    ) -> SimdCow<'_, T, Arch, crate::align::Unaligned> {
        match self {
            Self::Borrowed(view) => SimdCow::Borrowed(view.slice_unaligned(range)),
            Self::Owned(vec) => {
                let view = vec.view::<Arch>();
                SimdCow::Borrowed(view.slice_unaligned(range))
            }
        }
    }

    /// Zero-copy sub-slice over a range of indices, returning an aligned borrowed `SimdCow` with boundary `A` bytes.
    /// Returns `Some(SimdCow)` if the sub-slice satisfies the alignment, otherwise `None`.
    #[inline]
    pub fn slice_aligned<const A: usize>(
        &self,
        range: core::ops::Range<usize>,
    ) -> Option<SimdCow<'_, T, Arch, crate::align::Aligned<A>>> {
        match self {
            Self::Borrowed(view) => view.slice_aligned::<A>(range).map(SimdCow::Borrowed),
            Self::Owned(vec) => {
                let view = vec.view::<Arch>();
                view.slice_aligned::<A>(range).map(SimdCow::Borrowed)
            }
        }
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

impl<'a, 'b, T, Arch1, Arch2, Align1, Align2> PartialEq<SimdCow<'b, T, Arch2, Align2>>
    for SimdCow<'a, T, Arch1, Align1>
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
{
}

impl<'a, T: PartialEq, Arch: SimdArch, Align: Alignment> PartialEq<[T]>
    for SimdCow<'a, T, Arch, Align>
{
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        let s: &[T] = self;
        s == other
    }
}

impl<'a, T: PartialEq, Arch: SimdArch, Align: Alignment> PartialEq<SimdCow<'a, T, Arch, Align>>
    for [T]
{
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
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_slice(src: &[T]) -> Self {
        assert_arch_executable::<Arch>();
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

    /// Obtains a mutable `SimdView` over the data, promoting the Cow to owned if it is currently borrowed.
    #[inline]
    pub fn view_mut(&mut self) -> SimdView<'_, T, Arch, Align, Unmasked, &'_ mut [T]> {
        let vec = self.to_mut();
        vec.view_mut::<Arch>()
    }

    /// Zero-copy mutable sub-slice over a range of indices, promoting the Cow to owned if it is borrowed,
    /// and returning a mutable borrowed `SimdView`.
    #[inline]
    pub fn slice_unaligned_mut(
        &mut self,
        range: core::ops::Range<usize>,
    ) -> SimdView<'_, T, Arch, crate::align::Unaligned, Unmasked, &'_ mut [T]> {
        let vec = self.to_mut();
        let view = vec.view_mut::<Arch>();
        view.slice_unaligned_mut(range)
    }

    /// Zero-copy mutable sub-slice over a range of indices, promoting the Cow to owned if it is borrowed,
    /// and returning an aligned mutable borrowed `SimdView`.
    /// Returns `Some(SimdView)` if the sub-slice satisfies the alignment, otherwise `None`.
    #[inline]
    pub fn slice_aligned_mut<const A: usize>(
        &mut self,
        range: core::ops::Range<usize>,
    ) -> Option<SimdView<'_, T, Arch, crate::align::Aligned<A>, Unmasked, &'_ mut [T]>> {
        let vec = self.to_mut();
        let view = vec.view_mut::<Arch>();
        view.slice_aligned_mut::<A>(range)
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

impl<'a, T: Copy + 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    /// Construct a `SimdCow` from a standard-library `Cow<'a, [T]>`.
    ///
    /// - `std::borrow::Cow::Borrowed(s)`: attempts a zero-copy `borrow_slice`. If `s`
    ///   does not satisfy `Align`, falls back to one owned copy via `from_slice`.
    /// - `std::borrow::Cow::Owned(v)`: copies `v` into an `AlignedVec` (one allocation).
    ///
    /// The fallback ensures this conversion is always infallible while still achieving
    /// zero-copy for correctly-aligned inputs.
    ///
    /// # Panics
    /// If `Arch` cannot execute on this host.
    #[inline]
    pub fn from_std_cow(src: alloc::borrow::Cow<'a, [T]>) -> Self {
        assert_arch_executable::<Arch>();
        match src {
            alloc::borrow::Cow::Borrowed(s) => {
                // Try zero-copy; fall back to owned copy if alignment unsatisfied.
                Self::borrow_slice(s).unwrap_or_else(|| Self::Owned(AlignedVec::from_slice(s)))
            }
            alloc::borrow::Cow::Owned(v) => Self::Owned(AlignedVec::from_slice(&v)),
        }
    }
}
