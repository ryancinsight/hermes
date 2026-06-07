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
/// Norm, normalize, and scalar-broadcast arithmetic on Clone-on-Write SIMD containers.
pub mod math;
/// Zero-copy serialization support for Clone-on-Write SIMD containers using `rkyv`.
pub mod rkyv;

pub use rkyv::{ArchivedSimdCow, SimdCowResolver};
pub use hermes_numeric::{ArchivedPacked4Cow, Packed4CowResolver};

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

    /// Converts this `SimdCow` into an unaligned Cow, stripping the alignment guarantee zero-cost.
    #[inline]
    pub fn into_unaligned(self) -> SimdCow<'a, T, Arch, crate::align::Unaligned> {
        match self {
            Self::Borrowed(view) => SimdCow::Borrowed(view.into_unaligned()),
            Self::Owned(vec) => SimdCow::Owned(vec.into_alignment()),
        }
    }

    /// Attempts to promote the alignment of this `SimdCow` to boundary `A` bytes.
    /// Returns `Some(SimdCow)` if the start pointer is aligned to `A` bytes, otherwise `None`.
    #[inline]
    pub fn try_into_aligned<const A: usize>(self) -> Option<SimdCow<'a, T, Arch, crate::align::Aligned<A>>> {
        match self {
            Self::Borrowed(view) => view.try_into_aligned::<A>().map(SimdCow::Borrowed),
            Self::Owned(vec) => {
                let addr = vec.as_ptr() as usize;
                if addr % A == 0 {
                    Some(SimdCow::Owned(vec.into_alignment()))
                } else {
                    None
                }
            }
        }
    }

    /// Zero-copy sub-slice over a range of indices, returning an unaligned borrowed `SimdCow`.
    #[inline]
    pub fn slice_unaligned(&self, range: core::ops::Range<usize>) -> SimdCow<'_, T, Arch, crate::align::Unaligned> {
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
    pub fn slice_aligned<const A: usize>(&self, range: core::ops::Range<usize>) -> Option<SimdCow<'_, T, Arch, crate::align::Aligned<A>>> {
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

    /// Obtains a mutable `SimdView` over the data, promoting the Cow to owned if it is currently borrowed.
    #[inline]
    pub fn view_mut(&mut self) -> SimdView<'_, T, Arch, Align, Unmasked, &'_ mut [T]> {
        let vec = self.to_mut();
        vec.view_mut::<Arch>()
    }

    /// Zero-copy mutable sub-slice over a range of indices, promoting the Cow to owned if it is borrowed,
    /// and returning a mutable borrowed `SimdView`.
    #[inline]
    pub fn slice_unaligned_mut(&mut self, range: core::ops::Range<usize>) -> SimdView<'_, T, Arch, crate::align::Unaligned, Unmasked, &'_ mut [T]> {
        let vec = self.to_mut();
        let view = vec.view_mut::<Arch>();
        view.slice_unaligned_mut(range)
    }

    /// Zero-copy mutable sub-slice over a range of indices, promoting the Cow to owned if it is borrowed,
    /// and returning an aligned mutable borrowed `SimdView`.
    /// Returns `Some(SimdView)` if the sub-slice satisfies the alignment, otherwise `None`.
    #[inline]
    pub fn slice_aligned_mut<const A: usize>(&mut self, range: core::ops::Range<usize>) -> Option<SimdView<'_, T, Arch, crate::align::Aligned<A>, Unmasked, &'_ mut [T]>> {
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
                let vr = _op.apply::<Arch>(va, vb);
                Arch::store_unaligned(out_ptr.add(i), vr);
            }
            i += Arch::LANE_COUNT;
        }

        let remainder_self = chunks_self.remainder();
        let remainder_other = chunks_other.remainder();

        for (&a, &b) in remainder_self.iter().zip(remainder_other.iter()) {
            unsafe {
                core::ptr::write(out_ptr.add(i), _op.apply_scalar(a, b));
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
        // `to_mut` promotes borrowed → owned (one allocation if borrowed, free if owned).
        // Use the returned reference directly — no secondary match required.
        let out_slice = self.to_mut().as_mut_slice();

        let other_view = other.view();

        let self_view: SimdView<'_, T, Arch, Align, Unmasked, &mut [T]> =
            SimdView::new_mut(out_slice).expect("alignment invariant violated");

        let mut chunks_self = self_view.simd_chunks_mut();
        let mut chunks_other = other_view.simd_chunks();

        for (mut chunk_self, chunk_other) in (&mut chunks_self).zip(&mut chunks_other) {
            unsafe {
                let va = Arch::load_unaligned(chunk_self.as_ptr());
                let vb = Arch::load_unaligned(chunk_other.as_ptr());
                let vr = _op.apply::<Arch>(va, vb);
                Arch::store_unaligned(chunk_self.as_mut_ptr(), vr);
            }
        }

        let tail_self = chunks_self.into_remainder();
        let tail_other = chunks_other.remainder();
        for (a, &b) in tail_self.iter_mut().zip(tail_other.iter()) {
            *a = _op.apply_scalar(*a, b);
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

    // -----------------------------------------------------------------------
    // Arithmetic combinators — each returns `SimdCow<'static, ...>` (owned)
    // -----------------------------------------------------------------------

    /// Elementwise addition: `self[i] + other[i]`.
    ///
    /// Allocates one `AlignedVec` output. Zero-copy on both operands.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn add_cow(&self, other: &SimdCow<'_, T, Arch, Align>) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, crate::ops::Add)
    }

    /// Elementwise subtraction: `self[i] - other[i]`.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn sub_cow(&self, other: &SimdCow<'_, T, Arch, Align>) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, crate::ops::Sub)
    }

    /// Elementwise multiplication: `self[i] * other[i]`.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn mul_cow(&self, other: &SimdCow<'_, T, Arch, Align>) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, crate::ops::Mul)
    }

    /// Elementwise clamp to `[lo, hi]`: `min(max(self[i], lo), hi)`.
    ///
    /// Delegates to [`Self::map_unary`] with [`crate::ops::Clamp`].
    /// One allocation for the output `AlignedVec`. No manually inlined SIMD loop.
    #[inline]
    pub fn clamp_cow(&self, lo: T, hi: T) -> SimdCow<'static, T, Arch, Align> {
        self.map_unary(crate::ops::Clamp::new(lo, hi))
    }
}

// ---------------------------------------------------------------------------
// Ergonomic conversions
// ---------------------------------------------------------------------------

/// Adopt an `AlignedVec` as an owned `SimdCow` — zero-cost, no allocation.
impl<'a, T, Arch, Align> From<AlignedVec<T, Align>> for SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    #[inline]
    fn from(vec: AlignedVec<T, Align>) -> Self {
        Self::Owned(vec)
    }
}

/// Copy a standard `Vec<T>` into a new owned `SimdCow`, allocating one aligned buffer.
impl<'a, T: Copy, Arch, Align> From<alloc::vec::Vec<T>> for SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    #[inline]
    fn from(v: alloc::vec::Vec<T>) -> Self {
        Self::Owned(AlignedVec::from_slice(&v))
    }
}

impl<'a, T: Copy + 'a, Arch, Align> Extend<T> for SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    /// Extend the `SimdCow`, promoting to owned if currently borrowed.
    ///
    /// After promotion, subsequent `extend` calls are allocation-free as long as the
    /// `AlignedVec` has sufficient capacity.
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let vec = self.to_mut();
        for item in iter {
            vec.push(item);
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
    #[inline]
    pub fn from_std_cow(src: alloc::borrow::Cow<'a, [T]>) -> Self {
        match src {
            alloc::borrow::Cow::Borrowed(s) => {
                // Try zero-copy; fall back to owned copy if alignment unsatisfied.
                Self::borrow_slice(s).unwrap_or_else(|| Self::Owned(AlignedVec::from_slice(s)))
            }
            alloc::borrow::Cow::Owned(v) => Self::Owned(AlignedVec::from_slice(&v)),
        }
    }
}

// ---------------------------------------------------------------------------
// SimdCow unary map, scale, fill, and argmin/argmax extensions
// ---------------------------------------------------------------------------

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    /// Apply a `UnaryOp<T>` to every element, returning a fully-owned
    /// `SimdCow<'static, T, Arch, Align>` backed by a single `AlignedVec` allocation.
    ///
    /// Zero intermediate copies: one allocation, one vectorized pass.
    #[inline]
    pub fn map_unary<Op: crate::ops::UnaryOp<T>>(&self, op: Op) -> SimdCow<'static, T, Arch, Align> {
        let len = self.len();
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        // SAFETY: we write every element below before returning.
        unsafe { out.set_len(len); }
        // We don't gate on errors here: lengths match by construction.
        let _ = self.view().map_unary(op, out.as_mut_slice());
        SimdCow::Owned(out)
    }

    /// Apply a `UnaryOp<T>` in-place: `self[i] = op(self[i])`.
    ///
    /// Promotes `self` to owned if currently borrowed (one allocation).
    /// Subsequent calls on the same already-owned `SimdCow` are allocation-free.
    #[inline]
    pub fn map_unary_in_place<Op: crate::ops::UnaryOp<T>>(&mut self, op: Op) {
        self.view_mut().map_unary_in_place(op);
    }

    /// Multiply every element by `scalar` in-place: `self[i] *= scalar`.
    ///
    /// Uses `Arch::splat(scalar)` + `Arch::mul` to broadcast-multiply without
    /// a second `SimdCow`. Promotes to owned if currently borrowed (one allocation).
    #[inline]
    pub fn scale_in_place(&mut self, scalar: T) {
        let len = self.len();
        if len == 0 { return; }
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let vec = self.to_mut();
        let ptr = vec.as_mut_ptr();

        unsafe {
            let vsplat = Arch::splat(scalar);

            let load = |p: *const T| -> Arch::Vector {
                if Align::IS_ALIGNED { Arch::load_aligned(p) } else { Arch::load_unaligned(p) }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if Align::IS_ALIGNED { Arch::store_aligned(p, v); } else { Arch::store_unaligned(p, v); }
            };

            let mut i = 0usize;
            while i < simd_len {
                let p = ptr.add(i);
                let v = load(p);
                store(p, Arch::mul(v, vsplat));
                i += lane_count;
            }
        }

        // Scalar tail
        let slice = vec.as_mut_slice();
        for i in simd_len..len {
            slice[i] = slice[i] * scalar;
        }
    }

    /// Return an owned `SimdCow` with every element multiplied by `scalar`.
    ///
    /// One allocation. Delegates to `clone` + `scale_in_place`.
    #[inline]
    pub fn scale(&self, scalar: T) -> SimdCow<'static, T, Arch, Align> {
        let mut owned: SimdCow<'static, T, Arch, Align> = SimdCow::from_slice(self.as_ref());
        owned.scale_in_place(scalar);
        owned
    }

    /// Construct an owned `SimdCow` of length `len` with every element set to `value`.
    ///
    /// Uses `Arch::splat` + `Arch::store_unaligned` for the SIMD prefix;
    /// scalar assignment for the tail. One allocation.
    #[inline]
    pub fn splat_fill(value: T, len: usize) -> SimdCow<'static, T, Arch, Align> {
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        unsafe { out.set_len(len); }
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr = out.as_mut_ptr();

        unsafe {
            let vsplat = Arch::splat(value);
            let mut i = 0usize;
            while i < simd_len {
                Arch::store_unaligned(ptr.add(i), vsplat);
                i += lane_count;
            }
        }

        let slice = out.as_mut_slice();
        for i in simd_len..len {
            slice[i] = value;
        }

        SimdCow::Owned(out)
    }

    /// Construct an owned `SimdCow` of length `len` filled with `T::ZERO`.
    #[inline]
    pub fn zeros(len: usize) -> SimdCow<'static, T, Arch, Align> {
        Self::splat_fill(T::ZERO, len)
    }

    /// Construct an owned `SimdCow` of length `len` filled with `T::ONE`.
    #[inline]
    pub fn ones(len: usize) -> SimdCow<'static, T, Arch, Align> {
        Self::splat_fill(T::ONE, len)
    }

    /// Returns `Some((index, value))` of the minimum element, or `None` for empty.
    #[inline]
    pub fn argmin(&self) -> Option<(usize, T)>
    where
        T: crate::scalar::NumericElement,
    {
        self.view().argmin()
    }

    /// Returns `Some((index, value))` of the maximum element, or `None` for empty.
    #[inline]
    pub fn argmax(&self) -> Option<(usize, T)>
    where
        T: crate::scalar::NumericElement,
    {
        self.view().argmax()
    }

    /// Indirectly load (gather) elements from this view using indices, returning a new owned `SimdCow`.
    ///
    /// # Errors
    /// Returns `SimdError::IndexOutOfBounds` if any index in `indices` is out of bounds.
    #[inline]
    pub fn gather(&self, indices: &[i32]) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        let len = indices.len();
        let mut out = AlignedVec::with_capacity(len);
        unsafe { out.set_len(len); }
        self.view().gather(indices, out.as_mut_slice())?;
        Ok(SimdCow::Owned(out))
    }

    /// Perform a prefix scan (inclusive or exclusive) of the view using the specified operation,
    /// returning a new owned `SimdCow`.
    #[inline]
    pub fn prefix_scan<Op, SMode>(&self, op: Op, mode: SMode) -> Result<SimdCow<'static, T, Arch, Align>, SimdError>
    where
        Op: crate::ops::ScanOp<T>,
        SMode: crate::ops::ScanMode,
    {
        let len = self.len();
        let mut out = AlignedVec::with_capacity(len);
        unsafe { out.set_len(len); }
        self.view().prefix_scan(out.as_mut_slice(), op, mode)?;
        Ok(SimdCow::Owned(out))
    }

    /// Perform an in-place prefix scan (inclusive or exclusive) of the view using the specified operation.
    ///
    /// Promotes `self` to owned if currently borrowed (one allocation).
    /// Subsequent calls on the same already-owned `SimdCow` are allocation-free.
    #[inline]
    pub fn prefix_scan_in_place<Op, SMode>(&mut self, _op: Op, _mode: SMode) -> Result<(), SimdError>
    where
        Op: crate::ops::ScanOp<T>,
        SMode: crate::ops::ScanMode,
    {
        let len = self.len();
        if len == 0 { return Ok(()); }

        // `to_mut` promotes borrowed → owned (one allocation if borrowed, free if owned),
        // then returns a direct `&mut AlignedVec`. No secondary match required.
        let slice = self.to_mut().as_mut_slice();

        let mut acc = Op::identity();
        if SMode::IS_INCLUSIVE {
            for x in slice {
                acc = Op::combine(acc, *x);
                *x = acc;
            }
        } else {
            for x in slice {
                let temp = *x;
                *x = acc;
                acc = Op::combine(acc, temp);
            }
        }
        Ok(())
    }
}
