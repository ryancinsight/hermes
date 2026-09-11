//! Interleaved complex samples held in one SIMD register.
//!
//! [`ComplexReg`] wraps a [`Vector`] whose lanes carry `[re, im, re, im, ...]`
//! — the layout complex data has in memory — and gives it complex arithmetic
//! that never leaves the register: multiply, conjugate multiply, and rotation
//! by `±i`, built from the adjacent-pair shuffle and alternating-FMA
//! primitives the backends provide. Addition and subtraction need nothing
//! special: lane-wise add of interleaved data *is* complex add, so those come
//! from the wrapped vector unchanged.
//!
//! # Why this exists
//!
//! The planar alternative — separate `re` and `im` registers — doubles the
//! register pressure of every value it holds, which is what forces planar FFT
//! kernels down to radix-4 where interleaved kernels run radix-8. This type is
//! the vocabulary for register-resident kernels over data in its natural
//! interleaved layout: no deinterleave on load, no interleave on store, and a
//! complex multiply that is three shuffles and two fused multiply-adds.
//!
//! The multiply recipes are the ones documented on the backend contract
//! (`kernel/backend.rs`): for lanes `a = [a.re, a.im]` and `w = [w.re, w.im]`,
//!
//! ```text
//! a * w       : fmaddsub(dup_even(a) * w,  dup_odd(a) * swap_adjacent(w))
//! a * conj(w) : fmsubadd(dup_odd(a) * swap_adjacent(w),  dup_even(a) * w)
//! ```
//!
//! each producing the real part on even lanes and the imaginary part on odd
//! lanes with one rounding on the combining operation.
//!
//! # Eunomia owns the number, this type owns the register
//!
//! [`eunomia::Complex`] is the complex number: the single value type, the
//! arithmetic contract, the thing held in memory. `ComplexReg` is not a second
//! complex type — its element type stays the scalar `T`, and the complex
//! structure is a lane-layout contract over a register. Where a single sample
//! crosses the boundary it does so as `Complex<T>`; slices of `Complex<T>`
//! reinterpret as `[T]` at the load/store boundary (safe for its `repr(C)`
//! layout) because a register load wants contiguous scalars, not an element
//! type.

use super::vector_reg::Vector;
use crate::arch::SimdArch;
use crate::kernel::{SimdKernel, SimdStorage};
use crate::scalar::Scalar;
use eunomia::Complex;

/// A SIMD register of interleaved complex samples: even lanes are real parts,
/// odd lanes imaginary parts.
///
/// `#[repr(transparent)]` over [`Vector`], so wrapping and unwrapping are
/// layout-free and every method monomorphizes to the same code the raw vector
/// operations would emit.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct ComplexReg<T, Arch>(Vector<T, Arch>)
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>;

impl<T, Arch> ComplexReg<T, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    /// Number of complex samples per register.
    pub const COMPLEX_COUNT: usize = <Arch as SimdStorage<T>>::LANE_COUNT / 2;

    /// Wraps an interleaved vector. The caller asserts the layout contract:
    /// even lanes real, odd lanes imaginary.
    #[inline(always)]
    #[must_use]
    pub fn from_interleaved(v: Vector<T, Arch>) -> Self {
        Self(v)
    }

    /// Unwraps to the interleaved vector.
    #[inline(always)]
    #[must_use]
    pub fn into_interleaved(self) -> Vector<T, Arch> {
        self.0
    }

    /// All samples zero.
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self(Vector::zero())
    }

    /// Broadcasts one complex value across every sample — the form a twiddle
    /// factor takes when one factor multiplies a whole register.
    ///
    /// Takes [`eunomia::Complex`] rather than a bare `(re, im)` pair: eunomia
    /// owns the complex number, and this crate consumes it.
    #[inline(always)]
    #[must_use]
    pub fn splat(sample: Complex<T>) -> Self {
        // One pair broadcast: `[re, im]` repeated over the register. The
        // interleave of two scalar broadcasts computes the same value but
        // pays a shuffle pair per twiddle, which a stage loop repeats.
        Self(Vector::splat_pair(sample.re, sample.im))
    }

    /// Complex multiply by a twiddle held as its direct and swapped rows.
    ///
    /// `w_swapped` is `w` with each sample's real and imaginary lanes
    /// exchanged ([`Vector::swap_adjacent`]), which a kernel whose twiddles
    /// are stored can keep beside the direct row: the multiply is then two
    /// duplicates, one product and one alternating FMA, with no shuffle of
    /// the twiddle per use. The same value as `self * w`, bitwise.
    #[inline(always)]
    #[must_use]
    pub fn mul_with_swapped(self, w: Self, w_swapped: Self) -> Self {
        let re_a = self.0.dup_even();
        let im_a = self.0.dup_odd();
        Self(re_a.fmaddsub(w.0, im_a * w_swapped.0))
    }

    /// Complex multiply by the conjugate, sample-wise: `self * conj(w)`.
    #[inline(always)]
    #[must_use]
    pub fn mul_conj(self, w: Self) -> Self {
        let re_a = self.0.dup_even();
        let im_a = self.0.dup_odd();
        let w_swapped = w.0.swap_adjacent();
        Self(im_a.fmsubadd(w_swapped, re_a * w.0))
    }

    /// Rotates every sample by `+i`: `(re, im)` becomes `(-im, re)`.
    ///
    /// The quarter-turn twiddle of every power-of-two transform, done as one
    /// shuffle and one sign flip: the swapped register xored with the sign
    /// mask on its even lanes. Negation is the sign bit, so a zero lane comes
    /// out `-0.0`, as scalar negation gives it.
    ///
    /// An earlier form spent an alternating fused multiply-add against zero
    /// for the flip; LLVM expands that into an add and a subtract against
    /// zero and a blend (it cannot fold `0 - c` to `-c` with signed zeros),
    /// four instructions for what is two, on every rotation of every radix-4
    /// stage.
    #[inline(always)]
    #[must_use]
    pub fn mul_i(self) -> Self {
        Self(Self::flip_lanes(
            self.0.swap_adjacent(),
            T::SIGN_MASK,
            T::ZERO,
        ))
    }

    /// Rotates every sample by `-i`: `(re, im)` becomes `(im, -re)`.
    ///
    /// The mirror of [`Self::mul_i`]: the swap, then the sign flip on the odd
    /// lanes.
    #[inline(always)]
    #[must_use]
    pub fn mul_neg_i(self) -> Self {
        Self(Self::flip_lanes(
            self.0.swap_adjacent(),
            T::ZERO,
            T::SIGN_MASK,
        ))
    }

    /// Xors every even lane with `even` and every odd lane with `odd`, the
    /// mask register built without re-probing the host.
    ///
    /// [`Vector::splat_pair`] is a public constructor, so it asks whether the
    /// host supports `Arch` before handing one out. Inside a method taking
    /// `self` that question is already answered: a `ComplexReg` can only be
    /// built from a [`Vector`], and a `Vector` can only be built with proof.
    /// Asking again is not merely redundant — the probe is a call, so the
    /// register allocator spills every live vector around it, and a rotation
    /// used in an inner butterfly loop drags the whole working set through
    /// the stack each time (`apollo gap_audit.md#base-kernel-probes`).
    #[inline(always)]
    fn flip_lanes(value: Vector<T, Arch>, even: T, odd: T) -> Vector<T, Arch> {
        // SAFETY: the caller holds a `ComplexReg<T, Arch>`, which proves the
        // host executes `Arch` for `T`, since one cannot be constructed
        // without a `Vector`, which cannot be constructed without that proof.
        Vector::new(unsafe { Arch::bitxor(value.raw, Arch::splat_pair(even, odd)) })
    }

    /// Exchanges neighbouring complex samples: `[c0, c1, c2, c3]` becomes
    /// `[c1, c0, c3, c2]`.
    ///
    /// The operand pairing of a distance-one butterfly held in registers. A
    /// register holding a single sample passes through unchanged, per the
    /// backend's lone-pair convention.
    #[inline(always)]
    #[must_use]
    pub fn swap_samples(self) -> Self {
        Self(self.0.swap_pairs())
    }

    /// Transposes a square tile of complex-register rows in place.
    ///
    /// A tile contains exactly [`Self::COMPLEX_COUNT`] registers, each holding
    /// that many interleaved complex samples. Sample `(r, c)` moves to
    /// `(c, r)` while its real and imaginary lanes remain adjacent.
    ///
    /// # Panics
    /// Panics if `tile` does not hold exactly [`Self::COMPLEX_COUNT`] rows.
    #[inline(always)]
    pub fn transpose_square(tile: &mut [Self]) {
        assert_eq!(
            tile.len(),
            Self::COMPLEX_COUNT,
            "tile must hold COMPLEX_COUNT rows"
        );
        // SAFETY: `ComplexReg` and `Vector` are both `#[repr(transparent)]`
        // over `Arch::Vector`; the slice cast preserves layout. Constructing
        // every register proved host support for `Arch`, and the length check
        // establishes the backend method's exact-square precondition.
        unsafe {
            let raw = core::slice::from_raw_parts_mut(
                tile.as_mut_ptr().cast::<<Arch as SimdStorage<T>>::Vector>(),
                tile.len(),
            );
            Arch::transpose_interleaved_square(raw);
        }
    }

    /// The butterfly pair `(self + other, self - other)` in one call.
    ///
    /// Lane-wise add and subtract of interleaved data are complex add and
    /// subtract, so this is exactly two vector instructions; it exists so
    /// codelets read as butterflies rather than as lane arithmetic.
    #[inline(always)]
    #[must_use]
    pub fn butterfly(self, other: Self) -> (Self, Self) {
        (Self(self.0 + other.0), Self(self.0 - other.0))
    }
}

/// Sample-wise complex multiply.
///
/// Three shuffles and one multiply feeding one alternating FMA, so the
/// combining add/subtract rounds once. This is the operator form because the
/// semantics are exactly complex multiplication — there is no separate
/// method to confuse it with.
impl<T, Arch> core::ops::Mul for ComplexReg<T, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, w: Self) -> Self {
        let re_a = self.0.dup_even();
        let im_a = self.0.dup_odd();
        let w_swapped = w.0.swap_adjacent();
        Self(re_a.fmaddsub(w.0, im_a * w_swapped))
    }
}

impl<T, Arch> core::ops::Add for ComplexReg<T, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<T, Arch> core::ops::Sub for ComplexReg<T, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
