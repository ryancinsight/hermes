//! Popcount-style horizontal reductions, with periodic accumulator flushes.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::{SimdArith, SimdBitwise, SimdCompare, SimdLoadStore, SimdMask, SimdReduce};
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

/// Periodic accumulator-flush interval for popcount-style horizontal reductions,
/// sized by element width to bound intermediate-sum precision loss. 2-byte types
/// (`f16`/`bf16`/`i16`) have a small exact-integer range (256/2048), so partials
/// are flushed every 128 chunks; wider types tolerate 32768 chunks per flush.
#[inline(always)]
const fn flush_limit_for<T>() -> usize {
    if core::mem::size_of::<T>() == 2 {
        128
    } else {
        32768
    }
}

impl<
        'a,
        T: 'a,
        Arch: SimdArch
            + SimdLoadStore<T>
            + SimdArith<T>
            + SimdBitwise<T>
            + SimdCompare<T>
            + SimdMask<T>
            + SimdReduce<T>,
        Align: Alignment,
        Mode: ExecutionMode,
        Ref: 'a,
    > SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Computes the horizontal sum of population counts of all elements.
    #[inline]
    #[must_use]
    pub fn reduce_popcount(&self) -> usize {
        let data = self.as_slice();
        let len = data.len();
        let lane_count = Arch::LANE_COUNT;
        let unroll_factor = Arch::UNROLL_FACTOR;
        let chunk_size = lane_count * unroll_factor;
        let unrolled_simd_len = (len / chunk_size) * chunk_size;
        let simd_len = (len / lane_count) * lane_count;
        let mut total: usize = 0;
        let mut i = 0usize;

        // SAFETY: `Arch::load_*` is a target-feature kernel (module invariant),
        // and every call site passes a pointer whose `LANE_COUNT` read stays
        // within the source slice (offsets bounded by `simd_len`).
        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        let flush_limit = flush_limit_for::<T>();

        // Unrolled loop (4-way register accumulation)
        if unrolled_simd_len > 0 {
            let mut acc0 = unsafe { Arch::zero() };
            let mut acc1 = unsafe { Arch::zero() };
            let mut acc2 = unsafe { Arch::zero() };
            let mut acc3 = unsafe { Arch::zero() };
            let mut count = 0;

            while i < unrolled_simd_len {
                // SAFETY: `unrolled_simd_len` is a multiple of `chunk_size`, so
                // `i + lane_count*3 + LANE_COUNT <= unrolled_simd_len <= len`; each
                // load reads a `LANE_COUNT` window within `data`. Kernels covered
                // by the module invariant.
                unsafe {
                    let v0 = load(data.as_ptr().add(i));
                    let v1 = load(data.as_ptr().add(i + lane_count));
                    let v2 = load(data.as_ptr().add(i + lane_count * 2));
                    let v3 = load(data.as_ptr().add(i + lane_count * 3));

                    acc0 = Arch::add(acc0, Arch::popcount(v0));
                    acc1 = Arch::add(acc1, Arch::popcount(v1));
                    acc2 = Arch::add(acc2, Arch::popcount(v2));
                    acc3 = Arch::add(acc3, Arch::popcount(v3));
                }
                i += chunk_size;
                count += 1;

                if count == flush_limit {
                    unsafe {
                        let mut acc = Arch::add(acc0, acc1);
                        acc = Arch::add(acc, acc2);
                        acc = Arch::add(acc, acc3);
                        total += Arch::sum_reduce(acc).to_f64() as usize;
                        acc0 = Arch::zero();
                        acc1 = Arch::zero();
                        acc2 = Arch::zero();
                        acc3 = Arch::zero();
                    }
                    count = 0;
                }
            }

            unsafe {
                let mut acc = Arch::add(acc0, acc1);
                acc = Arch::add(acc, acc2);
                acc = Arch::add(acc, acc3);
                total += Arch::sum_reduce(acc).to_f64() as usize;
            }
        }

        // Middle loop (single register accumulation)
        if i < simd_len {
            let mut acc = unsafe { Arch::zero() };
            while i < simd_len {
                // SAFETY: `i < simd_len = (len / LANE_COUNT) * LANE_COUNT`, so the
                // load reads a `LANE_COUNT` window within `data`.
                unsafe {
                    let v = load(data.as_ptr().add(i));
                    acc = Arch::add(acc, Arch::popcount(v));
                }
                i += lane_count;
            }
            total += unsafe { Arch::sum_reduce(acc) }.to_f64() as usize;
        }

        // Masked final vector. Popcount is integer-valued, so reducing the
        // live tail as one masked vector preserves the exact count while
        // removing the element-at-a-time cleanup loop.
        let tail = len - simd_len;
        if tail != 0 {
            // SAFETY: `data` contains the exact `tail` suffix and the leading
            // mask selects no lane beyond it.
            let tail_count = unsafe {
                let mask = Arch::leading_k_mask(tail);
                let value = Arch::masked_load_partial(
                    data.as_ptr().add(simd_len),
                    tail,
                    mask,
                    Arch::zero(),
                );
                Arch::masked_sum_reduce(Arch::popcount(value), mask).to_f64() as usize
            };
            total += tail_count;
        }

        total
    }

    /// Horizontal sum of population counts of `op(self[i], other[i])` for a
    /// bitwise [`ElementOp`] (`BitAnd`/`BitOr`/`BitXor`).
    ///
    /// One generic 4-accumulator popcount reduction shared by
    /// [`reduce_popcount_and`](Self::reduce_popcount_and),
    /// [`reduce_popcount_or`](Self::reduce_popcount_or) and
    /// [`reduce_popcount_xor`](Self::reduce_popcount_xor). The combining op is a
    /// ZST monomorphized away, so each wrapper compiles to exactly the code its
    /// former hand-written body did — the three ~100-line bodies collapse to one.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] if slice lengths differ.
    #[inline]
    fn reduce_popcount_op<ORef, Op>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        op: Op,
    ) -> Result<usize, SimdError>
    where
        ORef: 'a,
        Op: crate::ops::ElementOp<T>,
    {
        crate::view::check_lengths_equal(self.len(), other.len())?;
        let s = self.as_slice();
        let o = other.as_slice();
        let len = s.len();
        let lane_count = Arch::LANE_COUNT;
        let unroll_factor = Arch::UNROLL_FACTOR;
        let chunk_size = lane_count * unroll_factor;
        let unrolled_simd_len = (len / chunk_size) * chunk_size;
        let simd_len = (len / lane_count) * lane_count;
        let mut total: usize = 0;
        let mut i = 0usize;

        // SAFETY: `Arch::load_*` is a target-feature kernel (module invariant),
        // and every call site passes a pointer whose `LANE_COUNT` read stays
        // within the source slice (offsets bounded by `simd_len`).
        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        let flush_limit = flush_limit_for::<T>();

        if unrolled_simd_len > 0 {
            let mut acc0 = unsafe { Arch::zero() };
            let mut acc1 = unsafe { Arch::zero() };
            let mut acc2 = unsafe { Arch::zero() };
            let mut acc3 = unsafe { Arch::zero() };
            let mut count = 0;

            while i < unrolled_simd_len {
                // SAFETY: `unrolled_simd_len` is a multiple of `chunk_size` and
                // `s`/`o` are equal length, so `i + lane_count*3 + LANE_COUNT`
                // stays within both slices. Kernels covered by the module invariant.
                unsafe {
                    let va0 = load(s.as_ptr().add(i));
                    let vb0 = load(o.as_ptr().add(i));
                    let va1 = load(s.as_ptr().add(i + lane_count));
                    let vb1 = load(o.as_ptr().add(i + lane_count));
                    let va2 = load(s.as_ptr().add(i + lane_count * 2));
                    let vb2 = load(o.as_ptr().add(i + lane_count * 2));
                    let va3 = load(s.as_ptr().add(i + lane_count * 3));
                    let vb3 = load(o.as_ptr().add(i + lane_count * 3));

                    acc0 = Arch::add(acc0, Arch::popcount(op.apply::<Arch>(va0, vb0)));
                    acc1 = Arch::add(acc1, Arch::popcount(op.apply::<Arch>(va1, vb1)));
                    acc2 = Arch::add(acc2, Arch::popcount(op.apply::<Arch>(va2, vb2)));
                    acc3 = Arch::add(acc3, Arch::popcount(op.apply::<Arch>(va3, vb3)));
                }
                i += chunk_size;
                count += 1;

                if count == flush_limit {
                    unsafe {
                        let mut acc = Arch::add(acc0, acc1);
                        acc = Arch::add(acc, acc2);
                        acc = Arch::add(acc, acc3);
                        total += Arch::sum_reduce(acc).to_f64() as usize;
                        acc0 = Arch::zero();
                        acc1 = Arch::zero();
                        acc2 = Arch::zero();
                        acc3 = Arch::zero();
                    }
                    count = 0;
                }
            }

            unsafe {
                let mut acc = Arch::add(acc0, acc1);
                acc = Arch::add(acc, acc2);
                acc = Arch::add(acc, acc3);
                total += Arch::sum_reduce(acc).to_f64() as usize;
            }
        }

        if i < simd_len {
            let mut acc = unsafe { Arch::zero() };
            while i < simd_len {
                // SAFETY: `i < simd_len` bounds both `s.add(i)`/`o.add(i)` loads
                // to a `LANE_COUNT` window within the equal-length slices.
                unsafe {
                    let va = load(s.as_ptr().add(i));
                    let vb = load(o.as_ptr().add(i));
                    acc = Arch::add(acc, Arch::popcount(op.apply::<Arch>(va, vb)));
                }
                i += lane_count;
            }
            total += unsafe { Arch::sum_reduce(acc) }.to_f64() as usize;
        }

        // The integer count remains exact while the operation stays in the
        // provider's SIMD seam without accessing beyond either source slice.
        let tail = len - simd_len;
        if tail != 0 {
            // SAFETY: both slices contain the exact `tail` suffix and the
            // leading mask selects no lane beyond it.
            let tail_count = unsafe {
                let mask = Arch::leading_k_mask(tail);
                let combined = op.apply::<Arch>(
                    Arch::masked_load_partial(s.as_ptr().add(simd_len), tail, mask, Arch::zero()),
                    Arch::masked_load_partial(o.as_ptr().add(simd_len), tail, mask, Arch::zero()),
                );
                Arch::masked_sum_reduce(Arch::popcount(combined), mask).to_f64() as usize
            };
            total += tail_count;
        }

        Ok(total)
    }

    /// Computes the horizontal sum of population counts of `self[i] & other[i]`.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] if slice lengths differ.
    #[inline]
    pub fn reduce_popcount_and<ORef>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
    ) -> Result<usize, SimdError>
    where
        ORef: 'a,
    {
        self.reduce_popcount_op(other, crate::ops::BitAnd)
    }

    /// Computes the horizontal sum of population counts of `self[i] | other[i]`.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] if slice lengths differ.
    #[inline]
    pub fn reduce_popcount_or<ORef>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
    ) -> Result<usize, SimdError>
    where
        ORef: 'a,
    {
        self.reduce_popcount_op(other, crate::ops::BitOr)
    }

    /// Computes the horizontal sum of population counts of `self[i] ^ other[i]`.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] if slice lengths differ.
    #[inline]
    pub fn reduce_popcount_xor<ORef>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
    ) -> Result<usize, SimdError>
    where
        ORef: 'a,
    {
        self.reduce_popcount_op(other, crate::ops::BitXor)
    }
}
