//! Horizontal and pairwise SIMD reductions over [`SimdView`](crate::view::SimdView).
//!
//! # Safety
//!
//! Every kernel call below is `#[target_feature]`-gated and is therefore sound
//! only on a host implementing `Arch`. That holds by construction rather than by
//! inspection: [`SimdView::new`](crate::view::SimdView::new) returns `None` for
//! an architecture the host cannot execute, and the sparse and copy-on-write
//! constructors assert the same condition, so possessing one of these
//! arch-parameterized values *is* the proof. Per-site `SAFETY` comments record
//! only the obligations that go beyond it — pointer provenance, bounds, and
//! alignment.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::{SimdArith, SimdBitwise, SimdCompare, SimdLoadStore, SimdMask, SimdReduce};
use crate::ops::ReductionOp;
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
    /// Generic SIMD horizontal reduction using a `ReductionOp<T>` strategy ZST.
    ///
    /// Processes `UNROLL_FACTOR × LANE_COUNT` elements per iteration using
    /// `UNROLL_FACTOR` independent accumulators to saturate FMA throughput.
    ///
    /// The vector accumulator is initialized to `Op::identity_vector()` — the
    /// identity element for this reduction (e.g. `+∞` for `Min`, `-∞` for `Max`,
    /// `0` for `Sum`). This is required for correctness: starting from `Arch::zero()`
    /// would produce wrong results for `Min`/`Max` on non-negative inputs.
    ///
    /// Zero-cost: `_op` is a ZST erased entirely by the compiler.
    #[inline]
    pub fn reduce<Op: ReductionOp<T>>(&self, _op: Op) -> T {
        let data = self.as_slice();
        let len = data.len();
        if len == 0 {
            return Op::identity_scalar();
        }

        let lane_count = Arch::LANE_COUNT;
        let unroll_factor = Arch::UNROLL_FACTOR;
        let chunk_size = lane_count * unroll_factor;
        let unrolled_len = (len / chunk_size) * chunk_size;

        // SAFETY: `Arch::load_*` is a target-feature kernel (module invariant).
        // Every caller only ever passes a pointer whose `LANE_COUNT`-element read
        // stays within `data` (offsets are bounded by `simd_len`/`unrolled_len`),
        // and the aligned variant is selected only when `Align` proves the base
        // is arch-aligned.
        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        // SAFETY: the `Op::*` and `identity_vector` calls are target-feature
        // kernels covered by the module invariant. `unrolled_len` is a multiple
        // of `chunk_size = LANE_COUNT * UNROLL_FACTOR`, so each `ptr.add(k)` in
        // the seeds/loop addresses a `LANE_COUNT` window fully within `data`
        // (`ptr` advances by `chunk_size` per iteration while `i < unrolled_len`).
        // Initialize with the identity vector so Min/Max start from the correct bound.
        let mut acc = unsafe { Op::identity_vector::<Arch>() };
        let mut i = 0usize;

        if unrolled_len >= chunk_size {
            // Seeds carry the per-element transform (identity for Sum/Min/Max,
            // abs for AbsSum/AbsMax) — a raw-load seed would skip it for the
            // first chunk. Cross-accumulator merges use combine_vectors, which
            // never re-applies the transform to already-transformed partials.
            let base = data.as_ptr();
            acc = unsafe {
                let mut acc0 = Op::transform_vector::<Arch>(load(base));
                let mut acc1 = Op::transform_vector::<Arch>(load(base.add(lane_count)));
                let mut acc2 = Op::transform_vector::<Arch>(load(base.add(lane_count * 2)));
                let mut acc3 = Op::transform_vector::<Arch>(load(base.add(lane_count * 3)));
                let mut ptr = base.add(chunk_size);
                i = chunk_size;

                while i < unrolled_len {
                    acc0 = Op::accumulate::<Arch>(acc0, load(ptr));
                    acc1 = Op::accumulate::<Arch>(acc1, load(ptr.add(lane_count)));
                    acc2 = Op::accumulate::<Arch>(acc2, load(ptr.add(lane_count * 2)));
                    acc3 = Op::accumulate::<Arch>(acc3, load(ptr.add(lane_count * 3)));
                    ptr = ptr.add(chunk_size);
                    i += chunk_size;
                }

                acc0 = Op::combine_vectors::<Arch>(acc0, acc1);
                acc2 = Op::combine_vectors::<Arch>(acc2, acc3);
                Op::combine_vectors::<Arch>(acc0, acc2)
            };
        }

        // Remaining full SIMD vectors.
        // SAFETY: `i < simd_len` and `simd_len = (len / LANE_COUNT) * LANE_COUNT`,
        // so `ptr.add(i)` addresses a `LANE_COUNT` window within `data`;
        // `Op::accumulate`/`finalize` are target-feature kernels (module invariant).
        let simd_len = (len / lane_count) * lane_count;
        let ptr = data.as_ptr();
        let mut total = unsafe {
            while i < simd_len {
                acc = Op::accumulate::<Arch>(acc, load(ptr.add(i)));
                i += lane_count;
            }
            Op::finalize::<Arch>(acc)
        };

        // Final partial vector. Transform-bearing reductions with a neutral
        // identity use the provider masked-reduction seam; other operations keep
        // their established scalar-tail contract until their ordering semantics
        // receive a dedicated proof. The masked reduction may change floating-
        // point grouping within this final vector, so callers must use the
        // reduction's documented numerical-order envelope rather than assume a
        // scalar left fold for this opt-in family.
        let tail = len - simd_len;
        if tail != 0 && Op::USE_MASKED_TAIL {
            const { Arch::LANE_BOUND_CHECK };
            let mut lanes = [T::ZERO; crate::kernel::MAX_SIMD_LANES];
            lanes[..tail].copy_from_slice(&data[simd_len..simd_len + tail]);
            let tail_value = unsafe {
                let mask = Arch::leading_k_mask(tail);
                Op::masked_finalize::<Arch>(Arch::load_unaligned(lanes.as_ptr()), mask)
            };
            total = Op::scalar_combine(total, tail_value);
        } else {
            while i < len {
                total = Op::scalar_accumulate(total, data[i]);
                i += 1;
            }
        }

        total
    }

    /// Sums all elements in the view through the reduction facet contract.
    #[inline(always)]
    #[must_use]
    pub fn sum(&self) -> T {
        self.reduce(crate::ops::Sum)
    }

    /// Generic pairwise SIMD reduction: `reduce(Op, a ⊗ b)`.
    ///
    /// Computes `a[i] * b[i]` lane-wise, then applies `Op::accumulate` and `Op::finalize`.
    /// For `Op=Dot` this is the standard dot product.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] if slice lengths differ.
    #[inline]
    pub fn zip_reduce<Op: ReductionOp<T>, ORef>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        _op: Op,
    ) -> Result<T, SimdError>
    where
        ORef: 'a,
    {
        super::check_lengths_equal(self.len(), other.len())?;
        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let unroll_factor = Arch::UNROLL_FACTOR;
        let chunk_size = lane_count * unroll_factor;
        let unrolled_len = (len / chunk_size) * chunk_size;

        // SAFETY: identical contract to `reduce`'s `load` — target-feature kernel
        // (module invariant), and every call passes a pointer whose `LANE_COUNT`
        // read stays within its slice (offsets bounded by `simd_len`).
        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        let s = self.as_slice();
        let o = other.as_slice();
        // SAFETY: target-feature kernels (module invariant). `s` and `o` are
        // equal length (checked above), and `unrolled_len` is a multiple of
        // `chunk_size`, so every `pa.add(k)`/`pb.add(k)` addresses a `LANE_COUNT`
        // window within its slice while `i < unrolled_len`.
        let mut acc = unsafe { Op::identity_vector::<Arch>() };
        let mut i = 0usize;

        if unrolled_len >= chunk_size {
            // Seed the four accumulators with the first pairwise products.
            // (First chunk cannot use FMA into zero, so we use separate mul.)
            acc = unsafe {
                let pair =
                    |pa: *const T, pb: *const T| -> Arch::Vector { Arch::mul(load(pa), load(pb)) };
                let base_a = s.as_ptr();
                let base_b = o.as_ptr();

                let mut acc0 = pair(base_a, base_b);
                let mut acc1 = pair(base_a.add(lane_count), base_b.add(lane_count));
                let mut acc2 = pair(base_a.add(lane_count * 2), base_b.add(lane_count * 2));
                let mut acc3 = pair(base_a.add(lane_count * 3), base_b.add(lane_count * 3));
                let mut pa = base_a.add(chunk_size);
                let mut pb = base_b.add(chunk_size);
                i = chunk_size;

                // Main unrolled loop — `fma_pair_accumulate` lets `Dot` emit a
                // single `vfmadd` instead of a separate `mul` + `add`.
                while i < unrolled_len {
                    acc0 = Op::fma_pair_accumulate::<Arch>(acc0, load(pa), load(pb));
                    acc1 = Op::fma_pair_accumulate::<Arch>(
                        acc1,
                        load(pa.add(lane_count)),
                        load(pb.add(lane_count)),
                    );
                    acc2 = Op::fma_pair_accumulate::<Arch>(
                        acc2,
                        load(pa.add(lane_count * 2)),
                        load(pb.add(lane_count * 2)),
                    );
                    acc3 = Op::fma_pair_accumulate::<Arch>(
                        acc3,
                        load(pa.add(lane_count * 3)),
                        load(pb.add(lane_count * 3)),
                    );
                    pa = pa.add(chunk_size);
                    pb = pb.add(chunk_size);
                    i += chunk_size;
                }

                acc0 = Op::accumulate::<Arch>(acc0, acc1);
                acc2 = Op::accumulate::<Arch>(acc2, acc3);
                Op::accumulate::<Arch>(acc0, acc2)
            };
        }

        // Remaining full SIMD vectors — use `fma_pair_accumulate` here too.
        // SAFETY: `i < simd_len` bounds each `pa.add(i)`/`pb.add(i)` to a
        // `LANE_COUNT` window within the equal-length slices; kernels covered by
        // the module invariant.
        let simd_len = (len / lane_count) * lane_count;
        let pa = s.as_ptr();
        let pb = o.as_ptr();
        let mut total = unsafe {
            while i < simd_len {
                acc = Op::fma_pair_accumulate::<Arch>(acc, load(pa.add(i)), load(pb.add(i)));
                i += lane_count;
            }
            Op::finalize::<Arch>(acc)
        };

        // Pairwise final vector. Both buffers are fully initialized because the
        // provider's masked-memory contract still requires a valid full-width
        // load even when inactive lanes are discarded by `masked_finalize`.
        // The reduction strategy applies its own transform and identity, so this
        // remains correct for every operation that opts into masked tails.
        let tail = len - simd_len;
        if tail != 0 && Op::USE_MASKED_TAIL {
            const { Arch::LANE_BOUND_CHECK };
            let mut left = [T::ZERO; crate::kernel::MAX_SIMD_LANES];
            let mut right = [T::ZERO; crate::kernel::MAX_SIMD_LANES];
            left[..tail].copy_from_slice(&s[simd_len..simd_len + tail]);
            right[..tail].copy_from_slice(&o[simd_len..simd_len + tail]);
            let tail_value = unsafe {
                let mask = Arch::leading_k_mask(tail);
                let pair = Arch::mul(
                    Arch::load_unaligned(left.as_ptr()),
                    Arch::load_unaligned(right.as_ptr()),
                );
                Op::masked_finalize::<Arch>(pair, mask)
            };
            total = Op::scalar_combine(total, tail_value);
        } else {
            // Product and any future strategy that does not opt into masked
            // tails retain the scalar pairwise contract.
            while i < len {
                total = Op::scalar_combine(total, s[i] * o[i]);
                i += 1;
            }
        }

        Ok(total)
    }

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
            const { Arch::LANE_BOUND_CHECK };
            let mut lanes = [T::ZERO; crate::kernel::MAX_SIMD_LANES];
            lanes[..tail].copy_from_slice(&data[simd_len..simd_len + tail]);
            let tail_count = unsafe {
                let mask = Arch::leading_k_mask(tail);
                Arch::masked_sum_reduce(Arch::popcount(Arch::load_unaligned(lanes.as_ptr())), mask)
                    .to_f64() as usize
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
        super::check_lengths_equal(self.len(), other.len())?;
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

        // Masked final vector. Both source buffers are initialized before the
        // full-width loads required by blend-based backends; the integer count
        // remains exact while the operation stays in the provider's SIMD seam.
        let tail = len - simd_len;
        if tail != 0 {
            const { Arch::LANE_BOUND_CHECK };
            let mut left = [T::ZERO; crate::kernel::MAX_SIMD_LANES];
            let mut right = [T::ZERO; crate::kernel::MAX_SIMD_LANES];
            left[..tail].copy_from_slice(&s[simd_len..simd_len + tail]);
            right[..tail].copy_from_slice(&o[simd_len..simd_len + tail]);
            let tail_count = unsafe {
                let mask = Arch::leading_k_mask(tail);
                let combined = op.apply::<Arch>(
                    Arch::load_unaligned(left.as_ptr()),
                    Arch::load_unaligned(right.as_ptr()),
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

impl<
        'a,
        T: 'a,
        Arch: crate::arch::SimdArch
            + crate::kernel::SimdLoadStore<T>
            + crate::kernel::SimdArith<T>
            + crate::kernel::SimdBitwise<T>
            + crate::kernel::SimdCompare<T>
            + crate::kernel::SimdMask<T>
            + crate::kernel::SimdReduce<T>,
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
        Ref: 'a,
    > SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: crate::scalar::Scalar + crate::scalar::NumericElement,
{
    /// Returns `Some((index, value))` for the first minimum element.
    ///
    /// Correctness: a SIMD reduction pass finds the minimum value, then one
    /// validation scan rejects NaNs while retaining its first occurrence.
    ///
    /// Returns `None` for an empty slice or when any element is NaN. The
    /// validation scan rejects the whole unordered domain, so an intermediate
    /// backend result never escapes. Equal extrema use the first slice element,
    /// including its signed-zero representation.
    #[inline]
    #[must_use]
    pub fn argmin(&self) -> Option<(usize, T)> {
        let data = self.as_slice();
        if data.is_empty() {
            return None;
        }
        let min_val = self.reduce(crate::ops::Min);
        Self::locate_ordered_extremum(data, min_val)
    }

    /// Returns `Some((index, value))` for the first maximum element.
    ///
    /// Correctness: a SIMD reduction pass finds the maximum value, then one
    /// validation scan rejects NaNs while retaining its first occurrence.
    ///
    /// Returns `None` for an empty slice or when any element is NaN. The
    /// validation scan rejects the whole unordered domain, so an intermediate
    /// backend result never escapes. Equal extrema use the first slice element,
    /// including its signed-zero representation.
    #[inline]
    #[must_use]
    pub fn argmax(&self) -> Option<(usize, T)> {
        let data = self.as_slice();
        if data.is_empty() {
            return None;
        }
        let max_val = self.reduce(crate::ops::Max);
        Self::locate_ordered_extremum(data, max_val)
    }

    #[inline]
    fn locate_ordered_extremum(data: &[T], extremum: T) -> Option<(usize, T)> {
        let lane_count = Arch::LANE_COUNT;
        // Shift-based construction avoids the `1 << 64` overflow a 64-lane
        // backend would hit; `lane_count` never exceeds `u64::BITS`.
        let lane_mask = u64::MAX >> (u64::BITS as usize - lane_count.min(64));
        let vector_len = (data.len() / lane_count) * lane_count;
        let mut first: Option<usize> = None;
        let mut index = 0usize;

        while index < vector_len {
            // SAFETY: `index <= vector_len - lane_count`, so the load reads
            // exactly `lane_count` in-bounds elements of `data`; the aligned
            // variant is selected only when `Align` guarantees the view's base
            // pointer is arch-aligned, and `index` is a multiple of `lane_count`.
            // Constructing `Arch` already asserts its target features.
            let (ordered, hits) = unsafe {
                let ptr = data.as_ptr().add(index);
                let v = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(ptr)
                } else {
                    Arch::load_unaligned(ptr)
                };
                // `x == x` is false exactly for NaN, so a lane absent from
                // `ordered` marks a NaN.
                let ordered =
                    Arch::mask_to_bitmask(Arch::vector_to_mask(Arch::cmp_eq(v, v))) & lane_mask;
                let hits = if first.is_none() {
                    let target = Arch::splat(extremum);
                    Arch::mask_to_bitmask(Arch::vector_to_mask(Arch::cmp_eq(v, target))) & lane_mask
                } else {
                    0
                };
                (ordered, hits)
            };

            if ordered != lane_mask {
                return None;
            }
            if hits != 0 {
                first = Some(index + hits.trailing_zeros() as usize);
            }
            index += lane_count;
        }

        for (offset, value) in data[index..].iter().copied().enumerate() {
            if value.is_nan() {
                return None;
            }
            if first.is_none() && value.partial_cmp(&extremum) == Some(core::cmp::Ordering::Equal) {
                first = Some(index + offset);
            }
        }

        // Report the stored element rather than the reduced extremum so equal
        // values keep their own representation, notably signed zero.
        first.map(|at| (at, data[at]))
    }
}
