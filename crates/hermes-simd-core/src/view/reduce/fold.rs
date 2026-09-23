//! Generic horizontal reductions driven by a `ReductionOp` ZST.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::{SimdArith, SimdCompare, SimdLoadStore, SimdMask, SimdReduce};
use crate::ops::ReductionOp;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<
        'a,
        T: 'a,
        Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdCompare<T> + SimdMask<T> + SimdReduce<T>,
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
            // first chunk — and fold it into the identity through `accumulate`
            // so the first chunk passes the same lane guard as every later one:
            // Min/Max ignore NaN lanes only inside `accumulate`, and a raw seed
            // let a NaN in the first chunk poison its accumulator (NEON `vminq`
            // propagates it; x86 `min` only drops it by operand order).
            // Cross-accumulator merges use combine_vectors, which never
            // re-applies the transform to already-transformed partials.
            let base = data.as_ptr();
            acc = unsafe {
                let mut acc0 =
                    Op::accumulate::<Arch>(acc, Op::transform_vector::<Arch>(load(base)));
                let mut acc1 = Op::accumulate::<Arch>(
                    acc,
                    Op::transform_vector::<Arch>(load(base.add(lane_count))),
                );
                let mut acc2 = Op::accumulate::<Arch>(
                    acc,
                    Op::transform_vector::<Arch>(load(base.add(lane_count * 2))),
                );
                let mut acc3 = Op::accumulate::<Arch>(
                    acc,
                    Op::transform_vector::<Arch>(load(base.add(lane_count * 3))),
                );
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
            // SAFETY: `data` contains the exact `tail` suffix and the leading
            // mask selects no lane beyond it.
            let tail_value = unsafe {
                let mask = Arch::leading_k_mask(tail);
                let value = Arch::masked_load_partial(
                    data.as_ptr().add(simd_len),
                    tail,
                    mask,
                    Arch::zero(),
                );
                Op::masked_finalize::<Arch>(value, mask)
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
        crate::view::check_lengths_equal(self.len(), other.len())?;
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

        // The reduction strategy applies its own transform and identity, so the
        // partial pair remains correct for every operation that opts into
        // masked tails without accessing past either slice.
        let tail = len - simd_len;
        if tail != 0 && Op::USE_MASKED_TAIL {
            // SAFETY: both slices contain the exact `tail` suffix and the
            // leading mask selects no lane beyond it.
            let tail_value = unsafe {
                let mask = Arch::leading_k_mask(tail);
                let pair = Arch::mul(
                    Arch::masked_load_partial(s.as_ptr().add(simd_len), tail, mask, Arch::zero()),
                    Arch::masked_load_partial(o.as_ptr().add(simd_len), tail, mask, Arch::zero()),
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
}
