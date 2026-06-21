use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::ops::ReductionOp;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
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

        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        // Initialize with identity vector so Min/Max start from the correct bound.
        let mut acc = unsafe { Op::identity_vector::<Arch>() };
        let mut i = 0usize;

        if unrolled_len >= chunk_size {
            // Seeds carry the per-element transform (identity for Sum/Min/Max,
            // abs for AbsSum/AbsMax) — a raw-load seed would skip it for the
            // first chunk. Cross-accumulator merges use combine_vectors, which
            // never re-applies the transform to already-transformed partials.
            let ptr = data.as_ptr();
            let mut acc0 = unsafe { Op::transform_vector::<Arch>(load(ptr)) };
            let mut acc1 = unsafe { Op::transform_vector::<Arch>(load(ptr.add(lane_count))) };
            let mut acc2 = unsafe { Op::transform_vector::<Arch>(load(ptr.add(lane_count * 2))) };
            let mut acc3 = unsafe { Op::transform_vector::<Arch>(load(ptr.add(lane_count * 3))) };
            i = chunk_size;
            let mut ptr = unsafe { ptr.add(chunk_size) };

            while i < unrolled_len {
                acc0 = unsafe { Op::accumulate::<Arch>(acc0, load(ptr)) };
                acc1 = unsafe { Op::accumulate::<Arch>(acc1, load(ptr.add(lane_count))) };
                acc2 = unsafe { Op::accumulate::<Arch>(acc2, load(ptr.add(lane_count * 2))) };
                acc3 = unsafe { Op::accumulate::<Arch>(acc3, load(ptr.add(lane_count * 3))) };
                ptr = unsafe { ptr.add(chunk_size) };
                i += chunk_size;
            }

            acc0 = unsafe { Op::combine_vectors::<Arch>(acc0, acc1) };
            acc2 = unsafe { Op::combine_vectors::<Arch>(acc2, acc3) };
            acc = unsafe { Op::combine_vectors::<Arch>(acc0, acc2) };
        }

        // Remaining full SIMD vectors
        let simd_len = (len / lane_count) * lane_count;
        let ptr = data.as_ptr();
        while i < simd_len {
            let v = load(unsafe { ptr.add(i) });
            acc = unsafe { Op::accumulate::<Arch>(acc, v) };
            i += lane_count;
        }

        let mut total = unsafe { Op::finalize::<Arch>(acc) };

        // Scalar tail — use Op::scalar_accumulate so per-element transforms (e.g. SquaredSum)
        // apply correctly. For Sum/Min/Max the default delegates to scalar_combine.
        while i < len {
            total = Op::scalar_accumulate(total, data[i]);
            i += 1;
        }

        total
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

        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        let s = self.as_slice();
        let o = other.as_slice();
        let mut acc = unsafe { Op::identity_vector::<Arch>() };
        let mut i = 0usize;

        if unrolled_len >= chunk_size {
            let mut pa = s.as_ptr();
            let mut pb = o.as_ptr();

            let pair = |pa: *const T, pb: *const T| -> Arch::Vector {
                unsafe { Arch::mul(load(pa), load(pb)) }
            };

            let mut acc0 = pair(pa, pb);
            let mut acc1 = pair(unsafe { pa.add(lane_count) }, unsafe { pb.add(lane_count) });
            let mut acc2 = pair(unsafe { pa.add(lane_count * 2) }, unsafe {
                pb.add(lane_count * 2)
            });
            let mut acc3 = pair(unsafe { pa.add(lane_count * 3) }, unsafe {
                pb.add(lane_count * 3)
            });
            pa = unsafe { pa.add(chunk_size) };
            pb = unsafe { pb.add(chunk_size) };
            i = chunk_size;

            while i < unrolled_len {
                acc0 = unsafe { Op::accumulate::<Arch>(acc0, pair(pa, pb)) };
                acc1 = unsafe {
                    Op::accumulate::<Arch>(acc1, pair(pa.add(lane_count), pb.add(lane_count)))
                };
                acc2 = unsafe {
                    Op::accumulate::<Arch>(
                        acc2,
                        pair(pa.add(lane_count * 2), pb.add(lane_count * 2)),
                    )
                };
                acc3 = unsafe {
                    Op::accumulate::<Arch>(
                        acc3,
                        pair(pa.add(lane_count * 3), pb.add(lane_count * 3)),
                    )
                };
                pa = unsafe { pa.add(chunk_size) };
                pb = unsafe { pb.add(chunk_size) };
                i += chunk_size;
            }

            acc0 = unsafe { Op::accumulate::<Arch>(acc0, acc1) };
            acc2 = unsafe { Op::accumulate::<Arch>(acc2, acc3) };
            acc = unsafe { Op::accumulate::<Arch>(acc0, acc2) };
        }

        // Remaining full SIMD vectors
        let simd_len = (len / lane_count) * lane_count;
        let pa = s.as_ptr();
        let pb = o.as_ptr();
        while i < simd_len {
            let va = load(unsafe { pa.add(i) });
            let vb = load(unsafe { pb.add(i) });
            let prod = unsafe { Arch::mul(va, vb) };
            acc = unsafe { Op::accumulate::<Arch>(acc, prod) };
            i += lane_count;
        }

        let mut total = unsafe { Op::finalize::<Arch>(acc) };

        // Scalar tail — use scalar_combine for correctness with Min/Max.
        while i < len {
            total = Op::scalar_combine(total, s[i] * o[i]);
            i += 1;
        }

        Ok(total)
    }

    /// Computes the horizontal sum of population counts of all elements.
    #[inline]
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

        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        // Determine periodic reduction flush limit based on element size to prevent precision loss.
        // For f16/bf16/i16 (2 bytes), exact integer range limit is low (256/2048), so we reduce every 128 chunks.
        // For larger types, we can safely reduce every 32768 chunks.
        let flush_limit = if core::mem::size_of::<T>() == 2 {
            128
        } else {
            32768
        };

        // Unrolled loop (4-way register accumulation)
        if unrolled_simd_len > 0 {
            let mut acc0 = unsafe { Arch::zero() };
            let mut acc1 = unsafe { Arch::zero() };
            let mut acc2 = unsafe { Arch::zero() };
            let mut acc3 = unsafe { Arch::zero() };
            let mut count = 0;

            while i < unrolled_simd_len {
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
                unsafe {
                    let v = load(data.as_ptr().add(i));
                    acc = Arch::add(acc, Arch::popcount(v));
                }
                i += lane_count;
            }
            total += unsafe { Arch::sum_reduce(acc) }.to_f64() as usize;
        }

        // Scalar tail loop
        while i < len {
            total += data[i].count_ones() as usize;
            i += 1;
        }

        total
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

        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        let flush_limit = if core::mem::size_of::<T>() == 2 {
            128
        } else {
            32768
        };

        if unrolled_simd_len > 0 {
            let mut acc0 = unsafe { Arch::zero() };
            let mut acc1 = unsafe { Arch::zero() };
            let mut acc2 = unsafe { Arch::zero() };
            let mut acc3 = unsafe { Arch::zero() };
            let mut count = 0;

            while i < unrolled_simd_len {
                unsafe {
                    let va0 = load(s.as_ptr().add(i));
                    let vb0 = load(o.as_ptr().add(i));
                    let va1 = load(s.as_ptr().add(i + lane_count));
                    let vb1 = load(o.as_ptr().add(i + lane_count));
                    let va2 = load(s.as_ptr().add(i + lane_count * 2));
                    let vb2 = load(o.as_ptr().add(i + lane_count * 2));
                    let va3 = load(s.as_ptr().add(i + lane_count * 3));
                    let vb3 = load(o.as_ptr().add(i + lane_count * 3));

                    acc0 = Arch::add(acc0, Arch::popcount(Arch::bitand(va0, vb0)));
                    acc1 = Arch::add(acc1, Arch::popcount(Arch::bitand(va1, vb1)));
                    acc2 = Arch::add(acc2, Arch::popcount(Arch::bitand(va2, vb2)));
                    acc3 = Arch::add(acc3, Arch::popcount(Arch::bitand(va3, vb3)));
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
                unsafe {
                    let va = load(s.as_ptr().add(i));
                    let vb = load(o.as_ptr().add(i));
                    acc = Arch::add(acc, Arch::popcount(Arch::bitand(va, vb)));
                }
                i += lane_count;
            }
            total += unsafe { Arch::sum_reduce(acc) }.to_f64() as usize;
        }

        while i < len {
            total += s[i].bitand(o[i]).count_ones() as usize;
            i += 1;
        }

        Ok(total)
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

        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        let flush_limit = if core::mem::size_of::<T>() == 2 {
            128
        } else {
            32768
        };

        if unrolled_simd_len > 0 {
            let mut acc0 = unsafe { Arch::zero() };
            let mut acc1 = unsafe { Arch::zero() };
            let mut acc2 = unsafe { Arch::zero() };
            let mut acc3 = unsafe { Arch::zero() };
            let mut count = 0;

            while i < unrolled_simd_len {
                unsafe {
                    let va0 = load(s.as_ptr().add(i));
                    let vb0 = load(o.as_ptr().add(i));
                    let va1 = load(s.as_ptr().add(i + lane_count));
                    let vb1 = load(o.as_ptr().add(i + lane_count));
                    let va2 = load(s.as_ptr().add(i + lane_count * 2));
                    let vb2 = load(o.as_ptr().add(i + lane_count * 2));
                    let va3 = load(s.as_ptr().add(i + lane_count * 3));
                    let vb3 = load(o.as_ptr().add(i + lane_count * 3));

                    acc0 = Arch::add(acc0, Arch::popcount(Arch::bitor(va0, vb0)));
                    acc1 = Arch::add(acc1, Arch::popcount(Arch::bitor(va1, vb1)));
                    acc2 = Arch::add(acc2, Arch::popcount(Arch::bitor(va2, vb2)));
                    acc3 = Arch::add(acc3, Arch::popcount(Arch::bitor(va3, vb3)));
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
                unsafe {
                    let va = load(s.as_ptr().add(i));
                    let vb = load(o.as_ptr().add(i));
                    acc = Arch::add(acc, Arch::popcount(Arch::bitor(va, vb)));
                }
                i += lane_count;
            }
            total += unsafe { Arch::sum_reduce(acc) }.to_f64() as usize;
        }

        while i < len {
            total += s[i].bitor(o[i]).count_ones() as usize;
            i += 1;
        }

        Ok(total)
    }

    /// Computes the horizontal sum of population counts of `self[i] ^ other[i]` (Hamming distance).
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

        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        let flush_limit = if core::mem::size_of::<T>() == 2 {
            128
        } else {
            32768
        };

        if unrolled_simd_len > 0 {
            let mut acc0 = unsafe { Arch::zero() };
            let mut acc1 = unsafe { Arch::zero() };
            let mut acc2 = unsafe { Arch::zero() };
            let mut acc3 = unsafe { Arch::zero() };
            let mut count = 0;

            while i < unrolled_simd_len {
                unsafe {
                    let va0 = load(s.as_ptr().add(i));
                    let vb0 = load(o.as_ptr().add(i));
                    let va1 = load(s.as_ptr().add(i + lane_count));
                    let vb1 = load(o.as_ptr().add(i + lane_count));
                    let va2 = load(s.as_ptr().add(i + lane_count * 2));
                    let vb2 = load(o.as_ptr().add(i + lane_count * 2));
                    let va3 = load(s.as_ptr().add(i + lane_count * 3));
                    let vb3 = load(o.as_ptr().add(i + lane_count * 3));

                    acc0 = Arch::add(acc0, Arch::popcount(Arch::bitxor(va0, vb0)));
                    acc1 = Arch::add(acc1, Arch::popcount(Arch::bitxor(va1, vb1)));
                    acc2 = Arch::add(acc2, Arch::popcount(Arch::bitxor(va2, vb2)));
                    acc3 = Arch::add(acc3, Arch::popcount(Arch::bitxor(va3, vb3)));
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
                unsafe {
                    let va = load(s.as_ptr().add(i));
                    let vb = load(o.as_ptr().add(i));
                    acc = Arch::add(acc, Arch::popcount(Arch::bitxor(va, vb)));
                }
                i += lane_count;
            }
            total += unsafe { Arch::sum_reduce(acc) }.to_f64() as usize;
        }

        while i < len {
            total += s[i].bitxor(o[i]).count_ones() as usize;
            i += 1;
        }

        Ok(total)
    }
}

impl<
        'a,
        T: 'a,
        Arch: crate::arch::SimdArch + crate::kernel::SimdKernel<T>,
        Align: crate::align::Alignment,
        Mode: crate::execution::ExecutionMode,
        Ref: 'a,
    > SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: crate::scalar::Scalar + crate::scalar::NumericElement,
{
    /// Returns `Some((index, value))` of the minimum element, or `None` for an empty slice.
    ///
    /// Correctness: uses a SIMD-accelerated reduction pass to find the minimum value,
    /// followed by a linear scan to locate the first occurrence of that value.
    #[inline]
    pub fn argmin(&self) -> Option<(usize, T)> {
        let data = self.as_slice();
        if data.is_empty() {
            return None;
        }
        let min_val = self.reduce(crate::ops::Min);
        let idx = data
            .iter()
            .position(|x| x.partial_cmp(&min_val) == Some(core::cmp::Ordering::Equal))?;
        Some((idx, min_val))
    }

    /// Returns `Some((index, value))` of the maximum element, or `None` for an empty slice.
    ///
    /// Correctness: uses a SIMD-accelerated reduction pass to find the maximum value,
    /// followed by a linear scan to locate the first occurrence of that value.
    #[inline]
    pub fn argmax(&self) -> Option<(usize, T)> {
        let data = self.as_slice();
        if data.is_empty() {
            return None;
        }
        let max_val = self.reduce(crate::ops::Max);
        let idx = data
            .iter()
            .position(|x| x.partial_cmp(&max_val) == Some(core::cmp::Ordering::Equal))?;
        Some((idx, max_val))
    }
}
