use crate::arch::SimdArch;
use crate::align::Alignment;
use crate::kernel::SimdKernel;
use crate::execution::ExecutionMode;
use crate::scalar::Scalar;
use crate::ops::ReductionOp;
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
            if Align::IS_ALIGNED {
                unsafe { Arch::load_aligned(p) }
            } else {
                unsafe { Arch::load_unaligned(p) }
            }
        };

        // Initialize with identity vector so Min/Max start from the correct bound.
        let mut acc = unsafe { Op::identity_vector::<Arch>() };
        let mut i = 0usize;

        if unrolled_len >= chunk_size {
            let ptr = data.as_ptr();
            let mut acc0 = load(ptr);
            let mut acc1 = load(unsafe { ptr.add(lane_count) });
            let mut acc2 = load(unsafe { ptr.add(lane_count * 2) });
            let mut acc3 = load(unsafe { ptr.add(lane_count * 3) });
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

            acc0 = unsafe { Op::accumulate::<Arch>(acc0, acc1) };
            acc2 = unsafe { Op::accumulate::<Arch>(acc2, acc3) };
            acc = unsafe { Op::accumulate::<Arch>(acc0, acc2) };
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

        // Scalar tail — use Op::scalar_combine for correctness with Min/Max.
        while i < len {
            total = Op::scalar_combine(total, data[i]);
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
            if Align::IS_ALIGNED {
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
            let mut acc1 =
                pair(unsafe { pa.add(lane_count) }, unsafe { pb.add(lane_count) });
            let mut acc2 = pair(
                unsafe { pa.add(lane_count * 2) },
                unsafe { pb.add(lane_count * 2) },
            );
            let mut acc3 = pair(
                unsafe { pa.add(lane_count * 3) },
                unsafe { pb.add(lane_count * 3) },
            );
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
}

impl<'a, T: 'a, Arch: crate::arch::SimdArch + crate::kernel::SimdKernel<T>, Align: crate::align::Alignment, Mode: crate::execution::ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
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
        let idx = data.iter()
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
        let idx = data.iter()
            .position(|x| x.partial_cmp(&max_val) == Some(core::cmp::Ordering::Equal))?;
        Some((idx, max_val))
    }
}
