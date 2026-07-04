use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::ops::ElementOp;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

/// Output-size threshold (bytes) at or above which [`SimdView::zip_into`]
/// switches to non-temporal (cache-bypassing) stores on backends that support
/// them.
///
/// Set to 8 MiB — past every consumer L2 — so streaming engages only for
/// outputs large enough that the read-for-ownership it avoids is not offset by
/// lost cache residency (a normal store would keep a smaller result hot for
/// reuse). Measured 1.71× at 64 MiB out-of-LLC (see `streaming_bench`).
const NT_STORE_MIN_BYTES: usize = 8 * 1024 * 1024;

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Sum all elements in the view.
    ///
    /// Iterates in unrolled chunks of `Arch::LANE_COUNT * Arch::UNROLL_FACTOR` elements,
    /// accumulating into multiple registers in parallel to break loop dependencies.
    #[inline(always)]
    pub fn sum(&self) -> T {
        let data = self.as_slice();
        let len = data.len();
        let lane_count = Arch::LANE_COUNT;
        let unroll_factor = Arch::UNROLL_FACTOR;
        let chunk_size = lane_count * unroll_factor;
        let unrolled_simd_len = (len / chunk_size) * chunk_size;
        let simd_len = (len / lane_count) * lane_count;
        let mut ptr = data.as_ptr();

        let accumulator = unsafe {
            if unrolled_simd_len > 0 {
                let load = |p| {
                    if crate::align::is_aligned_for_arch::<Arch, Align>() {
                        Arch::load_aligned(p)
                    } else {
                        Arch::load_unaligned(p)
                    }
                };

                let mut acc0 = load(ptr);
                let mut acc1 = load(ptr.add(lane_count));
                let mut acc2 = load(ptr.add(lane_count * 2));
                let mut acc3 = load(ptr.add(lane_count * 3));
                ptr = ptr.add(chunk_size);

                for _ in 1..(unrolled_simd_len / chunk_size) {
                    let v0 = load(ptr);
                    let v1 = load(ptr.add(lane_count));
                    let v2 = load(ptr.add(lane_count * 2));
                    let v3 = load(ptr.add(lane_count * 3));

                    acc0 = Arch::add(acc0, v0);
                    acc1 = Arch::add(acc1, v1);
                    acc2 = Arch::add(acc2, v2);
                    acc3 = Arch::add(acc3, v3);

                    ptr = ptr.add(chunk_size);
                }

                let mut acc = Arch::add(acc0, acc1);
                acc = Arch::add(acc, acc2);
                acc = Arch::add(acc, acc3);
                Some(acc)
            } else {
                None
            }
        };

        let mut acc = if let Some(a) = accumulator {
            a
        } else {
            unsafe { Arch::zero() }
        };

        // Middle SIMD loop for elements that didn't fit into the unrolled loop
        unsafe {
            let mut middle_ptr = data.as_ptr().add(unrolled_simd_len);
            for _ in 0..((simd_len - unrolled_simd_len) / lane_count) {
                let val = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(middle_ptr)
                } else {
                    Arch::load_unaligned(middle_ptr)
                };
                acc = Arch::add(acc, val);
                middle_ptr = middle_ptr.add(lane_count);
            }
        }

        let mut total = unsafe { Arch::sum_reduce(acc) };

        // Scalar tail loop
        for i in simd_len..len {
            total += data[i];
        }

        total
    }

    /// Compute the dot product between this view and another view of the same architecture and alignment.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if the view lengths are not identical.
    #[inline(always)]
    pub fn dot<ORef>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
    ) -> Result<T, SimdError>
    where
        ORef: 'a,
    {
        super::check_lengths_equal(self.len(), other.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let unroll_factor = Arch::UNROLL_FACTOR;
        let chunk_size = lane_count * unroll_factor;
        let unrolled_simd_len = (len / chunk_size) * chunk_size;

        let mut ptr1 = self.as_slice().as_ptr();
        let mut ptr2 = other.as_slice().as_ptr();

        let accumulator = unsafe {
            if unrolled_simd_len > 0 {
                let load = |p| {
                    if crate::align::is_aligned_for_arch::<Arch, Align>() {
                        Arch::load_aligned(p)
                    } else {
                        Arch::load_unaligned(p)
                    }
                };

                let v0_1 = load(ptr1);
                let v0_2 = load(ptr2);
                let mut acc0 = Arch::mul(v0_1, v0_2);

                let v1_1 = load(ptr1.add(lane_count));
                let v1_2 = load(ptr2.add(lane_count));
                let mut acc1 = Arch::mul(v1_1, v1_2);

                let v2_1 = load(ptr1.add(lane_count * 2));
                let v2_2 = load(ptr2.add(lane_count * 2));
                let mut acc2 = Arch::mul(v2_1, v2_2);

                let v3_1 = load(ptr1.add(lane_count * 3));
                let v3_2 = load(ptr2.add(lane_count * 3));
                let mut acc3 = Arch::mul(v3_1, v3_2);

                ptr1 = ptr1.add(chunk_size);
                ptr2 = ptr2.add(chunk_size);

                for _ in 1..(unrolled_simd_len / chunk_size) {
                    let v0_1 = load(ptr1);
                    let v0_2 = load(ptr2);
                    acc0 = Arch::fmadd(v0_1, v0_2, acc0);

                    let v1_1 = load(ptr1.add(lane_count));
                    let v1_2 = load(ptr2.add(lane_count));
                    acc1 = Arch::fmadd(v1_1, v1_2, acc1);

                    let v2_1 = load(ptr1.add(lane_count * 2));
                    let v2_2 = load(ptr2.add(lane_count * 2));
                    acc2 = Arch::fmadd(v2_1, v2_2, acc2);

                    let v3_1 = load(ptr1.add(lane_count * 3));
                    let v3_2 = load(ptr2.add(lane_count * 3));
                    acc3 = Arch::fmadd(v3_1, v3_2, acc3);

                    ptr1 = ptr1.add(chunk_size);
                    ptr2 = ptr2.add(chunk_size);
                }

                let mut acc = Arch::add(acc0, acc1);
                acc = Arch::add(acc, acc2);
                acc = Arch::add(acc, acc3);
                Some(acc)
            } else {
                None
            }
        };

        let simd_len = (len / lane_count) * lane_count;

        // Middle SIMD loop for elements that didn't fit into the unrolled loop.
        // Continue accumulating into the *vector* register via `fmadd` and reduce
        // to scalar ONCE at the end — rather than a horizontal `sum_reduce` per
        // lane group (which serialized the loop on the ~5-7-cycle reduction
        // latency and dominated small/odd-length dots, e.g. the bidiagonal-SVD
        // reflector applies).
        let mut acc_vec = accumulator;
        unsafe {
            let load = |p| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let mut middle_ptr1 = self.as_slice().as_ptr().add(unrolled_simd_len);
            let mut middle_ptr2 = other.as_slice().as_ptr().add(unrolled_simd_len);
            for _ in 0..((simd_len - unrolled_simd_len) / lane_count) {
                let v1 = load(middle_ptr1);
                let v2 = load(middle_ptr2);
                acc_vec = Some(match acc_vec {
                    Some(a) => Arch::fmadd(v1, v2, a),
                    None => Arch::mul(v1, v2),
                });
                middle_ptr1 = middle_ptr1.add(lane_count);
                middle_ptr2 = middle_ptr2.add(lane_count);
            }
        }
        let mut total = match acc_vec {
            Some(acc) => unsafe { Arch::sum_reduce(acc) },
            None => T::ZERO,
        };

        // Scalar tail loop
        let s_slice = self.as_slice();
        let o_slice = other.as_slice();
        for i in simd_len..len {
            total += s_slice[i] * o_slice[i];
        }

        Ok(total)
    }

    /// Multiply elementwise with another view and write the output to a mutable slice.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if operand lengths do not match, or
    /// `SimdError::InsufficientOutputLength` if the output slice is smaller than the input view.
    #[inline(always)]
    pub fn elementwise_mul<ORef>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        out: &mut [T],
    ) -> Result<(), SimdError>
    where
        ORef: 'a,
    {
        super::check_lengths_equal(self.len(), other.len())?;
        super::check_output_length(self.len(), out.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let mut ptr1 = self.as_slice().as_ptr();
        let mut ptr2 = other.as_slice().as_ptr();
        let mut ptr_out = out.as_mut_ptr();

        unsafe {
            let load = |p| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };

            let store = |p, val| {
                let is_out_aligned = crate::align::is_aligned_for_arch::<Arch, Align>()
                    && (p as usize) % Align::ALIGN_BYTES == 0;

                if is_out_aligned {
                    Arch::store_aligned(p, val);
                } else {
                    Arch::store_unaligned(p, val);
                }
            };

            for _ in 0..(simd_len / lane_count) {
                let v1 = load(ptr1);
                let v2 = load(ptr2);
                let res = Arch::mul(v1, v2);
                store(ptr_out, res);

                ptr1 = ptr1.add(lane_count);
                ptr2 = ptr2.add(lane_count);
                ptr_out = ptr_out.add(lane_count);
            }
        }

        let s_slice = self.as_slice();
        let o_slice = other.as_slice();
        for i in simd_len..len {
            out[i] = s_slice[i] * o_slice[i];
        }

        Ok(())
    }

    /// Pairwise elementwise operation on `self` and `other`, writing results to `out`.
    ///
    /// The SIMD vectorized loop covers `floor(len / LANE_COUNT) * LANE_COUNT` elements.
    /// The scalar tail handles the remaining elements element-by-element.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if operand lengths do not match, or
    /// `SimdError::InsufficientOutputLength` if `out.len() < self.len()`.
    #[inline(always)]
    pub fn zip_into<ORef, Op>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        out: &mut [T],
        op: Op,
    ) -> Result<(), SimdError>
    where
        ORef: 'a,
        Op: ElementOp<T>,
    {
        super::check_lengths_equal(self.len(), other.len())?;
        super::check_output_length(self.len(), out.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        // Route large write-only outputs through non-temporal stores: the write
        // bypasses the cache, avoiding the read-for-ownership (write-allocate)
        // traffic that dominates an out-of-LLC elementwise write (measured 1.71×
        // on AVX2 f32; see `streaming_bench`). Gated so it engages only when the
        // output clearly exceeds cache — below that, the RFO the NT store avoids
        // is offset by the cache residency a normal store would keep, so the
        // conservative path is a net win or wash and never a regression.
        if Arch::SUPPORTS_NT_STORE
            && len.saturating_mul(core::mem::size_of::<T>()) >= NT_STORE_MIN_BYTES
        {
            // SAFETY: lengths validated above; `zip_into_streaming` peels the
            // output to the NT-store alignment and issues the write barrier.
            return unsafe { self.zip_into_streaming(other, out, op, len, simd_len) };
        }

        let ptr_self = self.as_slice().as_ptr();
        let ptr_other = other.as_slice().as_ptr();
        let ptr_out = out.as_mut_ptr();

        unsafe {
            let load = |p| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };

            let store = |p, val| {
                let is_out_aligned = crate::align::is_aligned_for_arch::<Arch, Align>()
                    && (p as usize) % Align::ALIGN_BYTES == 0;
                if is_out_aligned {
                    Arch::store_aligned(p, val);
                } else {
                    Arch::store_unaligned(p, val);
                }
            };

            for i in (0..simd_len).step_by(lane_count) {
                let va = load(ptr_self.add(i));
                let vb = load(ptr_other.add(i));
                let vr = op.apply::<Arch>(va, vb);
                store(ptr_out.add(i), vr);
            }
        }

        let s_slice = self.as_slice();
        let o_slice = other.as_slice();
        for i in simd_len..len {
            out[i] = op.apply_scalar(s_slice[i], o_slice[i]);
        }

        Ok(())
    }

    /// Non-temporal (cache-bypassing) variant of the [`zip_into`](Self::zip_into)
    /// store loop for out-of-LLC outputs. The result is **byte-identical** to the
    /// regular path — only the store instruction changes, not the arithmetic.
    ///
    /// `out` is prefix-peeled to `LANE_COUNT · size_of::<T>()`-byte alignment
    /// (NT stores fault otherwise) with scalar ops, the aligned middle is
    /// streamed, the tail is scalar, and [`stream_write_barrier`] orders the
    /// weakly ordered stores before the caller reads `out`.
    ///
    /// # Safety
    /// `Arch::SUPPORTS_NT_STORE` must hold; `self`/`other`/`out` share `len`
    /// (validated by the caller); `simd_len == (len / LANE_COUNT) · LANE_COUNT`.
    ///
    /// [`stream_write_barrier`]: crate::kernel::SimdKernel::stream_write_barrier
    #[inline]
    unsafe fn zip_into_streaming<ORef, Op>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        out: &mut [T],
        op: Op,
        len: usize,
        _simd_len: usize,
    ) -> Result<(), SimdError>
    where
        ORef: 'a,
        Op: ElementOp<T>,
    {
        let lane_count = Arch::LANE_COUNT;
        let s = self.as_slice();
        let o = other.as_slice();
        let ptr_self = s.as_ptr();
        let ptr_other = o.as_ptr();
        let ptr_out = out.as_mut_ptr();

        // Elements to peel so the streamed region starts on a
        // `LANE_COUNT · size_of::<T>()` boundary. Slices are aligned to at least
        // `size_of::<T>()`, so `addr % align_bytes` is a whole number of
        // elements and the division is exact.
        let align_bytes = lane_count * core::mem::size_of::<T>();
        let addr = ptr_out as usize;
        let head = ((align_bytes - (addr % align_bytes)) % align_bytes) / core::mem::size_of::<T>();
        let head = head.min(len);

        for i in 0..head {
            out[i] = op.apply_scalar(s[i], o[i]);
        }

        let mid_end = head + ((len - head) / lane_count) * lane_count;
        let mut i = head;
        while i < mid_end {
            // SAFETY: `i < mid_end ≤ len`; loads are unaligned; the store target
            // `ptr_out + i` is aligned to `align_bytes` by construction of `head`.
            let va = Arch::load_unaligned(ptr_self.add(i));
            let vb = Arch::load_unaligned(ptr_other.add(i));
            let vr = op.apply::<Arch>(va, vb);
            Arch::store_streaming(ptr_out.add(i), vr);
            i += lane_count;
        }

        Arch::stream_write_barrier();

        for i in mid_end..len {
            out[i] = op.apply_scalar(s[i], o[i]);
        }

        Ok(())
    }

    /// Pairwise elementwise operation on `self` and `other`, returning a new `AlignedVec<T, Align>`.
    ///
    /// One allocation for the output buffer. Monomorphizes per `(T, Arch, Align, Op)` — the
    /// compiler generates the specialization most efficient for the target ISA and alignment.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if operand lengths do not match.
    pub fn zip_transform<ORef, Op>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        op: Op,
    ) -> Result<crate::vec::AlignedVec<T, Align>, SimdError>
    where
        ORef: 'a,
        Op: ElementOp<T>,
    {
        super::check_lengths_equal(self.len(), other.len())?;
        let len = self.len();
        let mut out = crate::vec::AlignedVec::with_capacity(len);
        // SAFETY: we write all `len` elements below via `zip_into`.
        unsafe {
            out.set_len(len);
        }
        self.zip_into(other, out.as_mut_slice(), op)?;
        Ok(out)
    }
}
