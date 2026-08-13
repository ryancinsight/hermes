use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::ops::{ScanMode, ScanOp};
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};
use core::mem::MaybeUninit;

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Perform a prefix scan (inclusive or exclusive) of the view using the specified operation,
    /// writing results to `out`.
    ///
    /// # Errors
    /// Returns `SimdError::InsufficientOutputLength` if `out.len() < self.len()`.
    #[inline(always)]
    pub fn prefix_scan<Op, SMode>(
        &self,
        out: &mut [T],
        op: Op,
        mode: SMode,
    ) -> Result<(), SimdError>
    where
        Op: ScanOp<T>,
        SMode: ScanMode,
    {
        // SAFETY: an initialized `[T]` is a valid `[MaybeUninit<T>]` — the cast
        // only widens the permitted state — and `T: Scalar` is `Copy`, so the
        // `MaybeUninit::write`s in the delegate drop nothing.
        let out_uninit = unsafe {
            core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<MaybeUninit<T>>(), out.len())
        };
        self.prefix_scan_into_uninit(out_uninit, op, mode)?;
        Ok(())
    }

    /// Prefix scan into a possibly-uninitialized buffer, returning the initialized prefix.
    ///
    /// This is the single scan implementation; [`prefix_scan`](Self::prefix_scan)
    /// is the initialized-slice wrapper. On `Ok` exactly the first `self.len()`
    /// elements of `out` are initialized (and returned); on `Err` — only when
    /// `out` is too short, checked before any store — nothing is written.
    /// Filling an `AlignedVec`'s
    /// [`spare_capacity_mut`](crate::vec::AlignedVec::spare_capacity_mut) through
    /// this method avoids a zero-fill of the output.
    ///
    /// # Errors
    /// Returns `SimdError::InsufficientOutputLength` if `out.len() < self.len()`.
    #[inline]
    pub fn prefix_scan_into_uninit<'o, Op, SMode>(
        &self,
        out: &'o mut [MaybeUninit<T>],
        _op: Op,
        _mode: SMode,
    ) -> Result<&'o mut [T], SimdError>
    where
        Op: ScanOp<T>,
        SMode: ScanMode,
    {
        let len = self.len();
        if out.len() < len {
            return Err(SimdError::InsufficientOutputLength);
        }

        let src = self.as_slice();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr_in = src.as_ptr();
        // Derive the output pointer once and write exclusively through it: mixing
        // it with `out[i]` slice reborrows would invalidate its provenance under
        // Stacked Borrows.
        let ptr_out = out.as_mut_ptr().cast::<T>();

        let mut carry = Op::identity();

        // SAFETY: `out` holds at least `len` slots (checked above) and
        // `MaybeUninit<T>` shares `T`'s layout, so the vector and scalar stores
        // below fill `[0, len)` through `ptr_out` without reading any slot first;
        // `ptr_in` reads the view's own `len` initialized elements.
        unsafe {
            let load = |p: *const T| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::store_aligned(p, v)
                } else {
                    Arch::store_unaligned(p, v)
                }
            };

            for i in (0..simd_len).step_by(lane_count) {
                let v = load(ptr_in.add(i));
                let (r, next_carry) = Arch::scan_vector::<Op, SMode>(v, carry);
                store(ptr_out.add(i), r);
                carry = next_carry;
            }

            if SMode::IS_INCLUSIVE {
                for i in simd_len..len {
                    carry = Op::combine(carry, src[i]);
                    core::ptr::write(ptr_out.add(i), carry);
                }
            } else {
                for i in simd_len..len {
                    let temp = src[i];
                    core::ptr::write(ptr_out.add(i), carry);
                    carry = Op::combine(carry, temp);
                }
            }

            // Every element of `[0, len)` is now initialized.
            Ok(core::slice::from_raw_parts_mut(ptr_out, len))
        }
    }
}

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
where
    T: Scalar,
{
    /// Perform an in-place prefix scan (inclusive or exclusive) using the
    /// specified operation.
    ///
    /// Vectorized via `Arch::scan_vector` with a scalar carry across chunks;
    /// the scalar tail uses `Op::combine`. Loads and stores at the same offset
    /// are sequential, so no intra-chunk aliasing hazard exists.
    #[inline(always)]
    pub fn prefix_scan_in_place<Op, SMode>(&mut self, _op: Op, _mode: SMode)
    where
        Op: ScanOp<T>,
        SMode: ScanMode,
    {
        let data = self.as_slice_mut();
        let len = data.len();
        if len == 0 {
            return;
        }

        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr = data.as_mut_ptr();

        let mut carry = Op::identity();

        unsafe {
            let load = |p: *const T| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::store_aligned(p, v)
                } else {
                    Arch::store_unaligned(p, v)
                }
            };

            for i in (0..simd_len).step_by(lane_count) {
                let v = load(ptr.add(i));
                let (r, next_carry) = Arch::scan_vector::<Op, SMode>(v, carry);
                store(ptr.add(i), r);
                carry = next_carry;
            }
        }

        if SMode::IS_INCLUSIVE {
            for x in &mut data[simd_len..] {
                carry = Op::combine(carry, *x);
                *x = carry;
            }
        } else {
            for x in &mut data[simd_len..] {
                let temp = *x;
                *x = carry;
                carry = Op::combine(carry, temp);
            }
        }
    }
}
