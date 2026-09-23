//! Index-returning reductions (`argmin` / `argmax`).

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::{SimdArith, SimdCompare, SimdLoadStore, SimdMask, SimdReduce};
use crate::scalar::{NumericElement, Scalar};
use crate::view::SimdView;

impl<
        'a,
        T: 'a,
        Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdCompare<T> + SimdMask<T> + SimdReduce<T>,
        Align: Alignment,
        Mode: ExecutionMode,
        Ref: 'a,
    > SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar + NumericElement,
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
