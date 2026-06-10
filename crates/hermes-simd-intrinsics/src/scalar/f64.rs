//! Fallback scalar f64 kernel.
//!
//! All SIMD operations degenerate to element-wise scalar loops. This is the
//! universal fallback when no hardware SIMD feature is detected.

use crate::Scalar;
use hermes_simd_core::kernel::SimdKernel;

impl SimdKernel<f64> for Scalar {
    type Vector = [f64; 2];
    type Mask = [bool; 2];
    type IndexVector = [i32; 2];
    const LANE_COUNT: usize = 2;
    const UNROLL_FACTOR: usize = 4;

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Vector {
        [*ptr, *ptr.add(1)]
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Vector {
        [*ptr, *ptr.add(1)]
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut f64, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 2);
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut f64, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 2);
    }

    #[inline(always)]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        [a[0] + b[0], a[1] + b[1]]
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        [a[0] * b[0], a[1] * b[1]]
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        [a[0] - b[0], a[1] - b[1]]
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        // f64::mul_add for IEEE 754 FMA consistency.
        [a[0].mul_add(b[0], c[0]), a[1].mul_add(b[1], c[1])]
    }

    #[inline(always)]
    unsafe fn sum_reduce(v: Self::Vector) -> f64 {
        v[0] + v[1]
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn masked_load_unaligned(
        ptr: *const f64,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        [
            if mask[0] { *ptr } else { src[0] },
            if mask[1] { *ptr.add(1) } else { src[1] },
        ]
    }

    #[inline(always)]
    unsafe fn masked_store_unaligned(ptr: *mut f64, mask: Self::Mask, val: Self::Vector) {
        if mask[0] {
            *ptr = val[0];
        }
        if mask[1] {
            *ptr.add(1) = val[1];
        }
    }

    // -----------------------------------------------------------------------
    // Masked arithmetic
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        [
            if mask[0] { a[0] + b[0] } else { src[0] },
            if mask[1] { a[1] + b[1] } else { src[1] },
        ]
    }

    #[inline(always)]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        [
            if mask[0] { a[0] * b[0] } else { src[0] },
            if mask[1] { a[1] * b[1] } else { src[1] },
        ]
    }

    #[inline(always)]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        [
            if mask[0] {
                a[0].mul_add(b[0], c[0])
            } else {
                c[0]
            },
            if mask[1] {
                a[1].mul_add(b[1], c[1])
            } else {
                c[1]
            },
        ]
    }

    #[inline(always)]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f64 {
        let mut s = 0.0f64;
        if mask[0] {
            s += v[0];
        }
        if mask[1] {
            s += v[1];
        }
        s
    }

    // -----------------------------------------------------------------------
    // Compress / Expand
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut out = [0.0f64; 2];
        let mut k = 0usize;
        for i in 0..2 {
            if mask[i] {
                out[k] = src[i];
                k += 1;
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mut out = fill;
        let mut k = 0usize;
        for i in 0..2 {
            if mask[i] {
                out[i] = src[k];
                k += 1;
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn gather(base: *const f64, indices: Self::IndexVector) -> Self::Vector {
        [
            *base.add(indices[0] as usize),
            *base.add(indices[1] as usize),
        ]
    }

    #[inline(always)]
    unsafe fn gather_masked(
        base: *const f64,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        [
            if mask[0] {
                *base.add(indices[0] as usize)
            } else {
                src[0]
            },
            if mask[1] {
                *base.add(indices[1] as usize)
            } else {
                src[1]
            },
        ]
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 2);
        [bits[0], bits[1]]
    }

    #[inline(always)]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        [k > 0, k > 1]
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn zero() -> Self::Vector {
        [0.0f64; 2]
    }

    #[inline(always)]
    unsafe fn splat(val: f64) -> Self::Vector {
        [val; 2]
    }

    #[inline(always)]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        let mut m = 0u64;
        for i in 0..2 {
            if mask[i] {
                m |= 1u64 << i;
            }
        }
        m
    }

    #[inline(always)]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        [
            if mask[0] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0f64
            },
            if mask[1] {
                f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)
            } else {
                0.0f64
            },
        ]
    }
}
