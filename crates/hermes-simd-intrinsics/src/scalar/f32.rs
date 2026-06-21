//! Fallback scalar f32 kernel.
//!
//! All SIMD operations degenerate to element-wise scalar loops. This is the
//! universal fallback when no hardware SIMD feature is detected.

use crate::Scalar;
use hermes_simd_core::kernel::SimdKernel;

impl SimdKernel<f32> for Scalar {
    type Vector = [f32; 4];
    type Mask = [bool; 4];
    type IndexVector = [i32; 4];
    const LANE_COUNT: usize = 4;
    const UNROLL_FACTOR: usize = 4;

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vector {
        [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)]
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vector {
        [*ptr, *ptr.add(1), *ptr.add(2), *ptr.add(3)]
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut f32, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 4);
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut f32, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 4);
    }

    #[inline(always)]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        // f32::mul_add is a fused multiply-add (no intermediate rounding),
        // consistent with Scalar::scalar_fmadd and hardware FMA paths.
        [
            a[0].mul_add(b[0], c[0]),
            a[1].mul_add(b[1], c[1]),
            a[2].mul_add(b[2], c[2]),
            a[3].mul_add(b[3], c[3]),
        ]
    }

    #[inline(always)]
    unsafe fn sum_reduce(v: Self::Vector) -> f32 {
        v[0] + v[1] + v[2] + v[3]
    }

    #[inline(always)]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        [a[0].sqrt(), a[1].sqrt(), a[2].sqrt(), a[3].sqrt()]
    }

    #[inline(always)]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        [
            1.0 / a[0].sqrt(),
            1.0 / a[1].sqrt(),
            1.0 / a[2].sqrt(),
            1.0 / a[3].sqrt(),
        ]
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn masked_load_unaligned(
        ptr: *const f32,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        [
            if mask[0] { *ptr } else { src[0] },
            if mask[1] { *ptr.add(1) } else { src[1] },
            if mask[2] { *ptr.add(2) } else { src[2] },
            if mask[3] { *ptr.add(3) } else { src[3] },
        ]
    }

    #[inline(always)]
    unsafe fn masked_store_unaligned(ptr: *mut f32, mask: Self::Mask, val: Self::Vector) {
        if mask[0] {
            *ptr = val[0];
        }
        if mask[1] {
            *ptr.add(1) = val[1];
        }
        if mask[2] {
            *ptr.add(2) = val[2];
        }
        if mask[3] {
            *ptr.add(3) = val[3];
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
            if mask[2] { a[2] + b[2] } else { src[2] },
            if mask[3] { a[3] + b[3] } else { src[3] },
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
            if mask[2] { a[2] * b[2] } else { src[2] },
            if mask[3] { a[3] * b[3] } else { src[3] },
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
            if mask[2] {
                a[2].mul_add(b[2], c[2])
            } else {
                c[2]
            },
            if mask[3] {
                a[3].mul_add(b[3], c[3])
            } else {
                c[3]
            },
        ]
    }

    #[inline(always)]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f32 {
        let mut s = 0.0f32;
        if mask[0] {
            s += v[0];
        }
        if mask[1] {
            s += v[1];
        }
        if mask[2] {
            s += v[2];
        }
        if mask[3] {
            s += v[3];
        }
        s
    }

    // -----------------------------------------------------------------------
    // Compress / Expand
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut out = [0.0f32; 4];
        let mut k = 0usize;
        for i in 0..4 {
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
        for i in 0..4 {
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
    unsafe fn gather(base: *const f32, indices: Self::IndexVector) -> Self::Vector {
        [
            *base.add(indices[0] as usize),
            *base.add(indices[1] as usize),
            *base.add(indices[2] as usize),
            *base.add(indices[3] as usize),
        ]
    }

    #[inline(always)]
    unsafe fn gather_masked(
        base: *const f32,
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
            if mask[2] {
                *base.add(indices[2] as usize)
            } else {
                src[2]
            },
            if mask[3] {
                *base.add(indices[3] as usize)
            } else {
                src[3]
            },
        ]
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 4);
        [bits[0], bits[1], bits[2], bits[3]]
    }

    #[inline(always)]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        [k > 0, k > 1, k > 2, k > 3]
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[inline(always)]
    unsafe fn zero() -> Self::Vector {
        [0.0f32; 4]
    }

    #[inline(always)]
    unsafe fn splat(val: f32) -> Self::Vector {
        [val; 4]
    }

    #[inline(always)]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        let mut m = 0u64;
        for i in 0..4 {
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
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0f32
            },
            if mask[1] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0f32
            },
            if mask[2] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0f32
            },
            if mask[3] {
                f32::from_bits(0xFFFF_FFFF)
            } else {
                0.0f32
            },
        ]
    }
}
