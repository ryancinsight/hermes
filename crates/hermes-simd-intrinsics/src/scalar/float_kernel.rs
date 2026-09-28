//! The scalar float kernel, stamped per element type.
//!
//! One `BackendKernel` implementation over `[T; LANES]` arrays, instantiated
//! by `f16`, `f32` and `f64` with their lane count and sign-bit layout.

macro_rules! impl_scalar_float_kernel {
    ($ty:ty, $lanes:expr, $sign_bit_shift:expr, $all_ones:expr) => {
        impl hermes_simd_core::kernel::BackendKernel<$ty> for crate::Scalar {
            type Vector = [$ty; $lanes];
            type Mask = [bool; $lanes];
            type IndexVector = [i32; $lanes];
            const LANE_COUNT: usize = $lanes;
            const UNROLL_FACTOR: usize = 4;

            #[inline(always)]
            unsafe fn load_aligned(ptr: *const $ty) -> Self::Vector {
                core::array::from_fn(|i| *ptr.add(i))
            }

            #[inline(always)]
            unsafe fn load_unaligned(ptr: *const $ty) -> Self::Vector {
                core::array::from_fn(|i| *ptr.add(i))
            }

            #[inline(always)]
            unsafe fn store_aligned(ptr: *mut $ty, val: Self::Vector) {
                core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, $lanes);
            }

            #[inline(always)]
            unsafe fn store_unaligned(ptr: *mut $ty, val: Self::Vector) {
                core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, $lanes);
            }

            #[inline(always)]
            unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i] + b[i])
            }

            #[inline(always)]
            unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i] * b[i])
            }

            #[inline(always)]
            unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i] - b[i])
            }

            #[inline(always)]
            unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i].mul_add(b[i], c[i]))
            }

            #[inline(always)]
            unsafe fn fmsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i].mul_add(b[i], -c[i]))
            }

            #[inline(always)]
            unsafe fn sum_reduce(v: Self::Vector) -> $ty {
                v.iter().copied().sum::<$ty>()
            }

            #[inline(always)]
            unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| a[i].sqrt())
            }

            #[inline(always)]
            unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
                core::array::from_fn(|i| (1.0 as $ty) / a[i].sqrt())
            }

            #[inline(always)]
            unsafe fn masked_load_unaligned(
                ptr: *const $ty,
                mask: Self::Mask,
                src: Self::Vector,
            ) -> Self::Vector {
                core::array::from_fn(|i| if mask[i] { *ptr.add(i) } else { src[i] })
            }

            #[inline(always)]
            unsafe fn masked_store_unaligned(ptr: *mut $ty, mask: Self::Mask, val: Self::Vector) {
                for i in 0..$lanes {
                    if mask[i] {
                        *ptr.add(i) = val[i];
                    }
                }
            }

            #[inline(always)]
            unsafe fn masked_add(
                a: Self::Vector,
                b: Self::Vector,
                mask: Self::Mask,
                src: Self::Vector,
            ) -> Self::Vector {
                core::array::from_fn(|i| if mask[i] { a[i] + b[i] } else { src[i] })
            }

            #[inline(always)]
            unsafe fn masked_mul(
                a: Self::Vector,
                b: Self::Vector,
                mask: Self::Mask,
                src: Self::Vector,
            ) -> Self::Vector {
                core::array::from_fn(|i| if mask[i] { a[i] * b[i] } else { src[i] })
            }

            #[inline(always)]
            unsafe fn masked_fmadd(
                a: Self::Vector,
                b: Self::Vector,
                c: Self::Vector,
                mask: Self::Mask,
            ) -> Self::Vector {
                core::array::from_fn(|i| {
                    if mask[i] {
                        a[i].mul_add(b[i], c[i])
                    } else {
                        c[i]
                    }
                })
            }

            #[inline(always)]
            unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> $ty {
                let mut s = 0.0 as $ty;
                for i in 0..$lanes {
                    if mask[i] {
                        s += v[i];
                    }
                }
                s
            }

            #[inline(always)]
            unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
                let mut out = [0.0 as $ty; $lanes];
                let mut k = 0usize;
                for i in 0..$lanes {
                    if mask[i] {
                        out[k] = src[i];
                        k += 1;
                    }
                }
                out
            }

            #[inline(always)]
            unsafe fn expand(
                src: Self::Vector,
                mask: Self::Mask,
                fill: Self::Vector,
            ) -> Self::Vector {
                let mut out = fill;
                let mut k = 0usize;
                for i in 0..$lanes {
                    if mask[i] {
                        out[i] = src[k];
                        k += 1;
                    }
                }
                out
            }

            #[inline(always)]
            unsafe fn gather(base: *const $ty, indices: Self::IndexVector) -> Self::Vector {
                core::array::from_fn(|i| *base.add(indices[i] as usize))
            }

            #[inline(always)]
            unsafe fn gather_masked(
                base: *const $ty,
                indices: Self::IndexVector,
                mask: Self::Mask,
                src: Self::Vector,
            ) -> Self::Vector {
                core::array::from_fn(|i| {
                    if mask[i] {
                        *base.add(indices[i] as usize)
                    } else {
                        src[i]
                    }
                })
            }

            #[inline(always)]
            unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
                debug_assert_eq!(bits.len(), $lanes);
                core::array::from_fn(|i| bits[i])
            }

            #[inline(always)]
            unsafe fn leading_k_mask(k: usize) -> Self::Mask {
                core::array::from_fn(|i| k > i)
            }

            #[inline(always)]
            unsafe fn zero() -> Self::Vector {
                [0.0 as $ty; $lanes]
            }

            #[inline(always)]
            unsafe fn splat(val: $ty) -> Self::Vector {
                [val; $lanes]
            }

            #[inline(always)]
            unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
                let mut m = 0u64;
                for i in 0..$lanes {
                    if mask[i] {
                        m |= 1u64 << i;
                    }
                }
                m
            }

            #[inline(always)]
            unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
                core::array::from_fn(|i| {
                    if mask[i] {
                        <$ty>::from_bits($all_ones)
                    } else {
                        0.0 as $ty
                    }
                })
            }

            #[inline(always)]
            unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
                // Read the sign bit to preserve all-ones compare-mask behavior.
                core::array::from_fn(|i| (v[i].to_bits() >> $sign_bit_shift) != 0)
            }
        }
    };
}
