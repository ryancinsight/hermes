//! Low-level SIMD operations trait implemented per architecture and primitive type.
//!
//! # Extension Surface (v2+)
//!
//! New in this iteration:
//! - `sub(a, b)` — elementwise subtraction, required for `Sub` ElementOp strategy.
//! - `mask_from_bitmask(bm)` — convert `BitMask<LANE_COUNT>` to native mask; default
//!   calls `mask_from_bools` via `BitMask::to_bools()`. AVX-512 impls override with direct cast.
//! - `zero()` — returns a vector of zeros; default implementation uses `splat(T::ZERO)`.
//!   Backends may override with an architecture-specific XOR-zero idiom if profiling shows benefit.
//!
//! # Extension Surface (v2)
//!
//! Beyond the base load/store/arithmetic/reduce methods, `SimdKernel` now exposes:
//!
//! - **Masked operations** (`masked_load_unaligned`, `masked_store_unaligned`,
//!   `masked_add`, `masked_mul`, `masked_fmadd`, `masked_sum_reduce`) — predicated
//!   arithmetic using hardware mask registers. The `src` parameter follows AVX-512
//!   merge-masking semantics: lanes where `mask[i] = 0` are taken from `src`.
//!
//! - **Compress / expand** — scatter/gather from/to contiguous storage:
//!   - `compress`: packs selected lanes (`mask[i]=1`) to low lanes of result.
//!   - `expand`: scatters low lanes of `src` to positions where `mask[i]=1`.
//!
//! - **Gather** (`gather`, `gather_masked`) — indirect indexed load from a base pointer.
//!
//! - **Mask construction** (`mask_from_bools`, `leading_k_mask`) — build masks from
//!   boolean arrays or lane counts for tail handling.
//!
//! # Architecture Mapping
//!
//! | Method | AVX-512 | AVX2 | NEON | Scalar |
//! |--------|---------|------|------|--------|
//! | `masked_add` | `_mm512_mask_add_ps` | `_mm256_blendv_ps(src,add,mask)` | `vbslq_f32` | loop+if |
//! | `compress` | `_mm512_mask_compressstoreu_ps` | emulated | emulated | loop+if |
//! | `gather` | `_mm512_i32gather_ps` | `_mm256_i32gather_ps` | emulated | loop |

/// Lane capacity of the fixed scalar-fallback stack buffers used by the default
/// `SimdKernel` methods (`scan_vector`, `swap_adjacent`, `dup_even`/`dup_odd`,
/// and the `kernel_helpers` scalar emulations). A backend's [`SimdKernel::LANE_COUNT`]
/// must not exceed this, or `store_unaligned` into those buffers would overflow
/// the stack. The current maximum is 64 (AVX-512 `i8`); the bound is checked at
/// compile time by [`SimdKernel::LANE_BOUND_CHECK`].
pub const MAX_SIMD_LANES: usize = 128;

/// Abstract trait defining low-level vector operations.
///
/// Implemented by ZST architecture markers. All methods are `unsafe` — the caller is
/// responsible for ensuring target-feature prerequisites are satisfied. The `#[target_feature]`
/// attribute on each `impl` block ensures the compiler emits the correct machine instruction;
/// calling from a non-gated context requires wrapping in an `unsafe { ... }` block inside
/// a function that is itself gated by `#[target_feature(enable = "...")]`.
///
/// # Examples
///
/// Use the always-available `Scalar` backend for cross-platform code paths:
///
/// ```rust
/// use hermes_simd_intrinsics::Scalar;
/// use hermes_simd_core::kernel::SimdKernel;
///
/// // SAFETY: `Scalar` requires no special ISA features.
/// let splat4: <Scalar as SimdKernel<f32>>::Vector =
///     unsafe { <Scalar as SimdKernel<f32>>::splat(1.0_f32) };
/// let sum: f32 = unsafe { <Scalar as SimdKernel<f32>>::sum_reduce(splat4) };
/// assert_eq!(sum, <Scalar as SimdKernel<f32>>::LANE_COUNT as f32);
/// ```
pub trait SimdKernel<T: crate::scalar::Scalar>:
    crate::private::Sealed + Send + Sync + Sized + 'static
{
    /// The underlying raw register/vector type for this architecture and element type.
    type Vector: Copy + Send + Sync + 'static;

    /// Hardware-native mask type.
    ///
    /// - AVX-512 f32: `__mmask16`
    /// - AVX-512 f64: `__mmask8`
    /// - AVX2 f32: `__m256` (float blend mask)
    /// - AVX2 f64: `__m256d`
    /// - NEON f32: `uint32x4_t`
    /// - NEON f64: `uint64x2_t`
    /// - Scalar f32: `[bool; 4]`
    /// - Scalar f64: `[bool; 2]`
    type Mask: Copy + Send + Sync + 'static;

    /// Integer index vector for gather operations.
    ///
    /// - AVX-512 f32 (16-lane): `__m512i` (16xi32)
    /// - AVX-512 f64 (8-lane): `__m256i` (8xi32)
    /// - AVX2 f32 (8-lane): `__m256i` (8xi32)
    /// - AVX2 f64 (4-lane): `__m128i` (4xi32)
    /// - NEON / Scalar: `[i32; LANE_COUNT]`
    type IndexVector: Copy + Send + Sync + 'static;

    /// Number of primitive elements of type `T` in one `Vector`.
    const LANE_COUNT: usize;

    /// Compile-time guard that [`LANE_COUNT`](Self::LANE_COUNT) fits the fixed
    /// `MAX_SIMD_LANES` scalar-fallback stack buffers. Referencing this const in
    /// the buffer-using default methods forces the assertion to be evaluated for
    /// each concrete backend at monomorphization, turning a would-be silent
    /// stack-buffer overflow into a compile error.
    const LANE_BOUND_CHECK: () = assert!(
        Self::LANE_COUNT <= MAX_SIMD_LANES,
        "SimdKernel::LANE_COUNT exceeds MAX_SIMD_LANES; widen the scalar-fallback stack buffers"
    );

    /// Loop unrolling register accumulation factor to break loop-carried dependency chains.
    const UNROLL_FACTOR: usize = 4;

    // -------------------------------------------------------------------------
    // Load / Store
    // -------------------------------------------------------------------------

    /// Load a vector from an aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for reads and aligned to `LANE_COUNT * size_of::<T>()` bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hermes_simd_intrinsics::Scalar;
    /// use hermes_simd_core::kernel::SimdKernel;
    ///
    /// #[repr(align(64))]
    /// struct AlignedBuf([f32; 4]);
    ///
    /// let buf = AlignedBuf([1.0, 2.0, 3.0, 4.0]);
    /// // SAFETY: buf is 64-byte aligned and valid for LANE_COUNT reads.
    /// let v = unsafe { <Scalar as SimdKernel<f32>>::load_aligned(buf.0.as_ptr()) };
    /// let sum: f32 = unsafe { <Scalar as SimdKernel<f32>>::sum_reduce(v) };
    /// assert_eq!(sum, 10.0_f32);
    /// ```
    unsafe fn load_aligned(ptr: *const T) -> Self::Vector;

    /// Load a vector from an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for reads.
    unsafe fn load_unaligned(ptr: *const T) -> Self::Vector;

    /// Store a vector to an aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for writes and aligned to `LANE_COUNT * size_of::<T>()` bytes.
    unsafe fn store_aligned(ptr: *mut T, val: Self::Vector);

    /// Store a vector to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for writes.
    unsafe fn store_unaligned(ptr: *mut T, val: Self::Vector);

    // -------------------------------------------------------------------------
    // Dense Arithmetic
    // -------------------------------------------------------------------------

    /// Elementwise addition: `a + b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Elementwise multiplication: `a * b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Elementwise subtraction: `a - b`.
    ///
    /// Default: scalar fallback via `crate::kernel_helpers::generic_binary_op`.
    /// Float and SIMD backends override this with the appropriate vectorized instruction
    /// (e.g., `_mm256_sub_ps` for AVX2 f32, `vsubq_f32` for NEON).
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| x - y)
    }

    /// Fused multiply-add: `(a * b) + c`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;

    /// Horizontal sum of all lanes.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hermes_simd_intrinsics::Scalar;
    /// use hermes_simd_core::kernel::SimdKernel;
    ///
    /// let data = [1.0_f32, 2.0, 3.0, 4.0];
    /// // SAFETY: Scalar requires no ISA feature; pointer is valid for LANE_COUNT reads.
    /// let v = unsafe { <Scalar as SimdKernel<f32>>::load_unaligned(data.as_ptr()) };
    /// let total: f32 = unsafe { <Scalar as SimdKernel<f32>>::sum_reduce(v) };
    /// assert!((total - 10.0_f32).abs() < 1e-6);
    /// ```
    unsafe fn sum_reduce(v: Self::Vector) -> T;

    // -------------------------------------------------------------------------
    // Masked Load / Store (merge masking: inactive lanes come from `src`)
    // -------------------------------------------------------------------------

    /// Masked load: active lanes loaded from `ptr`, inactive lanes taken from `src`.
    ///
    /// # Safety
    /// `ptr` must be valid for reading `LANE_COUNT` elements. Active lanes determined by `mask`.
    unsafe fn masked_load_unaligned(
        ptr: *const T,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Masked store: active lanes written to `ptr`, inactive lanes left unchanged.
    ///
    /// # Safety
    /// `ptr` must be valid for writing `LANE_COUNT` elements.
    unsafe fn masked_store_unaligned(ptr: *mut T, mask: Self::Mask, val: Self::Vector);

    // -------------------------------------------------------------------------
    // Masked Arithmetic (merge masking)
    // -------------------------------------------------------------------------

    /// Masked elementwise add: active lanes compute `a + b`, inactive lanes yield `src`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Masked elementwise multiply: active lanes compute `a * b`, inactive lanes yield `src`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Masked fused multiply-add: active lanes compute `(a * b) + c`, inactive lanes retain `c`.
    ///
    /// The merge source for inactive lanes is the addend `c`, matching AVX-512 semantics
    /// for `_mm512_mask_fmadd_ps(a, mask, b, c)`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector;

    /// Masked horizontal sum: only lanes where `mask[i]=1` contribute.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> T;

    // -------------------------------------------------------------------------
    // Compress / Expand
    // -------------------------------------------------------------------------

    /// Compress: pack selected lanes (where `mask[i]=1`) into the low lanes of the result.
    ///
    /// Unselected high lanes of the result are unspecified.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector;

    /// Expand: scatter the low lanes of `src` into result positions where `mask[i]=1`.
    ///
    /// Result positions where `mask[i]=0` are filled with `fill`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector;

    // -------------------------------------------------------------------------
    // Gather (indirect indexed load)
    // -------------------------------------------------------------------------

    /// Gather: load `LANE_COUNT` elements at `base + indices[i]` for each lane `i`.
    ///
    /// # Safety
    /// All `base + indices[i]` must be valid for reads.
    unsafe fn gather(base: *const T, indices: Self::IndexVector) -> Self::Vector;

    /// Masked gather: gather active lanes; inactive lanes take value from `src`.
    ///
    /// # Safety
    /// Active `base + indices[i]` must be valid for reads.
    unsafe fn gather_masked(
        base: *const T,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    // -------------------------------------------------------------------------
    // Mask Construction Helpers
    // -------------------------------------------------------------------------

    /// Construct a mask from a slice of booleans (length must equal `LANE_COUNT`).
    ///
    /// # Panics
    /// Panics in debug builds if `bits.len() != LANE_COUNT`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask;

    /// Construct a mask with the first `k` lanes active and the rest inactive.
    ///
    /// If `k >= LANE_COUNT`, all lanes are active. Used for tail handling.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn leading_k_mask(k: usize) -> Self::Mask;

    /// Convert a raw `u64` bitmask to the architecture-native mask type.
    ///
    /// Default: expands to a boolean array then calls `mask_from_bools`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask {
        crate::kernel_helpers::generic_mask_from_bitmask::<T, Self>(bm)
    }

    /// Convert the native mask back to a vector register where active lanes
    /// are set to `T::ALL_ONES` and inactive lanes to `T::ZERO`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector;

    /// Perform an intra-vector prefix scan (inclusive or exclusive) of the vector,
    /// using the specified `ScanOp` strategy and starting carry value.
    /// Returns the scanned vector and the final carry value.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn scan_vector<Op: crate::ops::ScanOp<T>, SMode: crate::ops::ScanMode>(
        v: Self::Vector,
        mut carry: T,
    ) -> (Self::Vector, T) {
        const { Self::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let lanes = Self::LANE_COUNT;
        Self::store_unaligned(buf.as_mut_ptr() as *mut T, v);

        let mut out_buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        if SMode::IS_INCLUSIVE {
            for j in 0..lanes {
                let temp = buf[j].assume_init();
                carry = Op::combine(carry, temp);
                out_buf[j].write(carry);
            }
        } else {
            for j in 0..lanes {
                let temp = buf[j].assume_init();
                out_buf[j].write(carry);
                carry = Op::combine(carry, temp);
            }
        }

        (Self::load_unaligned(out_buf.as_ptr() as *const T), carry)
    }

    /// Set all lanes to zero.
    ///
    /// Default: delegates to `splat(T::ZERO)`. Backends may override with an
    /// architecture-specific XOR-zero idiom (e.g., `_mm256_xor_ps`) if profiling
    /// shows a register-pressure benefit.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn zero() -> Self::Vector {
        Self::splat(T::ZERO)
    }

    /// Broadcast a scalar value to all lanes.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hermes_simd_intrinsics::Scalar;
    /// use hermes_simd_core::kernel::SimdKernel;
    ///
    /// // SAFETY: Scalar backend requires no ISA feature.
    /// let v = unsafe { <Scalar as SimdKernel<f32>>::splat(42.0_f32) };
    /// let sum: f32 = unsafe { <Scalar as SimdKernel<f32>>::sum_reduce(v) };
    /// assert_eq!(sum, 42.0_f32 * <Scalar as SimdKernel<f32>>::LANE_COUNT as f32);
    /// ```
    unsafe fn splat(val: T) -> Self::Vector;

    /// Elementwise division: `a / b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| x / y)
    }

    /// Elementwise bitwise AND: `a & b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| x.bitand(y))
    }

    /// Elementwise bitwise OR: `a | b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| x.bitor(y))
    }

    /// Elementwise bitwise XOR: `a ^ b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| x.bitxor(y))
    }

    /// Elementwise absolute value.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(a, |x| x.abs())
    }

    /// Elementwise minimum of `a` and `b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(
            a,
            b,
            |x, y| if x < y { x } else { y },
        )
    }

    /// Elementwise maximum of `a` and `b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(
            a,
            b,
            |x, y| if x > y { x } else { y },
        )
    }

    /// Elementwise square root.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(a, |x| x.sqrt())
    }

    /// Elementwise reciprocal square root.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(a, |x| T::ONE / x.sqrt())
    }

    /// Elementwise equal: `a == b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| {
            if x == y {
                T::ALL_ONES
            } else {
                T::ZERO
            }
        })
    }

    /// Elementwise not equal: `a != b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| {
            if x != y {
                T::ALL_ONES
            } else {
                T::ZERO
            }
        })
    }

    /// Elementwise less than: `a < b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| {
            if x < y {
                T::ALL_ONES
            } else {
                T::ZERO
            }
        })
    }

    /// Elementwise less than or equal: `a <= b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| {
            if x <= y {
                T::ALL_ONES
            } else {
                T::ZERO
            }
        })
    }

    /// Elementwise greater than: `a > b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| {
            if x > y {
                T::ALL_ONES
            } else {
                T::ZERO
            }
        })
    }

    /// Elementwise greater than or equal: `a >= b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, |x, y| {
            if x >= y {
                T::ALL_ONES
            } else {
                T::ZERO
            }
        })
    }

    /// Elementwise blend: select lanes from `true_val` where the sign bit of `mask` is set,
    /// and from `false_val` otherwise.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        crate::kernel_helpers::generic_blend::<T, Self>(mask, true_val, false_val)
    }

    /// Elementwise negate: `-a`.
    ///
    /// Default implementation: XOR each lane with `T::SIGN_MASK` (the IEEE 754 sign bit).
    /// This avoids the `sub(zero, a)` path, which panics on backends that do not implement
    /// subtraction (e.g. `bf16` on AVX2). Every backend implements `bitxor` and `splat`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn neg(a: Self::Vector) -> Self::Vector {
        Self::bitxor(a, Self::splat(T::SIGN_MASK))
    }

    /// Elementwise bitwise NOT: `!a`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn bitnot(a: Self::Vector) -> Self::Vector {
        Self::bitxor(a, Self::splat(T::ALL_ONES))
    }

    /// Convert the native mask back to a raw `u64` bitmask.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64;

    /// Horizontal minimum across all lanes.
    ///
    /// Default: scalar lane-by-lane scan using [`crate::scalar::NumericElement::min_scalar`].
    /// AVX-512 impls override with `_mm512_reduce_min_ps` / `_mm256_reduce_min_ps` or equivalent.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn min_reduce(v: Self::Vector) -> T {
        crate::kernel_helpers::generic_horizontal_reduce::<T, Self>(v, T::MAX_VALUE, |a, b| {
            a.min_scalar(b)
        })
    }

    /// Horizontal maximum across all lanes.
    ///
    /// Default: scalar lane-by-lane scan using [`crate::scalar::NumericElement::max_scalar`].
    /// AVX-512 impls override with `_mm512_reduce_max_ps` / `_mm256_reduce_max_ps` or equivalent.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn max_reduce(v: Self::Vector) -> T {
        crate::kernel_helpers::generic_horizontal_reduce::<T, Self>(v, T::MIN_VALUE, |a, b| {
            a.max_scalar(b)
        })
    }

    /// Elementwise population count (number of set bits).
    ///
    /// Default: scalar lane-by-lane scan using [`crate::scalar::NumericElement::count_ones`].
    /// Target-specific intrinsics override this.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(a, |x| {
            T::cast_from(x.count_ones() as i32)
        })
    }

    /// Horizontal bitwise AND across all lanes.
    ///
    /// Default: scalar lane-by-lane scan using [`crate::scalar::NumericElement::bitand`].
    /// Target-specific intrinsics override this.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn horizontal_bitwise_and(v: Self::Vector) -> T {
        crate::kernel_helpers::generic_horizontal_reduce::<T, Self>(v, T::ALL_ONES, |a, b| {
            a.bitand(b)
        })
    }

    /// Horizontal bitwise OR across all lanes.
    ///
    /// Default: scalar lane-by-lane scan using [`crate::scalar::NumericElement::bitor`].
    /// Target-specific intrinsics override this.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn horizontal_bitwise_or(v: Self::Vector) -> T {
        crate::kernel_helpers::generic_horizontal_reduce::<T, Self>(v, T::ZERO, |a, b| a.bitor(b))
    }

    /// Horizontal bitwise XOR across all lanes.
    ///
    /// Default: scalar lane-by-lane scan using [`crate::scalar::NumericElement::bitxor`].
    /// Target-specific intrinsics override this.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn horizontal_bitwise_xor(v: Self::Vector) -> T {
        crate::kernel_helpers::generic_horizontal_reduce::<T, Self>(v, T::ZERO, |a, b| a.bitxor(b))
    }

    // -------------------------------------------------------------------------
    // Adjacent-Pair Shuffles & Alternating FMA (interleaved complex support)
    // -------------------------------------------------------------------------
    //
    // These five methods are the minimal primitive set required to express
    // interleaved complex arithmetic (`[re, im, re, im, ...]` lane order)
    // entirely in vector registers:
    //
    //   a * b       = fmaddsub(dup_even(a), b, mul(dup_odd(a), swap_adjacent(b)))
    //   a * conj(b) = fmsubadd(dup_odd(a), swap_adjacent(b), mul(dup_even(a), b))
    //
    // Pair semantics assume an even `LANE_COUNT`; on a backend with an odd
    // lane count the last (unpaired) lane passes through unchanged.

    /// Swap each adjacent lane pair: `[a0, a1, a2, a3, ...] -> [a1, a0, a3, a2, ...]`.
    ///
    /// Default: scalar emulation via store/swap/load. x86 backends override with
    /// `_mm256_permute_ps(v, 0b1011_0001)` / `_mm256_permute_pd(v, 0b0101)` and
    /// the AVX-512 equivalents.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        const { Self::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let lanes = Self::LANE_COUNT;
        Self::store_unaligned(buf.as_mut_ptr() as *mut T, v);
        let mut i = 0usize;
        while i + 1 < lanes {
            buf.swap(i, i + 1);
            i += 2;
        }
        Self::load_unaligned(buf.as_ptr() as *const T)
    }

    /// Duplicate even lanes into odd lanes: `[a0, a1, a2, a3, ...] -> [a0, a0, a2, a2, ...]`.
    ///
    /// Default: scalar emulation. x86 backends override with `moveldup_ps` /
    /// `movedup_pd`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        const { Self::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let lanes = Self::LANE_COUNT;
        Self::store_unaligned(buf.as_mut_ptr() as *mut T, v);
        let mut out = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        for i in 0..lanes {
            let src_val = buf[i & !1].assume_init();
            out[i].write(src_val);
        }
        Self::load_unaligned(out.as_ptr() as *const T)
    }

    /// Duplicate odd lanes into even lanes: `[a0, a1, a2, a3, ...] -> [a1, a1, a3, a3, ...]`.
    ///
    /// Default: scalar emulation. x86 backends override with `movehdup_ps` /
    /// an odd-lane `permute_pd`. An unpaired trailing lane (odd `LANE_COUNT`)
    /// passes through unchanged.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        const { Self::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let lanes = Self::LANE_COUNT;
        Self::store_unaligned(buf.as_mut_ptr() as *mut T, v);
        let mut out = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        for i in 0..lanes {
            let src_val = buf[(i | 1).min(lanes - 1)].assume_init();
            out[i].write(src_val);
        }
        Self::load_unaligned(out.as_ptr() as *const T)
    }

    /// Alternating fused multiply: even lanes `a*b - c`, odd lanes `a*b + c`.
    ///
    /// Default: scalar emulation. x86 backends override with
    /// `_mm256_fmaddsub_ps/pd` / `_mm512_fmaddsub_ps/pd`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_alternating_fma::<T, Self, false>(a, b, c)
    }

    /// Alternating fused multiply: even lanes `a*b + c`, odd lanes `a*b - c`.
    ///
    /// Default: scalar emulation. x86 backends override with
    /// `_mm256_fmsubadd_ps/pd` / `_mm512_fmsubadd_ps/pd`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_alternating_fma::<T, Self, true>(a, b, c)
    }
}
