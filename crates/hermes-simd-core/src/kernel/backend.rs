//! Low-level SIMD operations trait implemented per architecture and primitive type.
//!
//! # Operation Families
//!
//! [`BackendKernel`] is the single sealed seam every backend implements. Its methods group
//! into the families below; a backend must supply the required ones and may
//! override any default where the ISA has a native instruction.
//!
//! - **Load / store** — aligned, unaligned, and non-temporal (`store_streaming`,
//!   gated on `SUPPORTS_NT_STORE` with `stream_write_barrier` for ordering).
//! - **Dense arithmetic** — `add`, `sub`, `mul`, `div`, `fmadd`, `fmsub`, `neg`, `abs`,
//!   `min`, `max`, `sqrt`, `recip_sqrt`, and the `floor`/`ceil`/`round`/`trunc`
//!   rounding set.
//! - **Bitwise** — `bitand`, `bitor`, `bitxor`, `bitnot`, `popcount`.
//! - **Comparison** — `cmp_eq`/`ne`/`lt`/`le`/`gt`/`ge`, returning lane masks as
//!   vectors, plus `blend`.
//! - **Reduction** — `sum_reduce`, `min_reduce`, `max_reduce`, and the
//!   `horizontal_bitwise_*` family.
//! - **Masked operations** — `masked_load_unaligned`, `masked_store_unaligned`,
//!   `masked_add`, `masked_mul`, `masked_fmadd`, `masked_sum_reduce`. Predication
//!   follows AVX-512 merge-masking semantics: lanes where `mask[i] = 0` are taken
//!   from `src`.
//! - **Compress / expand** — `compress` packs selected lanes (`mask[i] = 1`) into
//!   the low lanes of the result; `expand` scatters the low lanes of `src` back to
//!   the positions where `mask[i] = 1`.
//! - **Gather / scatter** — `gather`, `gather_masked`, `scatter`, `scatter_masked`:
//!   indirect indexed load and store through [`BackendKernel::IndexVector`].
//! - **Mask construction** — `mask_from_bools`, `mask_from_bitmask`,
//!   `leading_k_mask` (tail handling), and the `mask_to_vector` /
//!   `vector_to_mask` / `mask_to_bitmask` conversions.
//! - **Scan** — `scan_vector`, parameterized by a [`crate::ops::ScanOp`] and an
//!   inclusive/exclusive [`crate::ops::ScanMode`].
//! - **Cross-lane permutes** — `reverse`, `interleave`, `deinterleave`, all
//!   specified on the flat lane sequence rather than per 128-bit sub-lane.
//! - **Adjacent-pair shuffles** — `swap_adjacent`, `dup_even`, `dup_odd`,
//!   `fmaddsub`, `fmsubadd`: the minimal set for register-resident interleaved
//!   complex arithmetic.
//!
//! # Architecture Mapping
//!
//! | Method | AVX-512 | AVX2 | NEON | Scalar |
//! |--------|---------|------|------|--------|
//! | `masked_add` | `_mm512_mask_add_ps` | `_mm256_blendv_ps(src,add,mask)` | `vbslq_f32` | loop+if |
//! | `compress` | `_mm512_mask_compressstoreu_ps` | emulated | emulated | loop+if |
//! | `gather` | `_mm512_i32gather_ps` | `_mm256_i32gather_ps` | emulated | loop |

/// Lane capacity of the fixed scalar-fallback stack buffers used by the default
/// `BackendKernel` methods (`scan_vector`, `swap_adjacent`, `dup_even`/`dup_odd`,
/// and the `kernel_helpers` scalar emulations). A backend's [`BackendKernel::LANE_COUNT`]
/// must not exceed this, or `store_unaligned` into those buffers would overflow
/// the stack. The current workspace maximum is 64 (AVX-512 `i8`, 64×`i8`); the
/// bound is checked at compile time by [`BackendKernel::LANE_BOUND_CHECK`], so a
/// future wider backend fails to build rather than silently overflowing the stack.
pub const MAX_SIMD_LANES: usize = 64;

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
/// use hermes_simd_core::kernel::BackendKernel;
///
/// // SAFETY: `Scalar` requires no special ISA features.
/// let splat4: <Scalar as BackendKernel<f32>>::Vector =
///     unsafe { <Scalar as BackendKernel<f32>>::splat(1.0_f32) };
/// let sum: f32 = unsafe { <Scalar as BackendKernel<f32>>::sum_reduce(splat4) };
/// assert_eq!(sum, <Scalar as BackendKernel<f32>>::LANE_COUNT as f32);
/// ```
#[doc(hidden)]
pub trait BackendKernel<T: crate::scalar::Scalar>:
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
        "BackendKernel::LANE_COUNT exceeds MAX_SIMD_LANES; widen the scalar-fallback stack buffers"
    );

    /// Loop unrolling register accumulation factor to break loop-carried dependency chains.
    const UNROLL_FACTOR: usize = 4;

    /// Whether this scalar/backend pair requires the x86 F16C feature in
    /// addition to the architecture marker's ordinary target features.
    ///
    /// This is consumed only by target-feature dispatch. It keeps the public
    /// architecture marker singular while allowing reduced-precision kernels
    /// to enter the complete feature frame once at their operation boundary.
    #[doc(hidden)]
    const REQUIRES_F16C: bool = false;

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
    /// use hermes_simd_core::kernel::BackendKernel;
    ///
    /// #[repr(align(64))]
    /// struct AlignedBuf([f32; 4]);
    ///
    /// let buf = AlignedBuf([1.0, 2.0, 3.0, 4.0]);
    /// // SAFETY: buf is 64-byte aligned and valid for LANE_COUNT reads.
    /// let v = unsafe { <Scalar as BackendKernel<f32>>::load_aligned(buf.0.as_ptr()) };
    /// let sum: f32 = unsafe { <Scalar as BackendKernel<f32>>::sum_reduce(v) };
    /// assert_eq!(sum, 10.0_f32);
    /// ```
    unsafe fn load_aligned(ptr: *const T) -> Self::Vector;

    /// Load a vector from an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for reads.
    unsafe fn load_unaligned(ptr: *const T) -> Self::Vector;

    /// Whether this backend emits a measured read-prefetch instruction.
    const SUPPORTS_READ_PREFETCH: bool = false;

    /// Hints that a scalar address will be read by a future kernel iteration.
    ///
    /// The default is a no-op. Backends override it only where controlled
    /// measurement shows that an architecture read-prefetch instruction
    /// improves the consuming kernel.
    ///
    /// # Safety
    ///
    /// The processor must support this backend's target features, and `ptr`
    /// must be valid for one `T` read.
    #[inline(always)]
    unsafe fn prefetch_read(_ptr: *const T) {}

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

    /// Attempts a backend-native equal-lane numeric conversion into `destination`.
    ///
    /// The default reports that no native route exists, so callers retain the
    /// canonical scalar [`crate::scalar::CastFrom`] fallback. Backends override
    /// this only for type pairs whose native instruction preserves that exact
    /// conversion contract.
    ///
    /// # Safety
    ///
    /// The processor must support this backend's target features. `destination`
    /// must be valid for writes of [`Self::LANE_COUNT`] `U` elements, and the
    /// source and destination lane counts must be equal. A `true` result
    /// guarantees that every destination lane was initialized.
    #[must_use]
    #[inline(always)]
    unsafe fn try_cast<U>(_value: <Self as BackendKernel<T>>::Vector, _destination: *mut U) -> bool
    where
        U: crate::scalar::Scalar + crate::scalar::CastFrom<T>,
    {
        false
    }

    /// Whether this backend provides a *non-temporal* (cache-bypassing) store
    /// via [`store_streaming`](Self::store_streaming). Backends leaving this
    /// `false` keep the regular store default; callers gate the streaming path
    /// on this const so it is a compile-time branch, dead-code-eliminated where
    /// unsupported.
    const SUPPORTS_NT_STORE: bool = false;

    /// Store a vector with a non-temporal (streaming) hint that bypasses the
    /// cache, avoiding the read-for-ownership traffic a normal write-allocate
    /// pays for write-only data larger than the last-level cache (measured 1.71×
    /// on out-of-LLC AVX2 f32 elementwise writes; see `streaming_bench`).
    ///
    /// The default is a normal aligned store — correct but not cache-bypassing —
    /// so a backend without a non-temporal instruction inherits safe behavior.
    /// After a run of streaming stores the caller must issue
    /// [`stream_write_barrier`](Self::stream_write_barrier) before the results
    /// are read, since non-temporal stores are weakly ordered.
    ///
    /// # Safety
    /// `ptr` must be valid for writes and aligned to `LANE_COUNT * size_of::<T>()`
    /// bytes (non-temporal stores fault on misalignment).
    #[inline(always)]
    unsafe fn store_streaming(ptr: *mut T, val: Self::Vector) {
        Self::store_aligned(ptr, val);
    }

    /// Fence ordering this backend's non-temporal stores before subsequent
    /// reads. No-op by default (only meaningful where
    /// [`store_streaming`](Self::store_streaming) is a weakly ordered
    /// non-temporal store).
    #[inline(always)]
    fn stream_write_barrier() {}

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

    /// Fused multiply-subtract: `(a * b) - c`.
    ///
    /// The default negates `c` exactly and delegates to [`Self::fmadd`],
    /// preserving the single rounding of the multiply-accumulate operation.
    /// Backends with a native multiply-subtract instruction override it.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn fmsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        // SAFETY: the caller established this backend's target features, and
        // `c` is a register belonging to the same sealed backend.
        let negated = unsafe { Self::neg(c) };
        // SAFETY: the caller established this backend's target features, and
        // all operands are registers belonging to the same sealed backend.
        unsafe { Self::fmadd(a, b, negated) }
    }

    /// Horizontal sum of all lanes.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use hermes_simd_intrinsics::Scalar;
    /// use hermes_simd_core::kernel::BackendKernel;
    ///
    /// let data = [1.0_f32, 2.0, 3.0, 4.0];
    /// // SAFETY: Scalar requires no ISA feature; pointer is valid for LANE_COUNT reads.
    /// let v = unsafe { <Scalar as BackendKernel<f32>>::load_unaligned(data.as_ptr()) };
    /// let total: f32 = unsafe { <Scalar as BackendKernel<f32>>::sum_reduce(v) };
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
    ///
    /// Default: scalar-emulated merge via `kernel_helpers::generic_masked_load`.
    /// Backends with a native masked load (AVX-512, SVE) override this.
    unsafe fn masked_load_unaligned(
        ptr: *const T,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        crate::kernel_helpers::generic_masked_load::<T, Self>(ptr, mask, src)
    }

    /// Masked store: active lanes written to `ptr`, inactive lanes left unchanged.
    ///
    /// # Safety
    /// `ptr` must be valid for writing `LANE_COUNT` elements.
    ///
    /// Default: scalar-emulated merge via `kernel_helpers::generic_masked_store`.
    /// Backends with a native masked store override this.
    unsafe fn masked_store_unaligned(ptr: *mut T, mask: Self::Mask, val: Self::Vector) {
        crate::kernel_helpers::generic_masked_store::<T, Self>(ptr, mask, val);
    }

    /// Loads active lanes from a pointer with only `valid_lanes` accessible elements.
    ///
    /// Inactive lanes retain their value from `src`. Unlike
    /// [`masked_load_unaligned`](Self::masked_load_unaligned), this operation
    /// does not access inactive lanes and therefore supports allocation and page
    /// boundaries without a full-width staging buffer.
    ///
    /// # Safety
    ///
    /// The processor must support this backend's target features,
    /// `valid_lanes <= LANE_COUNT`, every active mask lane must be less than
    /// `valid_lanes`, and `ptr` must be valid for reading those active elements.
    #[inline(always)]
    unsafe fn masked_load_partial(
        ptr: *const T,
        valid_lanes: usize,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        debug_assert!(valid_lanes <= Self::LANE_COUNT);
        let valid_mask = if valid_lanes == u64::BITS as usize {
            u64::MAX
        } else {
            (1_u64 << valid_lanes) - 1
        };
        debug_assert_eq!(Self::mask_to_bitmask(mask) & !valid_mask, 0);
        // SAFETY: the caller guarantees validity for every active mask lane;
        // the generic implementation dereferences active lanes only.
        unsafe { crate::kernel_helpers::generic_masked_load::<T, Self>(ptr, mask, src) }
    }

    /// Stores active lanes to a pointer with only `valid_lanes` accessible elements.
    ///
    /// Inactive lanes are not read or written. Unlike
    /// [`masked_store_unaligned`](Self::masked_store_unaligned), this operation
    /// supports allocation and page boundaries without a full-width staging
    /// buffer.
    ///
    /// # Safety
    ///
    /// The processor must support this backend's target features,
    /// `valid_lanes <= LANE_COUNT`, every active mask lane must be less than
    /// `valid_lanes`, and `ptr` must be valid for writing those active elements.
    #[inline(always)]
    unsafe fn masked_store_partial(
        ptr: *mut T,
        valid_lanes: usize,
        mask: Self::Mask,
        val: Self::Vector,
    ) {
        debug_assert!(valid_lanes <= Self::LANE_COUNT);
        let valid_mask = if valid_lanes == u64::BITS as usize {
            u64::MAX
        } else {
            (1_u64 << valid_lanes) - 1
        };
        debug_assert_eq!(Self::mask_to_bitmask(mask) & !valid_mask, 0);
        // SAFETY: the caller guarantees validity for every active mask lane;
        // the generic implementation dereferences active lanes only.
        unsafe { crate::kernel_helpers::generic_masked_store::<T, Self>(ptr, mask, val) }
    }

    // -------------------------------------------------------------------------
    // Masked Arithmetic (merge masking)
    // -------------------------------------------------------------------------

    /// Masked elementwise add: active lanes compute `a + b`, inactive lanes yield `src`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    ///
    /// Default: `blend(mask_to_vector(mask), add(a, b), src)`. Backends with a
    /// native masked add override this.
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Self::blend(Self::mask_to_vector(mask), Self::add(a, b), src)
    }

    /// Masked elementwise multiply: active lanes compute `a * b`, inactive lanes yield `src`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    ///
    /// Default: `blend(mask_to_vector(mask), mul(a, b), src)`. Backends with a
    /// native masked multiply override this.
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Self::blend(Self::mask_to_vector(mask), Self::mul(a, b), src)
    }

    /// Masked fused multiply-add: active lanes compute `(a * b) + c`, inactive lanes retain `c`.
    ///
    /// The merge source for inactive lanes is the addend `c`, matching AVX-512 semantics
    /// for `_mm512_mask_fmadd_ps(a, mask, b, c)`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    ///
    /// Default: `blend(mask_to_vector(mask), fmadd(a, b, c), c)` — inactive lanes
    /// retain the addend `c`. Backends with a native masked FMA override this.
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        Self::blend(Self::mask_to_vector(mask), Self::fmadd(a, b, c), c)
    }

    /// Masked horizontal sum: only lanes where `mask[i]=1` contribute.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    ///
    /// Default: `sum_reduce(blend(mask_to_vector(mask), v, zero))` — inactive
    /// lanes contribute zero. Backends with a native masked reduction override this.
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> T {
        Self::sum_reduce(Self::blend(Self::mask_to_vector(mask), v, Self::zero()))
    }

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
    // Scatter (indirect indexed store)
    // -------------------------------------------------------------------------

    /// Scatter: store lane `i` of `val` to `base + indices[i]` for each lane `i`.
    ///
    /// The write-side dual of [`BackendKernel::gather`]. When `indices` repeats a
    /// value the highest lane holding it wins, matching the hardware
    /// last-writer-wins rule; callers needing a deterministic combine over
    /// duplicate indices must deduplicate before scattering.
    ///
    /// Default: a lane-sequential store loop. AVX-512 overrides this with
    /// `vscatterdps`/`vscatterdpd`; AVX2 and NEON have no scatter instruction
    /// and keep the default.
    ///
    /// # Safety
    /// All `base + indices[i]` must be valid for writes.
    unsafe fn scatter(base: *mut T, indices: Self::IndexVector, val: Self::Vector) {
        crate::kernel_helpers::generic_scatter::<T, Self>(base, indices, val);
    }

    /// Masked scatter: store only the lanes active in `mask`.
    ///
    /// Inactive lanes' indices are never dereferenced, so they may be out of
    /// range — which is what makes this the tail-safe form.
    ///
    /// # Safety
    /// Active `base + indices[i]` must be valid for writes.
    unsafe fn scatter_masked(
        base: *mut T,
        indices: Self::IndexVector,
        mask: Self::Mask,
        val: Self::Vector,
    ) {
        crate::kernel_helpers::generic_scatter_masked::<T, Self>(base, indices, mask, val);
    }

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
    /// Bits at and above `LANE_COUNT` are ignored. Default: expands to a
    /// boolean array then calls `mask_from_bools` — the natural form for the
    /// `[bool; N]`-masked emulated backends. Native-mask backends override
    /// register-only: AVX-512 truncates the bitmask into the k-register
    /// directly (the mask type *is* the bitmask), AVX2 broadcasts and
    /// compare-equals against per-lane bit constants, and NEON `vtst`s
    /// against a lane-bit table — the bit-packed sparse kernels call this
    /// once per vector, so the expansion is on their hot path.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[must_use]
    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask {
        crate::kernel_helpers::generic_mask_from_bitmask::<T, Self>(bm)
    }

    /// Convert the native mask back to a vector register where active lanes
    /// are set to `T::ALL_ONES` and inactive lanes to `T::ZERO`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector;

    /// Convert a comparison-result vector into the native mask, the inverse of
    /// [`BackendKernel::mask_to_vector`].
    ///
    /// A lane is active iff its sign bit is set, matching hardware movemask
    /// semantics (`_mm256_movemask_ps` and friends). The `cmp_*` family returns
    /// `Self::Vector` with active lanes set to `T::ALL_ONES` — whose sign bit is
    /// set — so composing this with [`BackendKernel::mask_to_bitmask`] yields one
    /// bit per comparison outcome, and `trailing_zeros` then locates the first
    /// matching lane without leaving vector registers.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask;

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
        Self::store_unaligned(buf.as_mut_ptr().cast::<T>(), v);

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

        (Self::load_unaligned(out_buf.as_ptr().cast::<T>()), carry)
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
    /// use hermes_simd_core::kernel::BackendKernel;
    ///
    /// // SAFETY: Scalar backend requires no ISA feature.
    /// let v = unsafe { <Scalar as BackendKernel<f32>>::splat(42.0_f32) };
    /// let sum: f32 = unsafe { <Scalar as BackendKernel<f32>>::sum_reduce(v) };
    /// assert_eq!(sum, 42.0_f32 * <Scalar as BackendKernel<f32>>::LANE_COUNT as f32);
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
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(
            a,
            b,
            eunomia::NumericElement::bitand,
        )
    }

    /// Elementwise bitwise OR: `a | b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(a, b, eunomia::NumericElement::bitor)
    }

    /// Elementwise bitwise XOR: `a ^ b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_binary_op::<T, Self, _>(
            a,
            b,
            eunomia::NumericElement::bitxor,
        )
    }

    /// Elementwise absolute value.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(a, eunomia::NumericElement::abs)
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
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(a, eunomia::NumericElement::sqrt)
    }

    /// Elementwise reciprocal square root, `1/√x`, to full `T` precision (~1 ulp).
    ///
    /// Native backends override this where a faster full-precision path exists: f32
    /// uses a hardware `rsqrt` seed plus one Newton–Raphson step (which already
    /// reaches f32's 23-bit mantissa); f64 has no `rsqrt` approximation accurate
    /// enough for its 52-bit mantissa, so it uses the correctly-rounded hardware
    /// `sqrt` + divide. The result is therefore precision-consistent across every
    /// backend — not a reduced-accuracy fast approximation.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(a, |x| T::ONE / x.sqrt())
    }

    /// Elementwise floor: the largest integer ≤ each lane.
    ///
    /// Matches the scalar `floor` contract bit-exactly on every backend,
    /// including the NaN/±Inf/signed-zero behavior. Native overrides use the
    /// hardware directed-rounding instructions, which implement the same
    /// round-toward-minus-infinity semantics as libm `floorf`/`floor`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn floor(a: Self::Vector) -> Self::Vector
    where
        T: crate::scalar::FloatElement,
    {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(
            a,
            <T as crate::scalar::FloatElement>::floor,
        )
    }

    /// Elementwise ceiling: the smallest integer ≥ each lane.
    ///
    /// Matches the scalar `ceil` contract bit-exactly on every backend; the
    /// hardware directed-rounding instructions implement round-toward-plus-
    /// infinity, identical to libm `ceilf`/`ceil`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn ceil(a: Self::Vector) -> Self::Vector
    where
        T: crate::scalar::FloatElement,
    {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(
            a,
            <T as crate::scalar::FloatElement>::ceil,
        )
    }

    /// Elementwise round to the nearest integer, ties to the even neighbor.
    ///
    /// This is the SIMD-hardware rounding contract: x86 `roundps`/
    /// `vrndscaleps` `_MM_FROUND_TO_NEAREST_INT` and NEON `FRINTN` both resolve
    /// exact halfway values to the even integer. The scalar default therefore
    /// uses [`crate::scalar::RoundTiesEven`], NOT libm's `round`
    /// (half-away-from-zero), so a native override and its default agree
    /// bit-exactly on every input including ties, negatives, ±Inf, NaN, and
    /// signed zeros.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn round(a: Self::Vector) -> Self::Vector
    where
        T: crate::scalar::RoundTiesEven,
    {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(
            a,
            crate::scalar::RoundTiesEven::round_ties_even,
        )
    }

    /// Elementwise truncation toward zero.
    ///
    /// Matches the scalar `trunc` contract bit-exactly on every backend; the
    /// hardware directed-rounding instructions implement round-toward-zero,
    /// identical to libm `truncf`/`trunc`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn trunc(a: Self::Vector) -> Self::Vector
    where
        T: crate::scalar::FloatElement,
    {
        crate::kernel_helpers::generic_unary_op::<T, Self, _>(
            a,
            <T as crate::scalar::FloatElement>::trunc,
        )
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
            if x == y {
                T::ZERO
            } else {
                T::ALL_ONES
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
        crate::kernel_helpers::generic_horizontal_reduce::<T, Self>(
            v,
            T::ZERO,
            eunomia::NumericElement::bitor,
        )
    }

    /// Horizontal bitwise XOR across all lanes.
    ///
    /// Default: scalar lane-by-lane scan using [`crate::scalar::NumericElement::bitxor`].
    /// Target-specific intrinsics override this.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn horizontal_bitwise_xor(v: Self::Vector) -> T {
        crate::kernel_helpers::generic_horizontal_reduce::<T, Self>(
            v,
            T::ZERO,
            eunomia::NumericElement::bitxor,
        )
    }

    // -------------------------------------------------------------------------
    // Cross-Lane Permutes
    // -------------------------------------------------------------------------
    //
    // General lane reordering, as opposed to the adjacent-pair shuffles below —
    // those are shaped for interleaved complex and express nothing else.
    //
    // All three are defined on the *flat* lane sequence, never per 128-bit
    // sub-lane. That distinction matters on x86: `_mm256_unpacklo_ps` and
    // friends operate within 128-bit halves, so they do not implement
    // `interleave` as specified here and cannot be dropped in as overrides
    // without additional cross-half permutes.
    //
    // `deinterleave` is the exact inverse of `interleave`, and `reverse` is its
    // own inverse; both identities are exercised as round-trip properties.

    /// Reverse lane order: `[a0, a1, ..., a_{n-1}] -> [a_{n-1}, ..., a1, a0]`.
    ///
    /// Default: scalar emulation via store/reverse/load. Backends override with
    /// a full cross-lane permute (`_mm256_permutevar8x32_ps`,
    /// `_mm256_permute4x64_pd`, `_mm512_permutexvar_ps`, NEON `vrev` plus a
    /// half swap).
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn reverse(v: Self::Vector) -> Self::Vector {
        const { Self::LANE_BOUND_CHECK };
        let lanes = Self::LANE_COUNT;
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        Self::store_unaligned(buf.as_mut_ptr().cast::<T>(), v);
        buf[..lanes].reverse();
        Self::load_unaligned(buf.as_ptr().cast::<T>())
    }

    /// Interleave two vectors lane-wise, returning the low and high halves of
    /// the interleaved `2n`-lane sequence.
    ///
    /// The conceptual result is `[a0, b0, a1, b1, ..., a_{n-1}, b_{n-1}]`; the
    /// first `n` elements are returned as `.0` and the last `n` as `.1`.
    ///
    /// Default: scalar emulation. This is the flat interleave, not the x86
    /// in-128-bit-lane `unpack` semantics — see the module note above.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        const { Self::LANE_BOUND_CHECK };
        let lanes = Self::LANE_COUNT;
        let mut buf_a = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut buf_b = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        Self::store_unaligned(buf_a.as_mut_ptr().cast::<T>(), a);
        Self::store_unaligned(buf_b.as_mut_ptr().cast::<T>(), b);

        let mut lo = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut hi = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        for i in 0..lanes {
            // Flat position `i` of the 2n-lane interleaving takes lane `i / 2`
            // of `a` when `i` is even and of `b` when odd; the high half
            // continues the same pattern from flat position `lanes`.
            let (src_lo, src_hi) = (i, i + lanes);
            let pick = |flat: usize| {
                let lane = flat / 2;
                if flat % 2 == 0 {
                    buf_a[lane].assume_init()
                } else {
                    buf_b[lane].assume_init()
                }
            };
            lo[i].write(pick(src_lo));
            hi[i].write(pick(src_hi));
        }
        (
            Self::load_unaligned(lo.as_ptr().cast::<T>()),
            Self::load_unaligned(hi.as_ptr().cast::<T>()),
        )
    }

    /// Deinterleave two vectors, the exact inverse of [`BackendKernel::interleave`].
    ///
    /// Treating `a` followed by `b` as one `2n`-lane sequence, `.0` collects its
    /// even-indexed lanes and `.1` its odd-indexed lanes, so
    /// `deinterleave(interleave(x, y)) == (x, y)` for every backend.
    ///
    /// Default: scalar emulation.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        const { Self::LANE_BOUND_CHECK };
        let lanes = Self::LANE_COUNT;
        let mut buf_a = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut buf_b = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        Self::store_unaligned(buf_a.as_mut_ptr().cast::<T>(), a);
        Self::store_unaligned(buf_b.as_mut_ptr().cast::<T>(), b);

        let mut even = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut odd = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        for i in 0..lanes {
            let pick = |flat: usize| {
                if flat < lanes {
                    buf_a[flat].assume_init()
                } else {
                    buf_b[flat - lanes].assume_init()
                }
            };
            even[i].write(pick(2 * i));
            odd[i].write(pick(2 * i + 1));
        }
        (
            Self::load_unaligned(even.as_ptr().cast::<T>()),
            Self::load_unaligned(odd.as_ptr().cast::<T>()),
        )
    }

    // -------------------------------------------------------------------------
    // Adjacent-Pair Shuffles & Alternating FMA (interleaved complex support)
    // -------------------------------------------------------------------------
    //
    // These methods are the minimal primitive set required to express
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
        Self::store_unaligned(buf.as_mut_ptr().cast::<T>(), v);
        let mut i = 0usize;
        while i + 1 < lanes {
            buf.swap(i, i + 1);
            i += 2;
        }
        Self::load_unaligned(buf.as_ptr().cast::<T>())
    }

    /// Swap adjacent lane *pairs*: `[a0, a1, a2, a3, a4, a5, a6, a7]` becomes
    /// `[a2, a3, a0, a1, a6, a7, a4, a5]`.
    ///
    /// On interleaved complex data each pair is one sample, so this exchanges
    /// neighbouring complex samples — the operand pairing of a
    /// distance-one butterfly held in registers. A trailing pair with no
    /// neighbour (`LANE_COUNT` not a multiple of four, NEON f64's two-lane
    /// register included) passes through unchanged, extending the lone-lane
    /// convention documented above to pair granularity.
    ///
    /// Default: scalar emulation via store/swap/load. x86 backends override
    /// with `permute`/`shuffle` forms at the 128-bit or 64-bit-pair
    /// granularity the width dictates.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn swap_pairs(v: Self::Vector) -> Self::Vector {
        const { Self::LANE_BOUND_CHECK };
        let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let lanes = Self::LANE_COUNT;
        Self::store_unaligned(buf.as_mut_ptr().cast::<T>(), v);
        let mut i = 0usize;
        while i + 3 < lanes {
            buf.swap(i, i + 2);
            buf.swap(i + 1, i + 3);
            i += 4;
        }
        Self::load_unaligned(buf.as_ptr().cast::<T>())
    }

    /// Transposes a square tile of `LANE_COUNT` vectors in place: lane `c`
    /// of row `r` moves to lane `r` of row `c` — the natural in-register
    /// granularity for blocked matrix transposes, where each
    /// `LANE_COUNT x LANE_COUNT` block loads, transposes in registers, and
    /// stores without touching memory in between.
    ///
    /// Default: scalar emulation via symmetric pair swaps staged through two
    /// row buffers (`MAX_SIMD_LANES` elements each because associated consts
    /// cannot size arrays on stable — never a `MAX_SIMD_LANES`-squared frame,
    /// which reserved 16–32 KiB of stack for tiles that are at most
    /// `LANE_COUNT`²; overriding backends never pay it). AVX2 f32/f64 override
    /// with unpack/cross-half permute networks; NEON f32 uses a `trn`/`zip`
    /// network; AVX-512 f64 and f32 use three- and four-stage
    /// `unpack`/`shuffle_fNxM` block networks. NEON f64 takes the generic
    /// default, which measured faster than a `trn` override.
    ///
    /// # Safety
    /// Processor must support the required target feature. `tile` must hold
    /// exactly `LANE_COUNT` vectors.
    #[inline(always)]
    unsafe fn transpose_square(tile: &mut [Self::Vector]) {
        const { Self::LANE_BOUND_CHECK };
        let lanes = Self::LANE_COUNT;
        debug_assert_eq!(tile.len(), lanes, "tile must hold LANE_COUNT rows");
        // In-place transpose: each symmetric pair (r, c) / (c, r) with r < c
        // swaps exactly once, staged through per-row lane buffers; diagonal
        // elements stay in place. `store_unaligned` initializes lanes
        // `0..lanes` of each buffer before any is read, and only those lanes
        // are accessed.
        let mut row_r = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        let mut row_c = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        for r in 0..lanes {
            Self::store_unaligned(row_r.as_mut_ptr().cast::<T>(), tile[r]);
            for c in (r + 1)..lanes {
                Self::store_unaligned(row_c.as_mut_ptr().cast::<T>(), tile[c]);
                core::mem::swap(&mut row_r[c], &mut row_c[r]);
                tile[c] = Self::load_unaligned(row_c.as_ptr().cast::<T>());
            }
            tile[r] = Self::load_unaligned(row_r.as_ptr().cast::<T>());
        }
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
        Self::store_unaligned(buf.as_mut_ptr().cast::<T>(), v);
        let mut out = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        for i in 0..lanes {
            let src_val = buf[i & !1].assume_init();
            out[i].write(src_val);
        }
        Self::load_unaligned(out.as_ptr().cast::<T>())
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
        Self::store_unaligned(buf.as_mut_ptr().cast::<T>(), v);
        let mut out = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
        for i in 0..lanes {
            let src_val = buf[(i | 1).min(lanes - 1)].assume_init();
            out[i].write(src_val);
        }
        Self::load_unaligned(out.as_ptr().cast::<T>())
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
