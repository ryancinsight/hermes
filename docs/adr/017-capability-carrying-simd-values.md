# ADR 017: Capability-Carrying SIMD Values

## Status
Accepted

## Context

ADR 016 added a consumer-facing target-feature entry and stated that holding an
`Arch`-parameterized value proves that the host can execute that architecture.
The implementation did not enforce that statement. `Vector::new` and
`Mask::new` were safe public constructors over raw registers, while almost every
operator, reduction, permutation, comparison, and view load repeated
`Arch::is_runtime_supported()` before invoking the backend operation.

The entry baseline before this decision applies
the interleaved complex butterfly used by Apollo to 256, 1,024, and 4,096 f64
scalars. It compares the checked slice API, a `SimdView` API, and the same
Hermes backend operations directly inside one `vectorize` dispatch. At 256 and
1,024 scalars, both safe routes take 232--236 ns and 885--903 ns respectively,
while the direct route takes 39--40 ns and 169--170 ns: a 5.3--6.0x gap. At
4,096 scalars the safe routes take 3.49--3.52 microseconds and the direct route
takes 1.78--1.80 microseconds, where memory traffic reduces the ratio to about
2x. `SimdView` therefore does not hoist the cost in the current implementation;
it repeats the same support assertion in every load, store, and arithmetic
operation.

The audit also found an independent correctness defect in chunk alignment.
`SimdChunks` preserves its parent's `Align` type on every register-width child.
That is false when the parent is over-aligned: for example, the second 32-byte
AVX2 chunk of a 64-byte-aligned f64 buffer starts 32 bytes past the base and is
not 64-byte aligned. Reconstructing `SimdView<Aligned<64>>` for that child then
panics. Immutable, mutable, and zipped chunk iterators share the defect.

The required contract is one host-support check at the boundary that creates a
safe architecture-bearing value, followed by check-free operations on that
value. Processor capabilities are process-wide and do not disappear while a
value is alive, so repeating the probe cannot strengthen the proof.

The reference comparison also exposed one operation-surface omission. PhastFT's
planar radix-2 kernel computes `2 * in0 - out0` with a uniform fused
multiply-subtract. Hermes exposed fused multiply-add and alternating
multiply-add/subtract, but no uniform fused subtract. Measuring a separate
multiply and subtract would confound both throughput and the single-rounding
numeric contract, so the comparison requires the operation on both substrates.

## Decision

`Vector<T, Arch>`, `Mask<T, Arch>`, and `SimdView<..., Arch, ...>` are
capability-carrying values.

- Raw `Vector` and `Mask` constructors are crate-private. Safe initial
  constructors retain their runtime support check; unsafe initial constructors
  retain a caller obligation that the host supports `Arch`.
- Operations requiring an existing `Vector`, `Mask`, or `SimdView` do not probe
  the host again. Their backend calls rely on the capability invariant carried
  by the input value.
- Safe slice constructors that create the first `Vector` remain checked. They
  are convenience boundaries, not the throughput API for a loop.
- A `SimdView` validates host support and base alignment once. Child views are
  constructed internally from its proven pointer and lifetime without another
  host probe.
- Register-width chunk iterators yield `Unaligned` child views. A parent
  alignment greater than the register stride is not preserved by every child;
  unaligned register loads are the only generally valid static contract.
- `vectorize` remains the target-feature scope entry chosen by ADR 016 and now
  passes a zero-sized `Simd<T, Arch>` capability into `LaneKernel::call`.
  `Simd` can be created safely only by the probed dispatcher; its public unsafe
  constructor retains the explicit host-support obligation needed by forced
  backend tests. It constructs views without another probe.
- `Simd::zero`, `Simd::splat`, and `Simd::mask_from_bitmask` construct constants
  and masks within that proven scope. Standalone `Vector::zero`/`splat` and
  `Mask::from_bitmask` retain their checked or unsafe construction contracts for
  callers outside `vectorize`. `Mask::to_bitmask` is safe because an existing
  mask already carries the host-support proof.
- Complete-lane iterators yield `SimdChunk` rather than a dynamically sized
  `SimdView`. The iterator proves that every item has exactly
  `Arch::LANE_COUNT` elements; `SimdChunk::load` and the mutable
  `SimdChunk::store` consume that proof without a per-item bounds branch.
- `Simd::io_chunks` accepts const-generic groups of planar input and output
  slices, computes their shortest complete-lane prefix once, and yields all
  planes under one loop limit. `into_remainders` starts at the iterator's
  current position, so early termination cannot omit unprocessed lane groups.
- `Vector::mul_sub` and the arithmetic facet's `fmsub` preserve one fused
  rounding. AVX2 and AVX-512 use native FMA instructions; NEON uses fused
  multiply-add with an exact sign change; scalar and default paths use the
  scalar/backend fused multiply-add contract rather than separate arithmetic.
- Immutable chunk iterators transfer shared access across threads and therefore
  require `T: Sync` for `Send`. Mutable-only iterators require `T: Send`, while
  a mutable/immutable zipped iterator requires both. The unsafe implementations
  state those auto-trait obligations at the pointer-owning types.

### Comparison capability matrix

| Concern | Hermes after this decision | `fearless_simd` 0.7 | PhastFT 0.4.1 production kernel |
| --- | --- | --- | --- |
| Runtime entry | `vectorize` passes `Simd<T, A>` | `dispatch!` passes `S: Simd` | Uses `dispatch!` |
| Native f64 vector | `Vector<f64, A>` at `A::LANE_COUNT` | `S::f64s` | Uses fixed `f64x8` in the audited DIT kernels |
| Probe-free constants | `simd.zero()` / `simd.splat(value)` | `S::f64s::splat(simd, value)` | Uses fixed-width `splat` |
| Uniform fused subtract | `Vector::mul_sub` | `SimdBase::mul_sub` | Uses `mul_sub` for `2 * in0 - out0` |
| Complete lane groups | `SimdChunk`; `Simd::io_chunks` shares one limit across planar inputs and outputs | Exact slices loaded into the selected vector type | Fixed-width slices, often in multiple groups per codelet |

The benchmark uses the native-width row for both substrates. PhastFT's fixed
`f64x8` choice maps to two AVX2 registers and changes unrolling, instruction-level
parallelism, and register pressure; it is a separate fixed-width capability and
kernel-scheduling question, not evidence about wrapper overhead.

This is a breaking public API correction. Code constructing `Vector` or `Mask`
from raw registers must use the existing unsafe load/conversion boundary or a
checked safe constructor. Code requiring an aligned chunk must establish that
alignment for that specific chunk instead of inheriting an invalid parent
claim. No compatibility alias or forwarding constructor is retained.

## Rejected Alternatives

1. **Cache each support result globally and retain per-operation checks.** This
   lowers probe cost but preserves an invalid ownership model and still adds a
   branch/load to every operation in a throughput loop.
2. **Trust `LaneKernel::call` without a capability argument.** The method is
   public and can be called directly with any architecture marker, so its type
   parameter alone proves neither runtime support nor target-feature entry.
3. **Preserve aligned chunk item types when the register stride happens to
   divide the parent alignment.** That requires a relation between runtime lane
   width and the alignment type that the current public type cannot express.
   Returning `Unaligned` is exact for every backend and still permits the
   backend's unaligned register instruction, which is the operation needed here.
4. **Move the correction into Apollo.** Raw pointers would avoid the checks but
   would reproduce Hermes' safety boundary and leave every other consumer with
   the same cost and alignment defect.

## Consequences

- Existing register operations become zero-extra-branch wrappers over their
  backend operations. Safe loop setup pays support and alignment validation
  once per view rather than once per lane operation.
- A consumer-authored lane kernel pays one backend probe in `vectorize`; all
  views created from its `Simd` capability and all complete chunks derived from
  those views are probe-free. The complete-lane loop retains only its loop
  control branch, not one bounds branch per input and output.
- The benchmark above remains the regression instrument. Each safe candidate
  is value-checked against the scalar butterfly before timing, and its median
  confidence interval is compared with the direct Hermes ceiling on the same
  host and in the same binary.
- The same binary contains planar f32 and f64 groups performing identical
  arithmetic through Hermes and `fearless_simd` 0.7 native-width vectors. One
  generic instrument supplies both precisions. Both paths use exact chunk
  iterators so slice lengths are hoisted out of the timed loop; an earlier
  offset-range version was rejected after assembly showed ten retained
  per-iteration bounds branches in only the reference path.
- Every candidate reuses the same output allocations. Separate output vectors
  were rejected after identical hot loops produced size-dependent timing
  inversions consistent with address/cache-set placement rather than substrate
  cost.
- On the pinned AVX2 host, corrected same-address medians and 95% confidence
  intervals are:

  | f64 scalars | Hermes | `fearless_simd` | median delta |
  | ---: | ---: | ---: | ---: |
  | 256 | 77.024 ns [76.444, 77.624] | 76.687 ns [76.378, 77.030] | +0.44% |
  | 1,024 | 1.0134 us [1.0070, 1.0201] | 1.0022 us [0.99266, 1.0130] | +1.12% |
  | 4,096 | 3.9417 us [3.9127, 3.9685] | 3.9392 us [3.9016, 4.0025] | +0.06% |

  All confidence intervals overlap. The separate Hermes view/direct diagnostic
  reports 56.405/55.837 ns, 158.58/158.39 ns, and 2.4103/2.4078 us at the same
  lengths. Assembly confirms both planar AVX2 loops have six vector loads, four
  stores, fused arithmetic, one loop-control branch, and no calls, feature
  probes, bounds branches, or panic paths in the hot loop.
- The exact locked bounded f32 run reports Hermes/reference medians
  of 32.175/32.090 ns, 169.18/161.70 ns, and 2.0452/2.0376 us at 256, 1,024,
  and 4,096 scalars. Every 95% confidence interval overlaps. Both emitted AVX2
  hot loops contain six 256-bit loads, four stores, six fused arithmetic
  instructions, one loop branch, and no calls or bounds branches. Timings are
  compared only within one run; affinity experiments on the shared hybrid-core
  host widened variance and are not retained as evidence.
- The same-address cross-lane instrument covers native f32/f64 interleave and
  deinterleave. Two unchanged runs moved absolute medians by 15--55% and changed
  or converged candidate ordering, so they do not establish a stable speed
  difference. Exact AVX2 loops have equal load, shuffle, store, call, and branch
  classes for f32 and interleave f64. The two four-shuffle f64 deinterleave
  sequences both model at 4.0 cycles per iteration under `llvm-mca` 22.1.8 for
  Arrow Lake S. No production change follows from unstable wall-clock ordering
  with equivalent hot-loop structure; the full intervals remain in
  `gap_audit.md`.
- AVX-512 floating bitwise operations use AVX-512F integer-domain bitwise
  instructions around zero-cost casts. The float-typed intrinsics require
  AVX-512DQ and otherwise remained outlined calls inside an AVX-512F kernel.
  Exact signed-zero and NaN-payload negation tests pin the bit contract.
- Differential tests cover arithmetic and permutation results across every
  host-supported backend, including exact f32 and f64 fused multiply-subtract
  results. Adversarial iterator tests cover over-aligned parents, forward and
  reverse iteration, mutable children, zipped children, unequal planar lengths,
  zero and sub-register inputs, exact iterator length, early termination, and
  scalar tails.
- The unsafe backend facets remain unchanged. The safe wrappers discharge their
  target-feature obligation through construction provenance rather than
  repeated runtime assertions.
- Apollo adopts only the corrected provider surface. RustFFT and PhastFT remain
  differential and performance references, not production dependencies.

## Revisions

- 2026-08-26: Added shared cross-lane f32/f64 comparison evidence under
  `HS-FEARLESS-PERMUTE-THROUGHPUT-2026-08-26`; exact codegen and modeled
  throughput reject a provider correction despite unstable wall-clock ordering.
- 2026-08-26: Extended the accepted comparison evidence to the native f32
  contract under `HS-FEARLESS-F32-THROUGHPUT-2026-08-26`; no provider change
  followed because the pinned intervals and AVX2 hot-loop structures match.
- 2026-08-26: Accepted from the `HS-LANE-THROUGHPUT-2026-08-25` measurement.
  This narrows ADR 016's statement that an architecture marker itself is a
  capability: the proof is carried by a value whose construction is checked or
  explicitly unsafe, not by the freely nameable marker type.
