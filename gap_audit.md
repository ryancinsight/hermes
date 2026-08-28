# Gap Audit - hermes-simd

Persistent gap register. Evidence tiers follow the repository instruction
hierarchy: machine-checked proof > type-level invariant > property/fuzz >
differential/empirical > source audit.

## NUMA generation-counter test isolation (2026-08-25)

Evidence tier: reproduced failure plus source audit. Found incidentally while
running the workspace suite under bare `cargo test` rather than the committed
nextest runner.

`test_numa_locality_caching_correctness_and_invalidation` in
`crates/hermes-simd/tests/types_tests.rs` reads a process-global counter twice
and asserts exact equality across the gap:

```text
let gen_start = get_alloc_generation();
...
let gen_after_alloc = get_alloc_generation();
assert_eq!(gen_after_alloc, gen_start);
```

`get_alloc_generation` is bumped by any `AlignedVec` deallocation anywhere in
the process. Under nextest each test is its own process, so nothing else can
bump it and the assertion holds. Under bare `cargo test` the tests are threads
in one process, and three sibling NUMA tests allocate and drop `AlignedVec`s —
so the counter moves between the two reads and the assertion fails. Observed
failing once and passing on immediate re-run, which is the signature.

The committed runner therefore hides a real property of the test: it asserts on
shared mutable global state rather than on the behavior it means to check. The
intended property is "allocation alone does not bump the generation, and
deallocation does". The second half is already tested correctly and robustly
(`assert!(gen_after_drop > gen_start)`); the first half cannot be expressed as
exact equality on a shared counter, because it is a statement about what *this*
allocation did, not about the counter's absolute value.

This is not a flake to re-run. Filed as `HS-NUMA-GEN-ISOLATION-2026-08-25`.
Nothing in the current gate is wrong — nextest is the sanctioned runner and it
passes — so this was latent rather than breaking, and the risk was that the
assertion was read as verifying something it did not.

**Correction delivered 2026-08-26:** the test now launches an exact child test
process for the contract body when run normally. An environment marker makes
the child execute the assertions directly, so no recursion or polling is
involved. The global counter is consequently isolated from sibling tests under
bare `cargo test`, while the test still asserts that this allocation leaves the
counter unchanged and that dropping the allocation advances it. Focused
nextest execution and the complete 28-test shared-process integration binary
pass within the standard test budget; no production code, allocator behavior,
timeout, or runner configuration changed.


## Lane throughput against fearless_simd (2026-08-25) <a id="lane-throughput-2026-08-25"></a>

Evidence tier: measured in a consumer's build, seven kernel variants, all
engines interleaved in one process, every variant correctness-gated before its
timing was read. No claim here is drawn from a cross-run comparison.

Apollo set out to close a 7x-10x FFT throughput gap against RustFFT and PhastFT
and eliminated every algorithmic cause it could find. What remained points here.

### The measurement

Power-of-two complex f64 FFT, arithmetic rate as `5 N log2 N` flops over elapsed
time, at N=2^10 where the whole 16 KB array is L1-resident:

| kernel | flops/ns |
| --- | --- |
| planar radix-2, plain loops left to autovectorization | 4.6 |
| planar, first three stages fused | ~4.6 |
| planar, cache-oblivious recursion to a 1024-element block | ~4.6 |
| planar, Hermes `Vector` ops through `vectorize` | ~4.5 |
| interleaved, Hermes `dup_even`/`dup_odd`/`swap_adjacent`/`fmaddsub` | 4.2 |
| the same, per-call slice validation hoisted to raw loads | 6.1 |
| Apollo's own hand-written AVX Stockham | 3.4-4.3 |
| **RustFFT** | **38.5** |
| **PhastFT, built on `fearless_simd`** | **32.8** |

A scalar f64 pipeline on this host is roughly 6 flops/ns; AVX2 is roughly 48.
The external engines reach 68-80% of peak. Everything routed through Hermes'
lane surface, or through Apollo's own AVX code, sits near or below the scalar
pipeline rate.

RustFFT and PhastFT run **inside the same test binary**, compiled with the same
profile and the same flags, which excludes build configuration. The data is
L1-resident, which excludes bandwidth. Apollo's fused pass count is four passes
for ten radix-2 stages — fewer than RustFFT's radix-4 needs — which excludes
pass count. Its complex path allocates zero per call, measured with a counting
allocator, which excludes allocation.

The interleaved row is the sharpest comparison available: it uses the exact
primitive sequence RustFFT's AVX path uses for a complex multiply —
`fmaddsub(dup_even(w), b, dup_odd(w) * swap_adjacent(b))`, one shuffle pair and
one fused instruction — against the same interleaved layout, and still lands at
4.2.

### One mechanism, partially identified

Replacing `Vector::load_unaligned_from_slice(..).unwrap()` with the raw
`load_unaligned` pointer form moved that kernel from 4.2 to 6.1 flops/ns. The
per-call length and alignment validation costs about **45%** in a kernel that
issues six loads and stores per two complex elements.

This is not an argument against the checked wrappers, which exist because the
Highway audit correctly identified their absence as a gap. It is an argument
that a consumer currently has to choose between the safe surface and throughput,
and that the safe surface should not cost 45% when the bounds are loop-invariant.

Hermes already has the machinery to hoist those checks — `SimdView` typestates
carry alignment and length as type parameters, and `SimdChunks`/`ZipChunks`
iterate pre-validated blocks. The prototypes did not use them, so the remaining
question is open rather than answered: does the existing chunk-iterator surface
close the distance, and if it does, why did the first consumer to write a
fine-grained kernel not reach for it?

### Why this is filed here and not in Apollo

Apollo eliminated layout, stage fusion, cache blocking, primitive selection, and
validation strategy across seven variants spanning 3.4 to 6.1 flops/ns. Whatever
separates that band from 33-38 is not reachable from a consumer's algorithm
choices. It is a property of the lane operations themselves, which is Hermes'
bounded context.

`fearless_simd` is the reference that makes the gap legible: PhastFT is an
`#![forbid(unsafe_code)]` FFT that reaches 32.8 flops/ns on this host through a
safe SIMD abstraction. Whatever it does at the lane level, Hermes should be able
to match — and the fact that a safe abstraction achieves it removes safety as
the explanation.

### Scope note

This does not claim a defect in any specific Hermes operation. It reports that
seven independent attempts to reach competitive lane throughput through this
substrate all failed within a narrow band, that the checked-load overhead
accounts for part of it, and that the remainder is unlocated. Locating it is
`HS-LANE-THROUGHPUT-2026-08-25`.

### Resolution (2026-08-26)

The gap was not one arithmetic instruction. Hermes repeatedly probed host
support at operations on already architecture-bearing values, reconstructed
dynamically sized child views for every lane group, inherited invalid alignment
claims into over-aligned children, lacked uniform fused multiply-subtract, and
made a many-plane consumer compose iterators whose checks LLVM could not reduce
to one loop limit. ADR 017 corrects those provider contracts and adds
`Simd::io_chunks` for const-generic planar input/output groups.

The corrected same-binary comparison reuses identical output addresses for both
substrates. Its pinned AVX2 medians (95% confidence intervals) are:

| f64 scalars | Hermes | `fearless_simd` 0.7 | median delta |
| ---: | ---: | ---: | ---: |
| 256 | 77.024 ns [76.444, 77.624] | 76.687 ns [76.378, 77.030] | +0.44% |
| 1,024 | 1.0134 us [1.0070, 1.0201] | 1.0022 us [0.99266, 1.0130] | +1.12% |
| 4,096 | 3.9417 us [3.9127, 3.9685] | 3.9392 us [3.9016, 4.0025] | +0.06% |

All confidence intervals overlap. The Hermes view/direct diagnostic reports
56.405/55.837 ns, 158.58/158.39 ns, and 2.4103/2.4078 us at those lengths.
AVX2 assembly gives both planar hot loops six vector loads, four stores, fused
arithmetic, one loop branch, and no calls, probes, bounds branches, or panic
paths. The old 3.4--6.1 flops/ns rows above remain the entry evidence that
located the provider defect; they are not the corrected substrate result.

### Native f32 confirmation (2026-08-26)

The original comparison established only f64 parity. The instrument now has one
generic planar implementation instantiated for both supported floating lane
types; each pair still reuses identical inputs and output addresses and passes
the precision-specific scalar oracle before timing. The exact locked bounded
suite reports these medians and 95% confidence intervals:

| f32 scalars | Hermes | `fearless_simd` 0.7 | median delta |
| ---: | ---: | ---: | ---: |
| 256 | 32.175 ns [31.842, 32.564] | 32.090 ns [31.859, 32.415] | +0.26% |
| 1,024 | 169.18 ns [165.16, 173.71] | 161.70 ns [159.23, 165.35] | +4.63% |
| 4,096 | 2.0452 us [2.0275, 2.0646] | 2.0376 us [2.0289, 2.0514] | +0.37% |

All confidence intervals overlap. Emitted AVX2 code gives both f32 hot loops
six 256-bit loads, four stores, six fused arithmetic instructions, one loop
branch, and no calls or bounds branches. Their loop-control spellings differ,
but neither retains work absent from the other. The evidence therefore exposes
no provider-owned f32 correction. Affinity experiments on this shared hybrid-
core host widened variance and reversed candidate order, so they are rejected
rather than reported as stronger evidence. Only same-run substrate comparisons
are claimed; repeated absolute timings are not treated as stable.

## Fearless SIMD audit — amendment and closure (2026-08-25) <a id="fearless-simd-amendment"></a>

Two corrections to the audit below, found while implementing the increment it
filed. Both narrow the gap; neither changes the conclusion.

**The safe operation surface already existed.** The audit's second finding read
as though Hermes exposed only the `unsafe` facets and no safe layer above them.
It does: `Vector<T, Arch>` carries the full operator set, `zero`/`splat`,
bounds-checked slice load and store, reductions, comparisons, `blend`, and array
conversion, each asserting host support once and discharging the facet
obligation. The real gap was narrower and sharper — the safe layer was missing
exactly `fmadd` and the cross-lane permutes (`reverse`, `interleave`,
`deinterleave`, `swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd`),
which are the operations a transform or stencil kernel needs and the reason a
consumer writing one dropped to the facets. Those nine methods are now on
`Vector`.

**ADR 011 needed no revision.** The audit proposed revising its exclusion of
`SimdKernel` from the safe-surface argument. That exclusion is correct and the
increment follows it: the facets stay `unsafe` as the implementation seam, and
the safe layer sits above them in `Vector` — which is precisely the structure
ADR 011 chose when it left `KoggeStone`'s ISA fills unsafe beneath a safe trait.
The new decision is recorded as ADR 016 instead.

**Closure.** `HS-FEARLESS-TOKEN-2026-08-25` is delivered as
`hermes_simd::vectorize` plus the `LaneKernel<T>` trait, generated by the same
`#[runtime_dispatch]` attribute as every other dispatcher rather than a second
mechanism. The acceptance oracle's codegen half is met: an AXPY kernel through
the entry emits 41 ymm-bearing instructions including a real `vfmadd213ps`, with
no call into the backend operations — only the one-time CPUID cache
initialization and cold panic paths. The same body invoked as
`kernel.call::<Avx2>()` without the entry emits zero ymm in the caller and five
call sites into `hermes-simd-intrinsics`, each an outlined stub. The
consumer-shaped conformance test compiles under `#![forbid(unsafe_code)]`.

One thing surfaced that the audit did not predict: `#[runtime_dispatch]` dropped
the annotated function's doc comments, so a `pub` generated dispatcher could not
satisfy `#![deny(missing_docs)]`. That is why every dispatch module in this
crate was crate-local, and it was invisible until a public dispatcher was
wanted. The macro now forwards them, and the `#![expect(missing_docs)]` in
`dispatch/popcount.rs` that existed only for this reason is retired.

## Fearless SIMD Reference Audit - 2026-08-25 <a id="fearless-simd-2026-08-25"></a>

Reference: `https://github.com/linebender/fearless_simd` at commit
`3ac40f9aad237183f8178ffd33a8f9c71fee644a`; crates.io `fearless_simd` 0.7.0,
published 2026-08-11, MSRV 1.89.

Entry route: the reference surfaced through
`https://github.com/QuState/PhastFT` at commit
`7bbbfa5bbac8681af7d1abf6fb02990d8eacb552` (crates.io `phastft` 0.4.1), an FFT
library Apollo now tracks as an external comparison. PhastFT declares
`#![forbid(unsafe_code)]` at its crate root and writes every butterfly and
bit-reversal kernel against Fearless SIMD's safe surface, so it is a working
consumer of the authorship model audited here rather than a claim about one.
The transform-side findings live in Apollo's `gap_audit.md`.

Evidence tier: source audit of both references, local Hermes code search, and
consumer measurement at a named Apollo revision. No benchmark or codegen claim
is made from this audit alone; the codegen measurement is the acceptance oracle
of the increment it drives, not evidence already collected.

Scope fit:
- In scope for Hermes: consumer-facing kernel authorship — how a crate outside
  `hermes-simd` writes one generic lane kernel and obtains per-ISA machine code
  for it.
- Out of scope for Hermes: replacing the sealed `SimdKernel` seam, the sparse,
  packed, AMX, tensor, or COW surfaces, or the Atlas compute boundaries.
  Fearless SIMD is a portable-lane substrate only and owns none of them.

### Capability matrix

| Capability family | Hermes | `fearless_simd` 0.7 | Classification |
| --- | --- | --- | --- |
| Runtime-dispatched capability value | `vectorize` supplies `Simd<T, A>` | `dispatch!` supplies `S: Simd` | Shared after ADR 017 |
| Native floating vectors | Native-width f32/f64 `Vector<T, A>` | Native-width `S::f32s` / `S::f64s` | Shared |
| Load and store | Checked standalone loads plus capability-scoped exact chunks | Slice loads/stores on selected vectors | Shared |
| Arithmetic, FMA, and FMS | Operators, `mul_add`, alternating FMA, and `mul_sub` | Operators, `mul_add`, alternating FMA, and `mul_sub` | Shared after ADR 017 |
| Comparisons, masks, and select | Typed masks, bitmask conversion, comparisons, blend/select | Masks, comparisons, select | Shared |
| Min/max, sqrt, and rounding | Min/max, sqrt, floor/ceil/round/trunc | Corresponding floating operations | Shared |
| Interleave and deinterleave | Flat-sequence interleave/deinterleave and adjacent permutes | Interleave/deinterleave and lane rearrangement | Shared |
| Reduced precision | f16 backend surface | No corresponding audited f16 lane family | Hermes broader |
| Irregular and predicated memory | Gather/scatter, masked memory, compress/expand, streaming stores | Portable slice/interleaved I/O | Hermes broader |
| Aggregate operations | Reductions and scans | Portable lane reductions | Hermes broader |
| Domain surfaces | Sparse, tensor, copy-on-write, packed, and AMX operations | None; portable lane substrate only | Hermes broader |
| Scalable Arm route | SVE-shaped dispatch route, currently emulated under its recorded contract | No SVE level in the audited release | Hermes broader |
| Integer and fixed-width lanes | Not part of the current dense floating consumer contract | Integer lanes and fixed-width vectors | Fearless broader; non-gap |
| Width and conversion algebra | Current contract uses backend-native width | Combine/split, widen/narrow, and conversions | Fearless broader; non-gap |
| Integer/lane rearrangement | Current consumers require the shared floating permutes above | Shifts, rotates, slides, and swizzles | Fearless broader; non-gap |
| Extended interleaved I/O | Two-way interleave/deinterleave | Four-way interleaved I/O | Fearless broader; non-gap |
| Additional floating operations | Exact reciprocal square root and existing arithmetic catalog | `copysign` and approximate reciprocal | Fearless broader; non-gap |
| Additional targets | Scalar, AVX2, AVX-512, NEON, and the recorded SVE route | SSE2, SSE4.2, and WASM in addition to wider targets | Fearless broader; non-gap; SSE remains ADR 006's recorded decision |

The accepted gaps were capability provenance and repeated per-operation probes,
exact-width chunk proofs, over-aligned child correctness, uniform fused
multiply-subtract, and the immutable iterator `Send` bound (`T: Sync`, not
`T: Send`). ADR 017 resolves all five. Fearless-broader rows are not Hermes gaps
without a current consumer contract; adding unused surface would expand the
sealed provider without acceptance evidence.

Apollo subsequently supplied one narrower width contract: its register-resident
128-point and planar row kernels require exactly four scalar lanes, while
widest-native dispatch selects eight f64 lanes on AVX-512. ADR 018 resolves that
selection gap with `vectorize_lanes::<LANES, T, K>`: one operation-boundary
dispatch selects the widest supported backend at the exact count and returns
`None` without calling the kernel when none exists. This does not add Fearless's
fixed-width storage/arithmetic family; that broader algebra remains a non-gap
without a consumer requiring it. Host value tests, AArch64 Windows std/no-std
strict-warning builds, and optimized x86 codegen establish the dispatch
contract; native AArch64 execution remains hosted-CI evidence.

### Cross-lane throughput confirmation — 2026-08-26

The same-binary lane instrument now compares the shared f32/f64 interleave and
deinterleave operations. Each provider reads the same two input allocations,
writes the same two output allocations, selects the same AVX2 lane width on the
measurement host, and is checked against an exact scalar lane-order oracle
before timing. Fearless exposes no direct whole-vector reverse operation, so
Hermes' reverse remains in the native-only permute suite rather than being
compared with a synthesized reference operation.

The first exact locked run reported these medians and 95% confidence intervals
(all values in ns):

| Operation | Type | Scalars | Hermes | `fearless_simd` |
| --- | --- | ---: | ---: | ---: |
| interleave | f32 | 256 | 19.746 [19.627, 19.951] | 19.372 [19.294, 19.496] |
| interleave | f32 | 1,024 | 66.580 [66.390, 66.796] | 87.963 [74.834, 103.31] |
| interleave | f32 | 4,096 | 774.01 [745.53, 808.85] | 706.09 [699.60, 712.92] |
| interleave | f64 | 256 | 31.216 [30.841, 31.732] | 33.720 [33.369, 34.529] |
| interleave | f64 | 1,024 | 142.76 [142.41, 143.13] | 142.57 [142.02, 143.03] |
| interleave | f64 | 4,096 | 1,346.7 [1,276.2, 1,425.6] | 1,216.5 [1,211.4, 1,221.4] |
| deinterleave | f32 | 256 | 28.881 [28.767, 29.006] | 28.664 [28.620, 28.703] |
| deinterleave | f32 | 1,024 | 73.404 [72.595, 74.322] | 72.420 [72.180, 72.790] |
| deinterleave | f32 | 4,096 | 989.34 [987.50, 991.77] | 983.46 [980.61, 985.65] |
| deinterleave | f64 | 256 | 31.732 [31.695, 31.765] | 31.429 [31.359, 31.489] |
| deinterleave | f64 | 1,024 | 183.18 [182.33, 184.25] | 181.80 [181.41, 182.29] |
| deinterleave | f64 | 4,096 | 1,614.3 [1,598.8, 1,622.6] | 1,598.9 [1,561.3, 1,618.7] |

An unchanged confirmation run then reported:

| Operation | Type | Scalars | Hermes | `fearless_simd` |
| --- | --- | ---: | ---: | ---: |
| interleave | f32 | 256 | 16.726 [16.707, 16.748] | 16.492 [16.353, 16.696] |
| interleave | f32 | 1,024 | 76.804 [76.648, 77.000] | 77.392 [76.465, 78.603] |
| interleave | f32 | 4,096 | 718.08 [709.68, 727.03] | 718.11 [709.11, 727.15] |
| interleave | f64 | 256 | 41.580 [37.708, 46.657] | 35.424 [34.472, 36.962] |
| interleave | f64 | 1,024 | 153.13 [152.08, 154.94] | 151.83 [151.57, 152.17] |
| interleave | f64 | 4,096 | 1,909.4 [1,843.6, 1,953.1] | 1,820.8 [1,773.0, 1,884.3] |
| deinterleave | f32 | 256 | 19.554 [17.782, 21.438] | 18.350 [16.507, 20.237] |
| deinterleave | f32 | 1,024 | 103.31 [96.499, 110.02] | 101.26 [93.032, 110.18] |
| deinterleave | f32 | 4,096 | 711.55 [668.78, 752.25] | 676.40 [611.36, 757.16] |
| deinterleave | f64 | 256 | 46.774 [43.937, 49.305] | 50.560 [46.934, 53.348] |
| deinterleave | f64 | 1,024 | 165.50 [152.18, 184.37] | 171.88 [156.82, 190.63] |
| deinterleave | f64 | 4,096 | 1,430.7 [1,235.7, 1,576.5] | 1,434.4 [1,288.1, 1,570.1] |

The material result is the instability, not a speedup claim. Absolute medians
moved by 15--55% between the two unchanged runs on the shared hybrid-core host,
and the candidate ordering changed or converged. Exact AVX2 assembly shows the
f32 pairs have two loads, four shuffles, two stores, and one loop branch with
the same shuffle instructions in a different order. Both f64 interleave loops
use six shuffles. The f64 deinterleave loops use different four-shuffle
sequences; `llvm-mca` 22.1.8 for the host's Arrow Lake S model predicts the same
4.0-cycle block throughput (423 Hermes versus 425 Fearless modeled cycles over
100 iterations). No calls, bounds branches, or provider-only hot-loop work
explain the disjoint intervals. The evidence therefore rejects a production
SIMD correction and records that wall-clock ordering on this host is not stable
enough to distinguish equivalent cross-lane loops.

### Interleaved complex-register throughput — 2026-08-27

The interleaved instrument now compares the raw Hermes vector recipe,
`ComplexReg`, and Fearless SIMD 0.7's public
deinterleave/planar/reinterleave composition for f32 and f64. Each provider
reads the same three input allocations, writes the same two output allocations,
uses the same native AVX2 width, and processes two registers per loop. A scalar
reference checks lane order and fused-rounding semantics before every timed row.

The first draft exposed an instrument defect rather than a provider defect:
calling standalone checked vector loads inside the timed loop retained six
`std_detect` runtime-support probes per two-register iteration. The corrected
instrument uses the existing capability-bearing `Simd::io_chunks` boundary,
which hoists the support and complete-lane proofs before the loop. This is the
public hot-path contract established by ADR 017.

The first corrected exact locked run reported these medians and 95% confidence
intervals (all values in ns):

| Type | Scalars | Hermes vector | Hermes `ComplexReg` | `fearless_simd` |
| --- | ---: | ---: | ---: | ---: |
| f32 | 256 | 24.752 [24.658, 24.850] | 24.978 [24.910, 25.034] | 37.434 [37.350, 37.532] |
| f32 | 1,024 | 89.301 [88.554, 90.228] | 88.315 [88.072, 88.591] | 147.65 [146.72, 148.81] |
| f32 | 4,096 | 653.88 [650.54, 658.35] | 648.76 [644.35, 653.13] | 1,167.5 [1,109.5, 1,264.0] |
| f64 | 256 | 39.177 [38.318, 41.096] | 38.345 [38.111, 38.646] | 147.84 [111.64, 184.18] |
| f64 | 1,024 | 177.30 [171.07, 188.83] | 190.60 [176.44, 215.13] | 353.63 [325.08, 387.46] |
| f64 | 4,096 | 1,186.6 [1,179.1, 1,195.4] | 1,193.6 [1,180.4, 1,208.7] | 1,985.3 [1,590.6, 2,375.1] |

An unchanged confirmation run reported:

| Type | Scalars | Hermes vector | Hermes `ComplexReg` | `fearless_simd` |
| --- | ---: | ---: | ---: | ---: |
| f32 | 256 | 21.936 [21.890, 21.988] | 21.971 [21.907, 22.071] | 36.415 [36.275, 36.547] |
| f32 | 1,024 | 74.923 [73.815, 76.490] | 73.992 [73.266, 74.967] | 147.60 [146.91, 148.32] |
| f32 | 4,096 | 670.25 [661.53, 680.09] | 634.91 [626.88, 645.30] | 994.88 [908.25, 1,091.8] |
| f64 | 256 | 62.991 [60.949, 65.318] | 65.783 [63.303, 68.142] | 85.482 [83.371, 88.845] |
| f64 | 1,024 | 324.11 [268.89, 369.12] | 277.29 [240.97, 317.89] | 530.07 [440.90, 647.94] |
| f64 | 4,096 | 1,490.4 [1,377.3, 1,610.9] | 1,523.9 [1,414.5, 1,632.3] | 2,802.5 [2,541.8, 3,062.3] |

Absolute f64 timings remain noisy on the shared hybrid-core host, but every
Hermes-versus-Fearless confidence interval is disjoint in Hermes' favor in both
runs. Exact AVX2 code generation supplies the mechanism: each Hermes loop has
six loads, six layout shuffles or duplications, two multiplies, two alternating
fused multiply-adds, four additions/subtractions, four stores, no calls, and no
support probes. Fearless retains the same memory and arithmetic instruction
counts but requires 20 layout shuffles for f32 and 24 for f64. LLVM emits one
Hermes provider function for the raw and `ComplexReg` recipes, so the newtype is
instruction-identical to the raw recipe. The evidence rejects a production
kernel correction. It covers AVX2 on one shared Windows host; other ISAs retain
compile and value-semantic coverage rather than local timing, and a future
direct interleaved Fearless operation would require a new comparison.

### Capability-scoped single-register load — rejected 2026-08-27

Apollo's strided batched kernels receive `Simd<T, A>` but call the standalone
checked slice loader inside their fused stage loops. A candidate
`Simd::load_unaligned_from_slice` tested whether binding that load to the
existing capability could close the cost without changing the consumer's
access shape. The benchmark inputs, scalar oracle, output allocations, timed
region, sizes, and checked/view/direct variants remained unchanged; only the
candidate row was added while measuring it.

Exact AVX2 assembly showed the candidate removed every `std_detect` cache read
and initialization call. It still retained five slice-failure edges per loop:
one for the controlling input range and four for the remaining input/output
ranges. The existing `SimdView`/`SimdChunk` route retained no bounds or panic
edge in its hot loop.

The first exact locked run reported medians and 95% confidence intervals:

| f64 scalars | checked | capability candidate | view | direct |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 83.078 ns [82.545, 83.597] | 45.628 ns [45.066, 46.289] | 37.912 ns [37.507, 38.567] | 37.220 ns [37.063, 37.400] |
| 1,024 | 321.47 ns [318.44, 325.06] | 216.32 ns [197.33, 234.17] | 174.28 ns [167.05, 183.03] | 154.75 ns [150.81, 159.28] |
| 4,096 | 2.1486 us [2.1271, 2.1693] | 1.7985 us [1.7488, 1.8426] | 1.8613 us [1.8430, 1.8794] | 1.8482 us [1.8103, 1.8706] |

An unchanged confirmation run reported:

| f64 scalars | checked | capability candidate | view | direct |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 83.789 ns [83.134, 84.203] | 45.839 ns [45.291, 46.496] | 40.260 ns [38.861, 42.868] | 39.051 ns [38.748, 39.433] |
| 1,024 | 331.08 ns [327.55, 334.85] | 209.91 ns [207.58, 212.94] | 201.29 ns [198.91, 203.68] | 203.36 ns [198.77, 208.71] |
| 4,096 | 1.4869 us [1.4500, 1.5274] | 1.3893 us [1.3813, 1.3972] | 1.4027 us [1.3959, 1.4092] | 1.3965 us [1.3801, 1.4181] |

The candidate materially improved the checked route, but its 256-element
interval remained disjoint from both ceilings in both runs, and the generated
loop retained the exact branches the accepted chunk contract removes. The
predeclared acceptance oracle therefore rejects the public method. No
production source or benchmark-instrument change remains. Strided in-place
consumers partition their disjoint rows once with standard slice APIs and then
use the existing capability-backed view/chunk iterators; Hermes does not clone
that general slice-partitioning operation into a SIMD-specific API.

### Stable Rust SIMD ecosystem refresh and Pulp parity — 2026-08-27 <a id="stable-rust-simd-2026-08-27"></a>

The reference set was refreshed against the current published documentation,
not selected from recalled crate surfaces:

| Substrate | Stable same-binary dispatch | Relevant contract | Treatment |
| --- | --- | --- | --- |
| [`fearless_simd` 0.7.0](https://github.com/linebender/fearless_simd/releases/tag/v0.7.0) | Yes | Cached `Level` selection and a safe native-width token; 0.7 adds SSE2, generic widen/narrow, and four-way interleaved I/O | Existing Hermes comparator |
| [`pulp` 0.22.3](https://docs.rs/pulp/0.22.3/pulp/) | Yes | `Arch::dispatch` invokes one `WithSimd` body over native f32/f64 vectors and exact vector/tail slice splits | Measured in a rejected local experiment; its `paste` dependency fails advisory policy |
| [`macerator` 0.3.4](https://docs.rs/macerator/0.3.4/macerator/) | Capability-vectorized | Sealed `Simd`, `WithSimd`, generic `Vector`, and per-operation acceleration queries | Source-audited; the same `paste` advisory and no distinct live consumer contract reject another timing row |
| [`archmage` 0.9.28](https://docs.rs/archmage/0.9.28/archmage/) | Yes | Capability tokens cache availability in per-tier `AtomicU8` values; its source includes a summon-overhead instrument and exhaustive fallback-tier permutation support | Source-audited as an independent cache and fallback-test design; no dependency retained |
| [`simd-abstraction` 0.7.1](https://docs.rs/crate/simd-abstraction/0.7.1/source/) | Yes | `simd_dispatch!` resolves each operation once and stores its selected function in an `AtomicPtr` | Source-audited; a permanent indirect call is rejected without a measured Hermes deficit |
| [`simdeez` 3.0.1](https://docs.rs/crate/simdeez/3.0.1) | Yes | Scalar, SSE2, SSE4.1, AVX2, AVX-512, NEON, and WebAssembly SIMD behind one abstract operation trait | Source-audited for target and operation coverage; its exact `paste = 1.0.15` dependency fails advisory policy |
| [`simply_simd` 0.1.0](https://docs.rs/simply-simd/0.1.0/simply_simd/) | Yes | Macro-generated target dispatch with explicit guidance to enter once outside hot loops and preserve the target-feature frame | Source-audited; Hermes already shares that scope contract through `vectorize` and `#[runtime_dispatch]` |
| [`wide`](https://github.com/Lokathor/wide) | No | Fixed vector types selected by build-time target features; its documentation states runtime feature detection does not work | Excluded from a portable same-binary comparison |
| [`std::simd`](https://doc.rust-lang.org/beta/std/simd/) | Not on stable | Portable const-width vectors under the experimental `portable_simd` feature | Excluded by Hermes' stable-toolchain contract |

Pulp was selected for a bounded experiment because it is stable, selects the
host ISA at runtime, and can execute the same native-width f32/f64 operation
without changing the binary or the workload. The temporary dev dependency
enabled only `std`, `x86-v3`, and `x86-v4`, preserving Hermes' AVX2/AVX-512
width ladder while its unrelated relaxed-WASM feature stayed disabled. Pulp's
published manifest and current upstream main both require `paste = "1"`, which
resolves to unmaintained 1.0.15
([RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436)); Macerator
0.3.4 has the same dependency. `cargo deny` therefore rejects both closures.
Neither dependency nor comparator row is retained.

The rejected Pulp row was one generic f32/f64 `WithSimd` kernel. Hermes,
Fearless SIMD, and Pulp read the same six input allocations, wrote the same four
output allocations, asserted equal native width, and passed the existing
precision-specific scalar fused-rounding oracle before timing. The first exact
locked run reported:

| Type | Scalars | Hermes | `fearless_simd` | Pulp |
| --- | ---: | ---: | ---: | ---: |
| f32 | 256 | 31.649 ns [31.548, 31.755] | 31.958 ns [31.760, 32.243] | 32.121 ns [31.930, 32.451] |
| f32 | 1,024 | 160.50 ns [155.40, 169.25] | 158.04 ns [157.43, 158.94] | 156.85 ns [155.52, 158.39] |
| f32 | 4,096 | 2.0260 us [1.8581, 2.1751] | 2.8262 us [2.6766, 2.9625] | 2.2984 us [2.1051, 2.4984] |
| f64 | 256 | 87.831 ns [80.115, 95.090] | 76.937 ns [73.049, 82.778] | 73.986 ns [70.012, 80.406] |
| f64 | 1,024 | 781.87 ns [727.86, 821.24] | 851.89 ns [810.18, 873.68] | 876.80 ns [874.42, 878.99] |
| f64 | 4,096 | 3.3352 us [2.9980, 3.5414] | 3.6106 us [3.4801, 3.6783] | 2.7030 us [2.4894, 2.9587] |

An unchanged confirmation run reported:

| Type | Scalars | Hermes | `fearless_simd` | Pulp |
| --- | ---: | ---: | ---: | ---: |
| f32 | 256 | 32.050 ns [31.315, 33.372] | 31.357 ns [31.257, 31.464] | 31.668 ns [31.551, 31.790] |
| f32 | 1,024 | 180.71 ns [140.54, 261.53] | 126.36 ns [124.25, 130.24] | 123.87 ns [123.35, 124.70] |
| f32 | 4,096 | 1.8708 us [1.8649, 1.8770] | 1.8799 us [1.8761, 1.8852] | 1.8694 us [1.8672, 1.8726] |
| f64 | 256 | 69.377 ns [64.544, 77.541] | 87.823 ns [78.191, 96.915] | 61.728 ns [61.643, 61.859] |
| f64 | 1,024 | 907.31 ns [904.30, 910.90] | 902.40 ns [898.63, 904.86] | 814.09 ns [779.39, 844.76] |
| f64 | 4,096 | 2.7448 us [2.4990, 3.0895] | 2.7166 us [2.5421, 2.8935] | 3.2215 us [2.9976, 3.4218] |

No Pulp advantage has a disjoint interval in both runs. Several regimes reverse
ordering or change between overlapping and disjoint intervals, and absolute
large-size timing remains unstable on the shared host. Exact AVX2 assembly
explains the absence of a provider deficit: Hermes and Pulp each emit
six 256-bit loads, four stores, six fused arithmetic instructions, one loop
branch, and no calls, bounds branches, support probes, or panic paths for both
precisions. Their loop controls are equivalent `add/cmp/jb` and `add/dec/jne`
spellings. Pulp performs more minimum-length setup before its loop; that cold
work does not establish a Hermes deficit. The audit retains the measurements
but removes the Pulp row and dependency, and makes no production-kernel change.
Timing evidence is limited to AVX2 on this shared Windows Arrow Lake S host;
AVX-512 received compile and value-semantic coverage only in the removed
experiment, not retained regression coverage.

### Cached dispatch boundary — 2026-08-27 <a id="cached-dispatch-2026-08-27"></a>

Archmage's per-token atomic cache, simd-abstraction's resolved function pointer,
and Fearless SIMD 0.7's cached `Level` make Hermes' repeated standard-library
feature checks a falsifiable performance hypothesis. The retained
`dispatch_boundary` group at instrument revision `9489556a` compares four equal
`#[inline(never)]` entry frames for f32 and f64: a direct tuple-return control,
Hermes `vectorize`, `Level::new` plus Fearless dispatch, and caller-reused
`Level` plus Fearless dispatch. Every provider returns the black-boxed input
canary and its selected lane count; the pre-timing assertion requires Hermes and
Fearless to select the same native width.

The first valid unchanged run reported:

| Precision | Direct control | Hermes | Fearless `Level::new` | Fearless reused `Level` |
| --- | ---: | ---: | ---: | ---: |
| f32 | 0.7527 ns [0.7495, 0.7577] | 0.9815 ns [0.9702, 0.9933] | 0.9686 ns [0.9634, 0.9752] | 0.9787 ns [0.9678, 0.9895] |
| f64 | 0.7515 ns [0.7501, 0.7530] | 1.9766 ns [1.7523, 2.1688] | 1.8583 ns [1.6121, 2.0635] | 1.6619 ns [1.6468, 1.6830] |

The second valid unchanged run reported:

| Precision | Direct control | Hermes | Fearless `Level::new` | Fearless reused `Level` |
| --- | ---: | ---: | ---: | ---: |
| f32 | 0.7082 ns [0.7049, 0.7123] | 1.3613 ns [1.3398, 1.3861] | 1.5094 ns [1.5067, 1.5129] | 0.9970 ns [0.9800, 1.0207] |
| f64 | 0.7249 ns [0.7173, 0.7319] | 1.3854 ns [1.3441, 1.4287] | 1.7404 ns [1.7212, 1.7571] | 1.3053 ns [1.2051, 1.4127] |

Exact release assembly confirms that the instrument retains each boundary.
The direct control is one move and return. Hermes loads the standard detection
cache three times and tests AVX-512F, AVX2, and FMA before calling the selected
target-feature helper. `Level::new` tests the Fearless `LazyLock` state, loads
the cached level, and dispatches through a jump table; the caller-reused row
starts at that jump table. The first f32 run overlaps all provider intervals;
the first f64 run separates Hermes only from reused `Level`. The second run
separates Hermes from reused `Level` only for f32, while the f64 intervals
overlap. No Hermes disadvantage is therefore disjoint for both precisions in
both runs. The audit retains the regression instrument but rejects another
cache, atomic, or function-pointer indirection. Evidence is AVX2 timing and
x86-64 release code generation on one shared Windows Arrow Lake S host; hosted
AArch64 and AVX-512 runs provide compile/value coverage, not timing portability.

Findings:
- [minor] Consumer target-feature entry. ADR 009 records why a lane kernel must
  be monomorphized inside a `#[target_feature]` scope: without it the annotated
  backend operations cannot be inlined into the loop body, which that ADR
  states "completely neutralizing SIMD throughput advantages". Hermes solves
  this only for its own kernels — `#[runtime_dispatch]` is applied across
  `crates/hermes-simd/src/dispatch/`, and `hermes-simd` re-exports no path to
  it. `dispatch_view`/`dispatch_view_to` hand a consumer a typed view but no
  target-feature scope to use it in. Fearless SIMD's equivalent is a
  `Simd`-token method that takes the consumer's `#[inline(always)]` kernel and
  runs it inside the level's target-feature scope, so the consumer writes one
  generic body and the substrate owns the per-ISA entry.
- [minor] Safe operation surface behind a capability token. Every
  `BackendKernel<T>` facet method is `unsafe fn` — `SimdArith::{add, mul,
  fmadd, splat, ...}` in `crates/hermes-simd-core/src/kernel/roles/arithmetic.rs`
  and `SimdPermute::{reverse, interleave, deinterleave, fmaddsub, ...}` in
  `.../roles/permute.rs`. The obligation each carries is "the target feature is
  active", which is a property of the enclosing scope rather than of the
  arguments. Fearless SIMD discharges that obligation once, at token
  construction, and exposes the operations as safe methods on the token. ADR 011
  already made `BitBoardKernel` safe on the grounds that its stated obligation
  matched no implementation, and explicitly excluded `SimdKernel` as "genuinely
  `#[target_feature]`-gated". That exclusion was correct for the mechanism
  available when it was written; safe `#[target_feature]` functions (RFC 2396)
  are stable since Rust 1.86 and the Hermes toolchain floor is 1.95, so the
  obligation can now be carried by a token instead of by every caller. Revising
  ADR 011's exclusion is part of the increment, not a silent divergence from it.
- [minor] Measured consumer cost. Apollo at revision `424ce431` depends on
  `hermes-simd` from every transform crate, yet `apollo-fft` reaches it in one
  file for one function (`interleaved_complex_mul_assign`) and otherwise carries
  its own ISA fork: 28 files importing `core::arch`/`std::arch` — every such
  file in the Apollo workspace — 90 `#[target_feature]` attributes, 429 `unsafe`
  blocks, and 228 `unsafe fn` declarations. The one Apollo crate that did write
  a generic Hermes kernel, `apollo-fwht`, calls `Vector::<T, Arch>::load_unaligned`
  and the arithmetic facets inside its dyadic butterfly loop from a function
  with no `#[target_feature]` attribute anywhere in the crate — the ADR 009
  defect reproduced in a consumer. The substrate is not being declined on
  preference; it currently gives a consumer no way to reach the codegen its own
  ADR requires.
- [patch] Second independent corroboration of the sub-AVX2 x86 baseline gap.
  PhastFT ships SSE4.2 and WASM levels through Fearless SIMD. The Highway audit
  already recorded Hermes jumping Scalar to AVX2 on x86 (ADR 006 assessed SSE2
  feasibility). A second unrelated reference reaching below AVX2 raises that
  finding from one reference's choice to a portable-substrate norm.
- [patch] README positioning. The External SIMD Reference Baseline section named
  only the Highway audit, so a reader could not tell whether the safe-authorship
  model had been assessed.

Decisions:
- Do not adopt `fearless_simd` as a dependency. Hermes is the Atlas owner of CPU
  lane kernels and ISA dispatch; taking a third-party portable-SIMD substrate
  would re-fork the dimension Hermes exists to own and would not carry Hermes'
  sparse, packed, AMX, tensor, or COW surfaces. Use it as a design reference for
  the authorship model only.
- Note the reference's own negative result. Fearless SIMD's README records that
  its 0.1.1 CPU-capability *witness type* approach "couldn't quite be made to
  work" and that the current design carries the capability in a value instead.
  Hermes' `Avx2`/`Avx512`/`Neon` ZST markers are that same witness-type shape.
  The increment below therefore adds a value-carrying token beside the markers
  rather than reinterpreting the markers as one; the markers keep their role as
  the type-level backend selector for monomorphization.
- Keep the sealed `SimdKernel` seam and its `unsafe` primitives. The safe
  surface is a layer above them, not a replacement: the raw facets stay as the
  implementation seam that backends implement, exactly as ADR 011 left
  `KoggeStone`'s ISA fills unsafe beneath a safe trait.

Next increments:
- P1: consumer-facing target-feature entry and token-carried safe operation
  surface — filed as `HS-FEARLESS-TOKEN-2026-08-25` in `backlog.md`.
- P2: publish the consumer authorship path in the README and the book, so a
  downstream crate has one documented route from "I have a lane kernel" to
  per-ISA code.
- P3: re-assess the sub-AVX2 x86 baseline against ADR 006 now that a second
  reference corroborates it.

## ATLAS-ORPHAN-MODULES-096-HERMES closure — 2026-08-19

The stale orphan-module item is closed. `crates/hermes-simd-core/src/tensor/
mut_view.rs` was unreachable from every Cargo target root and had no textual
consumer; commit `1fe438c` deleted the 259-line duplicate. The isolated-tree
orphan detector reports `0`, and the merged provider matrix at code state
`f4d444b5` passes in hosted run `31819198076` (format, warning-denied Clippy,
configured Nextest, doctests, Rustdoc, Miri, cross-compilation, ARM NEON,
Intel SDE, benchmark budgets, and supply-chain checks).

The local all-workspace Nextest retry was not used as closure evidence: the
Atlas development overlay redirects the Hermes package to the primary
checkout while its peer-owned `Cargo.lock` is dirty, so Cargo stops before
compilation under `--locked`. No lockfile or peer-owned source was changed.

## HS-437 closure — default lane scratch frame measurement

The proposed `LaneBuffer` refactor is closed without a source change. The
measurement harness instantiated the default `BackendKernel::interleave` path
for Scalar f64 and emulated `SveArch` f64 on x86-64, then compiled the same
wrapper for AArch64 NEON f64. Release assembly showed no stack allocation in
any wrapper: x86-64 emitted register moves/instructions directly, while the
AArch64 NEON wrapper emitted `zip1`, `zip2`, and `stp` directly. The AArch64
compile emitted assembly before the Windows-host linker rejected the foreign
`--eh-frame-hdr` option, so this is code-generation evidence only and makes no
cross-target execution claim. The over-sized source arrays are therefore not
observable stack waste in the measured default path; `MAX_SIMD_LANES` remains
the compile-time overflow bound and no typed-buffer abstraction is added on
principle alone.

## HS-433 closure — structured AMX downgrade event

The release-only observability gap is closed. `AdaptiveDispatcher` emits one
subscriber-owned warning event when remote NUMA placement forces AMX down to
AVX-512; the event carries the node and both backend names plus the trigger
reason. The diagnostic no longer writes to stderr or disappears in release
builds. The same slice removes the no-std AMX global `Cell`/`Sync` substitute,
so a no-std session rejects safely instead of claiming thread-local state it
cannot provide. Evidence is the subscriber-backed value-semantic test,
default-feature compilation, and the no-default-features source check. The
merged provider head `f4d444b5` passes the exact hosted package matrix in run
`31819198076`, including the bounded benchmark job, ARM NEON, Intel SDE, Miri,
cross-compilation, and supply-chain checks.

## SIMD/SIMT Capability and Completeness Audit - 2026-08-11 <a id="simd-simt-2026-08-11"></a>

Scope: the `SimdKernel` operation catalog, the backend/`TargetId` matrix, and
the SIMT question. Method: full read of `kernel.rs`, `kernel_helpers.rs`, the
per-ISA impls, `target.rs`, `cpu.rs`, and the ADR set; source-audit tier except
where a delivered increment carries its own differential evidence.

### SIMT scope — closed, not a gap

Hermes owns lane-parallel CPU execution; device-resident SIMT execution belongs
to Hephaestus, consumed through Coeus/Apollo (README, "Atlas Compute
Boundaries"). Adding GPU dispatch here would fork a dimension another stack
member owns, so no SIMT work is opened against Hermes. What Hermes does own of
the SIMT *programming model* — per-lane predication, masked memory, indexed
gather/scatter, compress/expand, cross-lane reduction — is the axis audited
below, and the masked-memory half is now complete (HS-413..HS-421 landed the
tail routing; every hot kernel runs the provider-owned masked seam).

### Delivered

- [minor] Indexed-store asymmetry (delivered as HS-422): `gather` and
  `gather_masked` existed with no write-side dual, so the lane-addressing model
  was read-only and any scatter-shaped consumer had to leave the vector domain.
  `SimdKernel::scatter`/`scatter_masked` plus `SimdView::scatter` close it, with
  native AVX-512 `vscatterdps`/`vscatterdpd` and the lane-sequential default on
  AVX2/NEON (neither ISA has a scatter instruction). Evidence: per-backend
  differential property tests, a gather∘scatter round-trip identity, and
  duplicate-index/error-contract tests. AVX-512 native execution is runner-gated.

### Operation-catalog status (source-audit tier)

First, the fact that bounds the rest of this section: `SimdKernel` is
implemented for `F16`, `f32`, and `f64` only — there is no generic integer lane
backend. Integer SIMD lives outside this trait, in the AMX/VNNI tiling kernels
and the SWAR bitboards, each with its own operation set. Gaps must therefore be
assessed against a float-lane trait, not a general one.

- [minor] Resolved as HS-423. The rounding family (`floor`/`ceil`/`round`/
  `trunc`) is now defined through Eunomia's native-precision scalar seam and
  Hermes' `generic_unary_op`, with AVX2, AVX-512, and NEON overrides. The
  `RoundTiesEven` policy keeps halfway behavior explicit, and differential
  coverage includes negative values, ties, infinities, NaNs, and signed zero
  across Scalar, SveArch, and AVX2, with the other ISA paths compile-covered
  for hosted execution. The delivered implementation is recorded in Hermes
  commits `58c31a9` and `df32296`; no widen-compute-narrow fallback remains.
- [minor] Resolved as HS-424. General cross-lane permute (`reverse`,
  `interleave`, `deinterleave`) had no seam; the only lane shuffles were the
  complex adjacent-pair primitives. All three now exist as defaulted trait
  methods on the flat lane sequence, with AVX2 native `reverse`. The reusable
  finding is the flat-vs-sublane distinction: x86 `unpack` and `permute_ps`
  operate within 128-bit halves, so they are *not* implementations of a flat
  permute and cannot be substituted as overrides without extra cross-half
  shuffles — a trap that would have produced a silently wrong fast path.
  Remaining native overrides tracked as HS-427, deferred rather than written
  unverified: this host has no AVX-512 and is not aarch64, and wrong permute
  indices return plausible wrong lanes instead of failing loudly.
- Corrected on the same pass: an earlier draft of this entry recorded missing
  integer shifts and missing saturating arithmetic as gaps. Both are withdrawn.
  Shifts are not well-founded against a float-only lane trait, and float
  saturation is not the domain contract; `NumericElement` already carries
  `saturating_add`/`saturating_mul` for the scalar integer types that want them.
  A generic integer lane backend is an [arch] question deserving an ADR, not a
  silent operation-by-operation accretion — it is not opened here because no
  consumer requirement is recorded for one.

### Verification infrastructure — the finding that reframes the rest

- [minor] Resolved 2026-08-11. Every AVX-512 branch is guarded by
  `is_x86_feature_detected!`, so on a runner without the silicon those tests do
  not fail — they *skip*. The capability-report step added with the SDE job
  proved the GitHub x86 runner reports **no AVX-512 and no AMX flags at all**,
  so the AVX-512 scatter override, the BF16 tile dispatch, VNNI, and AMX had
  been carried by a green CI that never executed one of them. A passing suite
  was asserting less than it appeared to, and nothing in the logs said so.
  The `test-avx512-sde` job runs the whole suite under Intel SDE emulating
  Sapphire Rapids: 444/444 pass in 176s (about 11x native), with
  `test_masked_ops_avx512`, `test_select_ops_avx512`, `test_vector_ops_avx512`,
  `interleaved_complex_avx512_matches_scalar_backend`,
  `avx512_tiling::int8_tests`, and `test_adaptive_dispatcher_and_amx_session`
  all executing rather than skipping. It remains the deterministic semantic
  gate; HS-429's `test-avx512-hosted` job adds real-silicon coverage and
  timing on a best-effort basis when the hosted x86 runner carries AVX-512.
  Reusable finding: a capability-gated test suite reports coverage it does not
  have. Automated backend selection is the right design, but it must be paired
  with automated *identification* — the suite now enumerates `TargetId::ALL`,
  prints which backends the runner actually executed, and asserts a per-runner
  expectation supplied as configuration. Silence must never read as coverage.
  Emulation is the fallback, not the mechanism: real silicon is used wherever
  it can be requested (aarch64 NEON on `ubuntu-24.04-arm`, AVX2 on x86, and
  AVX-512 on hosted x86 whenever that silicon is present); SDE covers the ISAs
  no pinned hosted runner guarantees — AVX-512, which is heterogeneous across
  the hosted x86 pool, and AMX, which is unavailable there entirely. SDE's
  limit is that it validates semantics and never performance, so a benchmark
  claim on those ISAs still needs real hardware (HS-429).
- Consequence for the HS-422 and HS-424 records: both shipped their AVX-512
  work marked "runner-gated, not executed". That caveat is now discharged for
  scatter — `prop_scatter_matches_reference_all_backends`,
  `prop_gather_scatter_roundtrip_all_backends`, and the duplicate-index and
  error-contract tests all exercise the native `vscatterdps`/`vscatterdpd`
  override under SDE.

### Hand-written intrinsics versus LLVM's lowering (HS-427, 2026-08-12)

- [minor] A native AVX2 `interleave` written as `unpack` + `permute2f128` — the
  textbook flat-interleave sequence — measured **37% slower** than the generic
  store/permute/load default it was meant to replace (two runs, p < 0.05, quiet
  host, L1-resident size). `deinterleave` was neutral-to-negative. Both were
  removed. AVX2 `reverse` survived on measurement (10.4% faster at 1024 f32).
  Reusable finding: LLVM already lowers the generic default's stack round-trip
  into good shuffle sequences, so "replace the portable path with intrinsics"
  is a hypothesis to measure, not a foregone win — the vectorization ladder's
  rule to escalate only on a measured shortfall applies to *lane permutes* as
  much as to arithmetic. Corollary for benchmark design: measure at a
  cache-resident size. The same comparison at 16384 elements shows no
  difference, because the working set spills and the permute cost vanishes into
  memory traffic — a size that would have hidden the regression entirely.
- Consequence: the AVX-512 and NEON permute overrides shipped in the same
  increment carry correctness evidence only. They are canonical single-
  instruction lowerings, but after the AVX2 result they are explicitly *not* a
  speed claim until measured. HS-430 now supplies the NEON measurement path:
  the native aarch64 job saves the existing `permute` Criterion rows, rebuilds
  with only the three NEON overrides disabled, and compares the identical rows
  on the same ARM host under a finite 300-second command. The hosted result
  still requires review before retaining or deleting an override. AVX-512
  timing is assigned to HS-429's `test-avx512-hosted` job (SDE is semantic
  evidence, not timing evidence).

### Square-transpose networks (HS-TRANSPOSE-NETWORKS, 2026-08-27)

- Retained: AVX2 f32's 8x8 register network reduces 408–421 ns to 4.21–4.32 ns
  across two unchanged Core Ultra 9 285K comparisons. NEON f32's 4x4 network
  repeats at 3.536–3.544 ns versus 7.281–7.324 ns for the generic default in
  hosted AArch64 runs `33137876655` and `33138579478`. Exact release assembly
  contains one tile-length branch followed by register shuffles with no scalar
  lane traffic.
- Removed: the incumbent NEON f64 `trn1`/`trn2` override measures
  2.265–2.268 ns versus 2.080–2.086 ns for LLVM's generic lowering. The generic
  route repeats at 2.075–2.083 ns after removal. Provisional AVX-512 f32/f64
  networks passed the index-coded oracle under Sapphire Rapids emulation and
  produced spill-free exact assembly, but no controlled real-silicon timing
  was available, so they were deleted rather than retained as unverified
  optimizations. The explicit AVX-512 benchmark rows remain the re-open
  instrument when suitable hardware is available.

### Backend matrix

- [major] Resolved as HS-425. `TargetId::Sve` now routes through both forced
  dispatch helpers to the lane-emulated `SveArch` backend, and conformance
  tests cover the public target and host-capability surfaces. `TargetId` is
  `#[non_exhaustive]`; automatic dispatch intentionally has no SVE branch, so
  the emulated backend remains explicitly requested. The breaking migration is
  recorded by ADR 014 and the changelog entry delivered with Hermes PR #49
  (merge `fb36e0f`, implementation `dd4cc78`).
- Native SVE remains blocked on stable Rust (scalable vectors are not
  expressible); `SveArch` stays lane-emulated and its hardware probe stays
  informational. Unchanged, correctly documented, not a defect.
- SSE2 and Arm SME each have a feasibility ADR (006, 007) and no backend. That
  is a recorded decision, not drift.

### Documentation hygiene

- [patch] Resolved as HS-426. `docs/adr/` had two ADRs numbered 007, eight of
  eleven with no `## Status` section (the generated index rendered `—`), and
  `Approved` instead of the canonical `Accepted` on the rest. The later
  duplicate renumbered to 011 with its references updated; all eleven now carry
  `Accepted`, and `adr-index.py check` passes for this repository. Note the
  generator already reported every one of these anomalies — the tooling was
  correct and unheeded, which is the reusable finding: a check whose output
  nobody burns down is not a gate.

### Cross-repo (reported, not owned here)

- The atlas stack overlay in `.cargo/config.toml` had gone stale against gaia's
  rename to package `gaia-mesh`, breaking dependency resolution under the
  umbrella. Regenerated via `scripts/atlas-stack-overlay.py generate` (the
  overlay is generated state, never hand-edited). Consumers still declaring
  `gaia = { git = ... }` remain unpatched until they rename the dependency —
  an atlas-level follow-up, outside Hermes.

- Resolved 2026-08-06 — HS-409 fused ternary AXPY provider facade: Hermes now
  exposes `axpy_mul(alpha, a, b, out)` and `SimdOps::axpy_mul` for the exact
  in-place contract `out[i] += alpha * a[i] * b[i]`. The runtime-dispatched
  kernel reuses `SimdKernel::mul` followed by `SimdKernel::fmadd`, writes each
  output lane once, and performs no temporary allocation. Length validation is
  centralized in the provider kernel; the scalar tail uses the same
  `Scalar::scalar_fmadd` operation as the vector path. Public f32/f64 tests cover
  the facade and tail-sized inputs; internal tests cover empty/tail/mismatch
  cases. The operation is provider capability only: Kwavers adoption remains
  downstream work and is not claimed here. Evidence tier: source implementation
  plus value-semantic public-facade tests. Local locked gates remain blocked by
  the pre-existing dirty Hermes lock overlay, which requires unrelated provider
  lock regeneration; no lockfile rewrite was retained.

- Resolved 2026-07-19 — HS-402 provider compatibility: Hermes' native
  `eunomia::F16`/`Bf16` source compiles against Eunomia 0.6 without restoring
  the retired foreign raw-half trait implementations. Cargo resolves the
  workspace to one Eunomia 0.6 identity at `df77dfd`; warning-denied Clippy,
  388 value-semantic Nextest cases, 18 runnable doctests, and warning-denied
  rustdoc pass.

- Resolved 2026-07-18 — HS-401 Eunomia reduced-precision ownership: Hermes
  replaces raw `half::f16`/`half::bf16` in scalar, F16C, AVX-512, NEON, AMX,
  tiled GEMM, tests, and benchmarks with `eunomia::F16`/`Bf16`/`F32`.
  Duplicate raw-half AMX and tiled-GEMM families are deleted, and all direct
  `half` manifest dependencies are removed. Source and manifest residue scans
  are empty; the locked graph contains one Eunomia 0.5.0 identity. Full
  all-feature warning-denied Clippy, 388 value-semantic Nextest cases,
  doctests, rustdoc, and no-default-feature compilation pass. Evidence tier:
  compile-time provider identity plus value-semantic and differential tests.
  The lock still contains transitive `half` through Eunomia's temporary
  raw-trait surface and Criterion's Ciborium dependency.
- Resolved 2026-07-18 — HS-401 remote host variance: PR #8 exposed two
  host-sensitive defects absent on the local CPU. Adaptive dispatch queried
  Bf16 capabilities for every operand type, allowing int8 GEMM to enter
  AVX-512 VNNI on a host without that extension; dispatch now binds its probes
  to `T: AmxSupport + Avx512Support`. The 64-lane AVX-512 masked-gather oracle
  also used a fixed 100-element fixture although its maximum index is
  `3 * (lanes - 1)`; fixture length is now derived as `3 * lanes`. Evidence
  tier: remote ISA failure reproduction plus type-bound dispatch and an
  analytical index bound. PR #8's final x86, AArch64 cross-compile, native
  AArch64 NEON, Miri, cargo-deny, and CodeRabbit gates pass at `f9e8ff5`;
  merge commit `8970ffc` closes the item.
- Residual 2026-07-18 — HS-401 historical semver baseline: `origin/main`
  resolves moving Eunomia main 0.5.0, so its historical raw-half implementation
  no longer compiles at `scalar/tiling.rs:76-77` and
  `x86_64/avx512_tiling.rs:85,91`. `cargo semver-checks` therefore cannot
  classify the current public delta against that baseline. This is baseline
  dependency drift; the current 0.4.0 workspace compiles and passes its gates.
  Re-open when the semver baseline can pin its historical Eunomia revision.
- Resolved 2026-07-18 — HS-401 AArch64 verification: rustup's 1.95 target
  libraries are incompatible with proc-macro artifacts already produced by the
  PATH MSYS Rust 1.95 Rev2 compiler in the mandatory shared target directory.
  No private target or destructive clean was used. PR #8 independently passes
  both the remote AArch64 cross-compile and native runtime NEON lanes. Evidence
  tier: cross-target compile-time validation plus native architecture tests.

- Resolved 2026-07-15 — provider default-branch convergence: Hermes removes
  revision pins and workspace-local patches for Mnemosyne, Eunomia, and Themis.
  `cargo tree --locked -d -p hermes-simd` reports one identity for each; the
  package-scoped format, Clippy, nextest, rustdoc, and `cargo deny check` gates
  pass. CI's source allowlist names the reviewed provider URLs directly and
  redundant sibling checkouts are deleted. Evidence tier: locked
  dependency-resolution and value-semantic package tests.

## 2026-07-08 CI: miri gate known-failing, tracked on upstream mnemosyne <a id="miri-known-failing-2026-07-08"></a>

**Evidence tier: machine-checked (Miri, both Stacked Borrows and Tree Borrows
aliasing models).** PR #5 (`cb0b1b0`) merged with the `miri` CI job left
red — this is a deliberate, tracked exception, not an overlooked failure.

- **Finding:** `cargo miri test -p hermes-simd-core` fails on a genuine
  aliasing violation inside `mnemosyne-local`'s allocator, not in
  hermes-owned code. Repro: `AlignedVec::with_capacity` (`vec/mod.rs:96`)
  calls `mnemosyne_local::alloc::thread_alloc_checked` (`alloc.rs:130`),
  which reads a `Page` via `NonNull::as_mut`; a later `dealloc()`
  (`vec/mod.rs:542`) writes through an aliasing pointer to the same backing
  memory, disabling the earlier `Page`-pointer's tag; a subsequent `alloc()`
  re-reads that now-disabled tag.
- **Ruled out as a Stacked-Borrows-specific false positive**: tested under
  `-Zmiri-tree-borrows` (a materially more permissive aliasing model
  designed to accept exactly this embedded-metadata-allocator pattern) — it
  still failed, with a clearer diagnostic pointing at the same `Page`
  pointer/foreign-write sequence. Agreement across both independent models
  is strong evidence of a real bug, not a model artifact. The Tree Borrows
  CI experiment was reverted (`166a7b9`, since it provided no benefit) after
  confirming this.
- **Owning repo:** mnemosyne, not hermes. Full reproduction instructions and
  working hypothesis documented in mnemosyne's own `gap_audit.md`
  ("2026-07-08 Miri: real aliasing violation in the alloc/free
  page-metadata path", commit `98a02b6`).
- **Why merge anyway:** the other 4 CI jobs (gates, cargo-deny,
  cross-compile, test-aarch64) all pass; this is a pre-existing bug in an
  upstream dependency, newly surfaced by CI graph resolution rather than
  introduced by this PR's changes; blocking hermes indefinitely on an
  upstream fix has no bounded timeline.
- **Follow-up (tracked, not deferred silently):** once mnemosyne's allocator
  fix lands and hermes bumps its pinned `mnemosyne` rev, re-run
  `cargo miri test -p hermes-simd-core` and confirm green before removing
  this entry. Until then the `miri` job on hermes CI is a **known-failing,
  tracked gate** — branch protection was overridden via `--admin` for this
  merge only; it is not disabled going forward, so future PRs will need the
  same override (or the upstream fix) to land.

## Comprehensive Audit - 2026-07-02 (round 8, 5-agent sweep) <a id="audit-2026-07-02-r8"></a>

Five parallel read-only audits (performance, memory/zero-copy, unsafe soundness,
architecture/redundancy, tests/benches/docs) plus two deep soundness sub-audits.
Evidence tier per item; source-audit unless a differential/property test is named.
Findings are the register below; fixes land as tracked backlog items in triage
order (correctness → architecture → tests → docs → PM).

### Correctness / soundness (HARD)

- **[RESOLVED 2026-07-02] SELL-p vectorized SpMV OOB read (PROVEN, 3 agents).**
  `sparse/spmv.rs` `sellp_spmv_vectorized` gathered `x[col_idx]` and loaded
  `values[offset..]` full-width with no bounds check, reachable from the safe
  `SparseView::<SellP<C>>::spmv` when `LANE_COUNT == C`; the sibling CSR/BCOO
  paths self-defend, SELL-p did not. `SellPMatrix` `pub` fields + no-op `new` +
  opt-in `validate()` let a caller drive both reads out of bounds. Same over-read
  in `sparse/ops.rs` `elementwise_mul_dense`. Fixed by routing both vectorized
  paths through the SSOT `SparseValidate::validate()` via
  `spmv::assert_sellp_validated` before the unsafe kernel; two `#[should_panic]`
  regressions on the Scalar-backed vectorized path (`LANE_COUNT 4 == C`). See
  [Resolved](#resolved).
- **[MITIGATED 2026-07-02] AMX dispatched on CPUID-only; OS tile-data permission
  never requested (PROVEN).** `cpu.rs` `has_amx` checked only CPUID leaf 7; no
  `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` (Linux) anywhere, so the
  first `tileloadd`/`tdpbf16ps` from `tile_matmul::gemm` `#NM`-faulted on capable
  Sapphire-Rapids Linux hosts; `__cpuid_count(7,_)` also lacked a max-leaf guard
  (leaf-1 aliasing). Interim: the AMX probes now return `false` (AMX dispatch
  disabled), preserving the safe-dispatch contract — the fault can no longer
  occur. **Still open:** restoring AMX behind a permission-aware probe (hardware +
  XCR0 TILECFG/TILEDATA + a one-time Linux XTILEDATA `arch_prctl`). Blocked on: the
  stable toolchain rejects the AMX strings in `is_x86_feature_detected!`
  (`x86_amx_intrinsics` unstable), and verifying the raw `arch_prctl` syscall
  needs an AMX-capable Linux host. `[minor]` DoR: acceptance = AMX GEMM dispatches
  and matches the scalar reference on a Sapphire-Rapids Linux runner.
- **[RESOLVED 2026-07-02] AVX-512 sub-feature gating gaps (PROVEN/SUSPECTED).**
  `cpu.rs` `Avx512Support` read raw CPUID (AVX512_BF16 / VNNI) without OSXSAVE/XCR0
  and, for bf16, probed the unrelated `avx512bf16` bit while the tile kernel
  enables `avx512f,avx512bw,avx512vl` and never uses `dpbf16`. `unpack.rs`
  `widen_i8_to_i16` ran AVX-512**BW** `_mm512_cvtepi8_epi16` under an
  AVX-512**F**-only guard → SIGILL on KNL. Fixed: the two AVX-512 tile probes now
  use `is_x86_feature_detected!` for the exact enabled set (macro handles
  XCR0/max-leaf); `widen_i8_to_i16` gated on `avx512bw`. See [Resolved](#resolved).
- **[RESOLVED 2026-07-05] Unsound safe API surfaces (PROVEN).**
  `AmxSession::new` / `AmxBatchSession::begin` now return
  `AmxSessionError::UnsupportedTarget` before `ldtilecfg`; `release` does not
  issue `tilerelease` unless a supported active session exists. `SimdArch`
  carries the runtime-support probe used by `TargetId` and the safe `Vector` /
  `Mask` wrappers; unsupported AVX-512 hosts get `SimdError::UnsupportedTarget`
  from fallible constructors and checked slice wrappers before any AVX-512
  instruction. Infallible vector conveniences panic before ISA execution.
  Evidence tier: type-level trait seam + value-semantic unsupported-host
  regressions. See [Resolved](#resolved).

### Performance (source-audit tier; each needs a criterion baseline before/after)

- **[REVISED 2026-07-02, measured] SELL-P/BCOO chunk-width dispatch — the
  simple fix is a measured regression; only the AVX-512 case remains open.**
  A chunk-aware ladder routing to the widest ISA whose `LANE_COUNT == C` was
  implemented and A/B-benchmarked on this AVX2 host: for `sellp4` (C=4,
  100k rows, 10% density) the *old* widest-first path ran 7.48 ms
  (13.7 Gelem/s) vs 17.6 ms (5.8 Gelem/s) for lane-matched routing to the
  4-lane scalar-marker kernel — **2.4× slower**, because the "scalar fallback"
  loop executes inside the AVX2 `#[target_feature]` dispatch helper and LLVM
  auto-vectorizes it at full 8-lane width, beating the narrow emulated-gather
  kernel. The change was reverted; a dispatcher-independent SELL-8 multislice
  differential test was kept. Still open, hardware-gated: on an AVX-512 host a
  SELL-8 f32 matrix runs the auto-vectorized fallback where the *native* AVX2
  8-lane gather kernel in the same binary might win — unmeasurable without
  AVX-512. DoR: acceptance = criterion A/B of sellp8 widest-first vs
  AVX2-routed on an AVX-512 runner; do not re-implement without that number.
  The `C = k·LANE_COUNT` kernel generalization remains a separate `[minor]`
  candidate under the same measurement gate.
- **[RESOLVED 2026-07-05] Per-call CSR/BCOO/SELL-p SpMV re-validation
  (HIGH, 2 agents).** Added `Validated<F>`/`ValidatedData<S>` and moved CSR,
  SELL-p, and Blocked-COO `SparseSpMv` impls to
  `SparseView<Validated<_>>`. Public `spmv_csr`/`spmv_bcoo`/`spmv_sellp`
  dispatch now requires validated storage, and validated `SparseCow`
  constructors preserve the run-once invariant for repeated solver calls. Raw
  malformed structures fail at validated view/COW/public-dispatch construction;
  hot SpMV kernels keep only runtime-vector size checks for `x`/`y`. Evidence:
  type-level typestate plus regression/property coverage over construction-time
  rejection and value-semantic SpMV for CSR/SELL-p/Blocked-COO. Local compile
  verification is pending shared Cargo target lock clearance.
- **[RESOLVED 2026-07-03] GEMM/GEMV scalar column tails (HIGH).** **GEMM:**
  `tiled_gemm`'s `n % block_n` trailing columns run `leading_k_mask`-guarded
  fmadd lane groups; measured 3.43× at n=63 (25.17 → 7.33 µs), bitwise
  differential m=7/n=45/k=13, Theorem 1 updated. **GEMV:** the `ncols % lane`
  tail (both blocked and remainder paths) folds into the vector accumulator via
  one masked fmadd (x-tail loaded once, reused across rows). Added the
  workspace's first GEMV bench (`gemv_bench.rs`, tail-isolating); measured at
  cache-resident 256×256 the tail row improved 3.58 → 2.83 µs (+27% throughput),
  aligned neutral within noise, DRAM rows bandwidth-bound/neutral. f32 facade
  differential (n=21, nrows=11, dyadic-exact ⇒ bitwise) + existing f64 tail-shape
  suite. Follow-on candidate: same masked-tail treatment for `gemv_transpose`
  and `axpy` (both still scalar-tailed, both now benchmarkable).
- **Resolved 2026-08-09 — HS-417 transposed GEMV column tail.**
  `gemv_transpose_strided_impl` now handles `ncols % LANE_COUNT` through one
  initialized local lane-buffer path and `SimdKernel::masked_fmadd`, preserving
  the full-width-valid masked-memory contract even on blend-based backends.
  Only the live tail is copied back to `y`; no caller slice is over-read or
  over-written. The f32 facade has a non-dyadic tolerance regression because
  the provider-owned fused operation may round differently from scalar
  multiply-plus-add. Evidence tier: provider implementation plus differential
  shape tests and non-dyadic f32 tolerance coverage. Native SVE remains a
  separate stable-toolchain-blocked item.
- **Resolved 2026-08-09 — HS-421 native AVX-512 BF16 tile kernel.** The
  `Bf16 × Bf16 → F32` tile now has a native `DPBF16PS` path on hosts reporting
  `avx512bf16`, with a single lower-level capability SSOT in
  `hermes-simd-intrinsics`. The existing AVX-512F/BW/VL conversion/FMA tile
  remains the fallback for AVX-512 hosts without BF16 dot products. The native
  differential uses non-dyadic BF16 inputs and nonzero `C`, validating the
  `C += A·B` contract; ordinary hosts skip only the hardware-specific execution
  while still compiling and testing the fallback. A hosted AVX-512 BF16 runtime
  and benchmark gate remains open.
- **[open] Scalar tails on every hot kernel (MED-HIGH).** `view/reduce.rs`,
  `view/ops.rs`, `dispatch/axpy.rs`, etc. end in element-at-a-time loops although
  `leading_k_mask` exists on every backend; up to 15 scalar iters on AVX-512 f32
  tails, dominating short/odd-length vectors. `[minor]`. **Partial closure
  2026-08-07 (HS-410/HS-411/HS-412):** `dispatch/axpy.rs` now routes its final
  partial vector through `masked_fmadd`, `dispatch/scale.rs` routes its final
  partial vector through `masked_mul`, and `dispatch/axpy.rs`'s `axpy_mul` path routes its
  final partial vector through register scaling plus `masked_fmadd`; all use
  initialized local lane buffers that preserve safety for AVX2 blend-based
  masked operations.  **HS-415 (2026-08-07):** `reduce_popcount` and the shared binary
  `reduce_popcount_op` now route their final partial vectors through
  `masked_sum_reduce` after copying source lanes into initialized local buffers;
  each masked tail count is exact while the existing whole-reduction accumulator
  contract remains unchanged. Generic reduction and broader view tails are
  covered by HS-416; other kernels remain open. **HS-414 (2026-08-07):** `AbsSum` and
  `AbsMax` now route their final
  partial vector through a generic masked reduction seam after copying live
  elements into initialized local lanes and applying the transform before
  identity merge. Generic reduction and broader view tails are covered by
  HS-416; other kernels remain open. **HS-413 (2026-08-07):**
  `axpy_rows` now routes its final partial vector
  through the same initialized-buffer `masked_fmadd` helper, and `axpy_rows_batch`
  does so per row after preserving its depth accumulation order. Non-dyadic f32
  regressions cover both paths. Reductions, view operations, and other kernels
  remain open; no repo-wide closure or performance claim is made.
- **Resolved 2026-08-09 — HS-418 dense dot-product tail.**
- **Resolved 2026-08-09 — HS-419 pairwise reduction tail.** `SimdView::zip_reduce(Dot)`
  now copies live pairwise inputs into initialized provider-local buffers and
  uses the generic masked reduction seam for its final partial vector. This
  removes the dot cleanup loop without violating the full-width masked-load
  contract; forced emulated-SVE non-dyadic f32 coverage allows the documented
  reassociation tolerance.  `Product` remains on its scalar pairwise tail path
  because its multiplicative identity and reduction ordering are distinct.
- **Resolved 2026-08-09 — HS-420 mutable generic view tail.**
  `SimdView::transform_in_place` now stages its final partial operands in
  initialized provider-local buffers, applies the generic `ElementOp` vector
  seam, and copies back only live result lanes. Add/Sub/Mul/Div therefore share
  one bounds-safe implementation; forced emulated-SVE odd-length coverage pins
  the tail path. Remaining scalar tails are tracked per kernel rather than
  claimed closed globally.
  `SimdView::dot` now copies its short remainder into initialized local lane

  buffers and folds it through `SimdKernel::masked_fmadd`, avoiding caller-slice
  over-read while retaining the provider's fused arithmetic path. The final
  reduction includes only live lanes. Odd non-dyadic f32 coverage uses the
  documented tolerance for fused-rounding differences. Remaining scalar tails
  are tracked per kernel rather than claimed closed globally.

- **[REJECTED 2026-07-03, measured] No K/M cache blocking in GEMM.** Hypothesis:
  the full `k × block_n` B panel spilling L1d degrades large-`k` GEMM, and a
  BLIS KC loop bounding the panel to L1d would recover it. **Falsified by
  measurement.** For AVX2 f32 (`TilingPolicy<3,3>`, block_n = 24) the panel is
  `k·96` bytes — 24 KiB at k=256, 48 KiB at k=512, 72 KiB at k=768, 96 KiB at
  k=1024 (2× this CPU's 48 KiB L1d). Measured square-GEMM throughput (criterion):
  256³ = 78.4, **512³ = 69.8**, 768³ = 79.9, 1024³ = **85.6 GFLOP/s** — flat to
  *rising* with `k`, with the largest (most-spilled) panel the fastest. The 512³
  dip is a power-of-two cache-set-conflict artifact (768³, non-power-of-two,
  recovers to 79.9), not L1 spill. The current full-panel pack + L2-residency
  design is correct for this microarchitecture (large fast L2 holds the panel,
  and packing amortizes better over more row blocks as `m` grows); KC-blocking
  would add `⌈k/KC⌉` passes of C load/store to fix a non-problem. Bench rows
  256/512/768 retained as the scaling-regression gate. Not re-opened without a
  microarchitecture whose L2 cannot hold the panel (measured, not assumed).
- **[open] No software prefetch in gather-bound SpMV (MED); no streaming/NT stores
  for out-of-LLC writes (MED); `Aligned` typestate dead at the dispatch facade —
  every op uses unaligned loads and NT stores are blocked on it (LOW-MED);
  uniform `UNROLL_FACTOR=4` under-fills the FMA pipeline for cache-resident
  reductions (LOW-MED).** The per-call detection-cache hypothesis is resolved:
  the retained f32/f64 dispatch-boundary instrument exposes the extra feature
  loads but rejects another cache because no Hermes deficit repeats across two
  unchanged runs. Remaining entries are `[patch]`/`[minor]` and
  measurement-gated.
- **[RESOLVED 2026-07-02] Integer/half emulated-kernel throughput — measured,
  split verdict (host-capability sweep, criterion).** (a) *Integer dense ops*:
  LLVM fully auto-vectorizes the emulated `[i32; 8]` kernels inside the
  `#[target_feature]` wrappers — `sum::<i32>` ~12× scalar (50–62 Gelem/s),
  `dot::<i32>` ~7.4× (bandwidth-bound); hand-written AVX2 integer kernels
  REJECTED as no-win duplication; i32 bench rows are the regression gate.
  (b) *int8 GEMM*: new 256-bit **AVX-VNNI** tile backend (`vpdpbusd` + exact
  +128 bias correction) — 17.3–20.2× measured over scalar tiles; dispatch
  ladder now AMX → AVX-512 VNNI → AVX-VNNI → scalar. (c) *f16*: AVX2 kernel's
  arithmetic core upgraded to **F16C** hardware conversion (bitwise-identical
  to the software semantics) — `dot::<f16>` 221 Melem/s → 7.22 Gelem/s
  (31.7×). (d) *bf16*: ~2 Gelem/s emulated (shift conversion partially
  auto-vectorizes); hardware core deferred until a consumer needs it —
  remaining emulated gap is gather/compress/mask ops (scalar loops), deferred
  until a sparse-integer consumer exists. See CHANGELOG [Unreleased].

### Memory / zero-copy

- **[RESOLVED 2026-08-27] `DenseWithMask` `[bool]` mask bit-packed
  (was MED-HIGH).** New arbitrary-length `PackedMask` (canonical `mask`
  module) replaces the byte-per-element `[bool]` in `DenseWithMaskData` /
  `OwnedDenseWithMask` in place — 8× mask footprint reduction, packed once at
  the construction boundary — and the SpMV / `sum_values` /
  `elementwise_mul_dense` kernels read packed lane windows via
  `mask_from_bitmask` instead of the per-chunk per-call bool-conversion loop.
  Unused `DenseWithMaskBitMaskData` partial variant deleted. PR #81,
  backlog `HS-DENSEMASK-BITPACK-2026-08-27`.
- **[RESOLVED 2026-08-27] Packed-mask and dense logical-shape bounds.** Public
  `PackedMask` extraction now rejects every out-of-range bit or window in debug
  and release builds without offset arithmetic overflow. `DenseWithMask`
  validation requires exact values and mask lengths against checked
  `nrows * ncols`; its accessors and kernels validate at their operation
  boundaries, while vector loops use crate-private prevalidated extraction. Release
  adversarial tests cover the boundary and overflow cases. Exact AVX2 inspection
  shows only pre-loop validation added, and an unchanged bounded comparison
  detects no stable throughput change (`HS-PACKED-MASK-SHAPE-SAFETY-2026-08-27`).
- **[RESOLVED 2026-08-27] `cmp_*_mask` stack round-trip.** The six public
  comparison-mask methods now convert the backend comparison register directly
  to its native mask. Two unchanged f32/f64 measurements and exact AVX2
  disassembly identified and then removed the store, lane-wise scalar scan, and
  mask reconstruction (`HS-NATIVE-COMPARISON-MASK-2026-08-27`).
- **[RESOLVED 2026-08-27] AVX2 `f32` to `i32` cast scalarization.** Two unchanged
  measurements and exact disassembly found eight scalar conversions per public
  vector versus one packed conversion in Fearless SIMD. Hermes now emits the
  same packed precise-conversion sequence, including Rust-compatible positive
  overflow, NaN, and infinity correction; boundary and arbitrary-bit tests
  match scalar `as`. The corrected whole-output checksum instrument reports
  6.9–11.4x finite in-range public-path gains across 256–4096 elements; the
  provider-to-provider rows remain host-load-sensitive, so no parity claim is
  attached to this result. Provider `18da238` merged through PR #86 as
  `5734b85`; PR #87 fixed target-test imports as `4f6a1eb`, and hosted run
  `33120584552` is green. AVX2 intrinsics cannot execute under Miri; targeted
  host property tests and exact code generation cover this path. Local ASan
  remained unavailable because the MSVC toolchain lacks
  `clang_rt.asan_dynamic_runtime_thunk-x86_64.lib`. Lane-count-changing
  conversion remains outside the current `Vector::cast` contract.
- **[RESOLVED 2026-08-27] AArch64 all-target warning escape.** PR #86's local
  AArch64 evidence compiled library targets only, while hosted CI compiled test
  targets and rejected three host-conditional imports under `-D warnings`. The
  imports are now scoped to x86 or removed, and both AArch64 Linux and Windows
  all-target checks reproduce the hosted warning policy. The prevention is to
  run the hosted all-target command, not a library-only approximation, whenever
  target-specific test code changes.
- **[RESOLVED 2026-08-27] `argmin`/`argmax` per-vector single pass rejected.**
  A same-binary candidate preserved empty/NaN rejection, first-occurrence ties,
  signed-zero representation, generic f32/f64 execution, and zero allocation.
  Two unchanged runs at 256/1024/4096/16384 elements lost every row: f32 was
  2.07–4.00× slower and f64 was 1.23–1.58× slower. Exact AVX2 assembly shows
  the cause: each vector executes a serial horizontal-minimum shuffle chain
  before scalar comparison and mask extraction. The current vector reduction
  plus locating/NaN scan remains unchanged. A lane-local value/index
  accumulation design requires new backend index-vector arithmetic, selection,
  and extraction; it is not justified by this rejected experiment.
- **[RESOLVED 2026-07-04] `compress` per-chunk buffer zero-init.** The hot
  compaction loop re-declared `[T::ZERO; 64]` each chunk (256–512 B of zero
  stores) though the vector store writes `lane_count` lanes and the copy reads
  only `pop ≤ lane_count`. Now a single hoisted `MaybeUninit<T>` array
  (`MAX_SIMD_LANES`, `LANE_BOUND_CHECK`-guarded) with the loop-invariant popcount
  hoisted; behavior unchanged (compress + `expand∘compress` identity tests pass).
  Focused Criterion coverage now records public `SimdView::compress` scalar and
  host-AVX2 all/half/quarter-mask rows at 1K, 16K, and 256K elements; regression
  self-check covered 102 committed Hermes benchmark rows.
- **[RESOLVED 2026-07-03] Non-temporal (streaming) stores for out-of-LLC writes
  — strongly beneficial, productionized.** Focused experiment
  (`streaming_bench.rs`): `out = a + b` over 16 Mi f32 (192 MiB working set, past
  L3), identical AVX2 loads+add, differing only in the store — normal
  `_mm256_store_ps` vs `_mm256_stream_ps` + `sfence`. **Regular 10.24 ms
  (18.3 GiB/s) → streaming 5.98 ms (31.3 GiB/s) = 1.71×**, far above the ~25%
  RFO-avoidance estimate. Productionizing: a `SimdKernel::store_streaming` seam
  (default = `store_aligned`; `SUPPORTS_NT_STORE` const gate; x86 f32/f64
  override via the codegen template's `__PREFIX___stream___SUFFIX__`) plus
  `stream_write_barrier` (sfence), and a size-gated (`len·sizeof(T) ≥ LLC-ish
  threshold), prefix-peeled-to-alignment streaming path in the elementwise
  `zip_into` SSOT. Differential test: streaming result is byte-identical to the
  regular store (same op, cache bypass only).
- **[RESOLVED 2026-07-03] `reduce_popcount_{and,or,xor}` triplication.** Three
  byte-identical ~104-line popcount reductions differing only in the bitwise op
  collapsed to one generic `reduce_popcount_op<Op: ElementOp<T>>` + three ZST
  wrappers (`BitAnd`/`BitOr`/`BitXor`); −153 lines, zero-cost (op monomorphized).
- **[RESOLVED 2026-07-03] `AlignedVec` growth churn.** Added `reserve` +
  `extend_from_slice` (single realloc via the shared `grow_to` SSOT); `Extend for
  SimdCow` now reserves `size_hint().0` up front instead of a push loop's
  ⌈log₂ n⌉ reallocations. Tests cover request-satisfaction, pointer-stability,
  no-op-when-sufficient, and value/empty/pre-sized/ZST extend paths.
- **[RESOLVED 2026-07-02] `SimdCow::scale` copy-then-rescale** — now delegates
  to the fused `mul_scalar_cow` (`broadcast_op` SSOT); halves traffic, removes
  the duplicate implementation.

### Architecture / redundancy / hygiene

- **[RESOLVED 2026-07-02] Tracked scratch at repo root** — `apply_changes.ps1`,
  `do_changes.ps1`, `check_errors.txt` deleted from git; untracked logs removed;
  `check_errors.txt` gitignored. Dead dep declarations dropped (`divan`,
  2×`bytemuck`, intrinsics `rkyv`; facade `rkyv` corrected to dev-dependency).
  `benchmarks/benchmarks_baseline.json`/`benchmarks/benchmarks_results.md` kept (live baseline).
- **[RESOLVED 2026-08-21] `codegen.rs` ungoverned SSOT.** The former 1424-line
  binary was not a complete source of truth: a direct `rustc +1.97.0` build and
  run rewrote all four x86 f32/f64 files while dropping 28 shipped methods
  (five from each AVX2 file and nine from each AVX-512 file). It also had no
  x86 f16 or AArch64 NEON model, no generated-file marker, and no CI freshness
  gate. The binary is deleted, ADR 005 now records checked-in ISA files as the
  canonical sources, and the four files were restored unchanged before the
  cleanup was committed.
- **[open] ~150-200 lines cross-backend scaffold duplication** — compress/expand
  emulation, AVX-512 cmp-mask blend, popcount LUT, masked-reduce, and NEON
  sign-flip constants remain candidates for shared helpers or trait defaults.
  A future consolidation must preserve each ISA/precision contract; the retired
  generator is not a sanctioned destination. `[minor]`.
- **[PARTIAL 2026-07-02] Doc drift** — README `hermes-numeric` entry replaced
  with the eunomia provenance note; the fictional lib.rs feature table replaced
  with the real set; `gemm_int8` example corrected to `gemm::<i8,i8,i32>`.
  ADR number collisions resolved 2026-07-03 (the duplicate 001/002/003 —
  refined-simd-view, target-feature-inlining, numa-memory — renumbered to
  008/009/010 via `git mv` + title fix; 001–010 now unique, no cross-refs
  affected). **Still open:** stale backlog/checklist `hermes-numeric` refs,
  `dispatch/mod.rs` (828) split into trait/impls/facade. `[patch]`.
- **[open] `widen_I8_*` type-named duplicate API + undocumented unsafe + SIMD
  branch untestable at n=5** — collapse to one generic (`#[repr(transparent)]`),
  add SAFETY, differential test at n ∈ {31,32,33,47,1024}. `[patch→minor]`.

### Tests / benches / docs

- **[RESOLVED 2026-07-02] CI runs bare `cargo test`** — both test jobs now run
  `cargo nextest run --workspace` (committed timeout instrument applies) plus
  explicit `cargo test --doc`; x86_64 job gains a `cargo build --examples`
  rot gate (verified green locally; CI run pending next push).
- **[open] ~25 magic-tolerance assertion sites** vs the repo's own demonstrated
  derivation discipline (complex_tests derives the bound); derive+cite each.
  **AVX-512 differential suite silently skips on CI hosts (unreported).**
  **Stale criterion baseline** — newest GEMM/AMX groups ungated; axpy/gemv/complex
  unbenched. **`missing_docs` absent on `hermes-simd-macros`; ~5 doctests for ~60
  facade fns; `# Errors`/`# Safety` gaps.** **No `cargo build --examples`,
  semver-checks, or bench-compile CI gates.** Each `[patch]`/`[minor]`.

### Verified clean (negative results — do not re-chase)

- Proc-macro `#[runtime_dispatch]` ladder: every `#[target_feature]` helper is
  called only behind a compile-time `cfg!` arm or a runtime `is_x86_feature_detected!`
  gate; no bypass, no env override.
- CSR & BlockedCoo SpMV self-defend on adversarial input.
- `tiling/` dims fully checked (prior overflow fix holds in dev+release);
  `AlignedVec` init paths panic-safe (no uninit read); `SimdView` borrow lifetime
  precludes aliased in-place APIs from safe code.
- AMX inline asm operand/clobber model + `AmxSession` Drop under `panic=abort`
  are sound (tile state released on unwind→abort).
- Sparse formats are SoA with `i32` indices; no `Arc/Rc/Box` nesting; SimdCow
  promotion single-allocation; dispatch facade is out-slice/in-place throughout.

## Allocator Dependency Audit - 2026-06-28 (round 7) <a id="audit-2026-06-28-r7"></a>

hermes is unchanged since round 6 and remains lean (no new findings). This round
audited the upstream allocator (`mnemosyne`) that backs `AlignedVec`/the global
path, which was concurrently rewritten lock-free (segment + huge pools as tagged
Treiber stacks; bucket lock removed). Adversarial concurrency review found **no
memory-safety bug**: 16-bit tagged pointers (address in low 48 bits, tag in high
16) are masked before every deref; push/pop CAS loops pair Release/Acquire and
bump the tag (ABA-immune); `take_all` is a single Acquire swap; the huge-pool
first-fit scan **pops-before-touch** (CAS-removes each node before reading it),
avoiding the classic lock-free use-after-free. **Verified hermes integration:
371 workspace tests pass against the lock-free allocator.**

Residual risks (upstream `mnemosyne`, surfaced for the owner — not reworked here,
as it is another agent's fresh, tested code):
- No `loom` model for the lock-free pools — correctness rests on design reasoning
  + std-thread stress tests (empirical tier), not machine-checked interleavings.
  The repo's own rule asks for `loom` alongside stress tests for lock-free code.
- `take_all` head-swap and count-reset are separate atomics → a push interleaving
  between them transiently skews the advisory `retained`/`total_count` counters
  (telemetry only; no safety/correctness impact under the documented contracts).
- Tag lives in the high 16 address bits, so addresses ≥ 2^48 (LA57 / AArch64
  52-bit VA) trip a fail-safe `abort` rather than corrupting — a portability
  limit, not UB. Low-bit tagging (segments are 2 MiB-aligned ⇒ 21 free low bits)
  would remove the dependency and widen the tag.

## Highway Reference Audit - 2026-06-14 <a id="highway-2026-06-14"></a>

Reference: `https://github.com/NikoMalik/highway.git` at
`0984271e74db124cf5e200de542e745348eb0b9e`.

Evidence tier: source audit plus local Hermes code search. No benchmark or
correctness claim is made from this audit alone.

Scope fit:
- In scope for Hermes: target-safe runtime dispatch, lane/mask API coverage,
  safe slice wrappers over unsafe kernel primitives, cross-target conformance
  tests, and x86 baseline coverage below AVX2.
- Out of scope for Hermes: replacing Hermes' domain-specific sparse, packed,
  AMX, tensor, COW, and Atlas-boundary surfaces with Highway's `WithSimd`
  user-kernel model.

Findings:
- [minor] Target-token dispatch safety: Highway exposes a `TargetId` +
  `dispatch_to` path that verifies target support before entering
  `#[target_feature]` trampolines. Hermes has runtime-dispatched public
  functions and direct architecture markers, but no single explicit forced
  target API for tests/benchmarks.
- [minor] Safe slice memory wrappers: Highway separates raw-pointer unsafe
  loads/stores from safe bounds-checked slice wrappers. Hermes has typestate
  views and `AlignedVec`, but `SimdKernel` load/store methods remain raw
  unsafe primitives without a small safe wrapper layer for one-vector
  load/store use cases.
- [minor] SSE2 baseline backend: Highway includes SSE2 as a 128-bit x86_64
  target between Scalar and AVX2. Hermes currently jumps from Scalar to AVX2
  on x86; this leaves older x86_64 machines and conservative CI targets with
  only scalar execution.
- [minor] Cross-target conformance matrix: Highway tests operations by forcing
  every available target and comparing results. Hermes has backend property
  tests and host capability tests, but no common forced-target matrix covering
  the public dense facade consistently across Scalar/AVX2/AVX-512/NEON.
- [minor] Operation-family gap map: Highway documents a broad operation catalog
  across arithmetic, bitwise, comparison, masks, conversions, shuffle/rearrange,
  reductions, float, memory, and crypto. Hermes has strong dense/sparse/packed
  domain kernels, but backlog coverage for missing primitive families is still
  coarse (`gather/scatter variants, additional reductions/scans`).
- [patch] README positioning: Hermes README did not identify the Highway audit
  baseline, making it harder to distinguish intentional scope differences from
  missing SIMD substrate capabilities.

Decisions:
- Do not adopt Highway's `WithSimd` user-kernel model as a replacement for
  Hermes' sealed `SimdKernel` + facade APIs. Hermes' current shape preserves
  Atlas-owned domain kernels and monomorphized public operations.
- Use Highway as a coverage checklist for portable SIMD substrate gaps. Each
  accepted gap must land as a Hermes-native trait/API/test increment with
  value-semantic verification.

Next increments:
- P1: delivered 2026-06-14 as `TargetId`, `dispatch_view_to`, and
  `dispatch_view_mut_to`, with unsupported targets rejected before typed view
  construction.
- P1: delivered 2026-06-14 as safe one-vector `Vector<T, Arch>` slice
  load/store wrappers with length and alignment failure tests.
- P2: delivered 2026-06-21 as SSE2 backend feasibility ADR (ADR 006) covering trait coverage, CI value, and maintenance cost.
- P2: delivered 2026-06-15 as host-supported `TargetId` dense conformance
  tests against Scalar for reductions, elementwise arithmetic, gather, and
  select.
- P3: delivered 2026-06-17 as a per-family coverage map in README and
  backlog, with consumer-demand admission rules for pending families.

## Consumer-Driven SIMD Coverage - 2026-06-15

Evidence tier: value-semantic differential and boundary tests.

- [minor] Batched dense row-panel accumulation: delivered `axpy_rows_batch`
  as one runtime-dispatched fused AXPY-family kernel. The API avoids repeated
  public facade dispatch for depth-major row-panel consumers, allocates no
  temporaries, and keeps output memory traffic to one load/store per output
  lane by accumulating across depth in registers. Coverage compares against
  repeated `axpy_rows` and asserts exact `SimdError::LengthMismatch` failures
  for invalid output stride, alpha panel, and RHS panel extents. Benchmark
  coverage now compares `axpy_rows_batch` against repeated public `axpy_rows`
  on the same depth-major row panels.
- [patch] Dense/AXPY error-contract hardening: selected length-mismatch tests
  now assert exact `SimdError::LengthMismatch` values. This removes
  existence-only failure assertions from the touched dense facade and AXPY
  contract surface.
- [patch] Select/unary error-contract hardening: selected select, unary-map,
  and COW FMA tests now assert exact `SimdError` variants for length mismatch
  and insufficient output capacity.
- [patch] Operation-family error-contract hardening: selected new operation,
  strategy, complex, gather, scan, and COW math tests now assert exact
  `SimdError` variants instead of existence-only failures.
- [patch] COW unary invariant cleanup: `SimdCow::map_unary` no longer
  discards the `SimdView::map_unary` result; the locally constructed equal
  length invariant is explicit in the panic message.
- [patch] GEMM tiling rustdoc cleanup: private implementation names in module
  theorem prose no longer emit public rustdoc private-link warnings.
- [patch] Runtime FMA capability probe: `has_fma3` no longer relies on the raw
  CPUID FMA bit alone; it follows Rust's runtime feature detector and is tested
  against `std::is_x86_feature_detected!("fma")` on x86 hosts.
- [patch] GEMV rustdoc link cleanup: public dispatch docs no longer emit
  ambiguous intra-doc links for same-named GEMV modules and functions.

## NumKong Reference Audit - 2026-06-17 <a id="numkong-2026-06-17"></a>

Reference: `https://github.com/ashvardanian/NumKong` (successor to SimSIMD)

Evidence tier: source audit of features and design philosophy, plus local Hermes code search.

Scope fit:
- In scope for Hermes: Low-level SIMD primitive extensions (fast reciprocal square root with Newton-Raphson refinement, active-lane masked load/store primitives, horizontal reductions and bitwise population count `popcnt`, sub-byte/integer widening/unpacking), target-safe CPU architecture probes.
- Out of scope for Hermes: High-level vector search similarity/distance metric algorithms (Cosine Similarity/Distance, Jaccard Index, Hamming Distance, KL/JS Divergence, geospatial distance) which belong in the Leto operations layer and the Hephaestus GPU layer; thread/MIMD execution scheduling (Moirai).

Findings:
- [minor] Masked tail-load/store elimination: NumKong completely eliminates scalar tail loop overhead on hardware that supports masking (AVX-512, SVE) by using active lane masks. Hermes currently defaults to scalar tail loops in [mod.rs](file:///d:/atlas/repos/hermes/crates/hermes-simd-core/src/ops/mod.rs) for irregular lengths. Supporting first-class masked load/store APIs in Hermes would allow Leto to bypass scalar tails in its domain kernels.
- [minor] Fast reciprocal square root: NumKong optimizes vector norms and Cosine similarity by using hardware-native fast reciprocal square root approximations (`rsqrtps` on x86, `frsqrte` on Arm) refined with a Newton-Raphson iteration. Hermes lacks a fast reciprocal square root strategy in [unary.rs](file:///d:/atlas/repos/hermes/crates/hermes-simd-core/src/ops/unary.rs) and iterates standard/vector `sqrt`.
- [minor] Popcount and horizontal reductions for binary/integer metrics: Binary and integer distance calculations (like Jaccard and Hamming) in Leto require highly efficient SIMD population count (`popcnt`) and horizontal reductions (bitwise reductions). Hermes lacks generic `popcnt` and horizontal bitwise fold primitives.
- [minor] Low-precision integer/sub-byte unpacking: NumKong leverages VNNI algebraic transforms and sub-byte type unpacking (e.g. 4-bit/6-bit) to accelerate low-precision dot products and similarity metrics. Hermes defines sub-byte scalar types like `Bf4` and `F4` but has not exposed vector sign-extension, widening, and byte-alignment unpacking primitives.
- [minor] Arm SME (Scalable Matrix Extension) support: NumKong utilizes Arm SME for tiled matrix multiplication on newer hardware (Apple M4/M5). Hermes lacks SME target feature probes and kernels.

Decisions:
- Do not implement similarity/distance metrics (Cosine, Jaccard, Hamming, KL/JS Divergence) directly in Hermes. These belong in Leto/Hephaestus.
- Implement a Hermes-native fast reciprocal square root (`ops::RecipSqrt` or `rsqrt`) with a Newton-Raphson refinement step to enable fast norm computations in Leto.
- Expose masked load/store primitives on `Vector<T, Arch>` / `SimdKernel` for AVX-512 and SveArch to enable Leto to construct tail-free kernels.
- Add population count (`popcnt`) and horizontal bitwise reduction primitives to the Hermes facade to enable Leto to implement Hamming and Jaccard distance metrics.
- Expose low-precision integer/sub-byte unpacking and sign-extension primitives to support VNNI/Neon dot product optimizations in Leto.
- Defer Arm SME implementation until SVE vector types are fully stabilized and verified.

Next increments:
- P1: Fast reciprocal square root (`ops::RecipSqrt`) with Newton-Raphson refinement.
- P1: Masked tail-load/store API infrastructure for `Avx512` and `SveArch` to enable Leto to run tail-free kernels.
- P2: Expose population count (`popcnt`) and bitwise horizontal reduction primitives to enable Jaccard/Hamming in Leto.
- P2: Expose sub-byte sign-extension and unpacking/widening SIMD primitives (for `Bf4`/`F4`/`I8`).
- P3: Arm SME target-feature feasibility study.

## Allocator / Cross-Repo Audit - 2026-06-26 <a id="alloc-audit-2026-06-26"></a>

Memory-efficiency deep dive on the hermes ↔ Mnemosyne boundary. Evidence tier:
empirical (measured mapped bytes via `mnemosyne::memory_stats`) + type-level
soundness argument for the alignment guarantee.

Root cause found: Mnemosyne routed every allocation with `align > 16` to its
large/huge path, reserving a ~2 MiB segment each (committed on Windows). Because
hermes allocates `AlignedVec<_, Aligned<64>>` pervasively, small SIMD buffers
cost ~2 MiB apiece — **512 live 256-byte/64-aligned `AlignedVec`s mapped
~1056 MiB**. The hermes-side `adjust_layout_for_mnemosyne` 8 KiB padding was a
*counterproductive* workaround for a different (size-based) tcache concern: it
inflated small unaligned allocations into the same huge path without the claimed
benefit (live RSS unchanged in measurement, since the 2 MiB slack is decommitted
for align ≤ 16).

Resolved this sprint:
- [upstream, Mnemosyne `perf/aligned-small-alloc-tcache`] Alignment-aware
  size-class selection: small allocations whose chosen class block stride is a
  multiple of the requested alignment now use the thread-cache path. Sound
  because page starts are `PAGE_SIZE`-aligned and blocks are carved at
  `block_size` stride. Non-power-of-two-stride classes still fall to huge.
  Verified by a value-semantic alignment/usability test.
- [patch] hermes: removed the `adjust_layout_for_mnemosyne` padding and the no-op
  `dealloc_on_node` NUMA bind. With the upstream fix, the same 512-allocation
  workload drops **~1056 MiB → ~4 MiB** mapped (264×).
- [patch] hermes: BlockedCoo `spmv`/`elementwise_mul_dense` unchecked SIMD column
  loads now bounds-guarded (O(nblocks), pre-loop).

Deferred (recorded, not silently dropped):
- The four GEMV dispatchers share a thin register-blocking skeleton, but each
  carries a distinct theorem (operand-reuse vs output-reuse), tile orientation,
  and test suite. The *kernels* are already deduplicated; collapsing the
  dispatcher glue into a proc-macro-attributed `macro_rules!` would obscure the
  per-variant documentation/tests for marginal gain. Left as four clear files.

Resolved later (2026-06-26, scope review):
- CSR `spmv` SIMD-gather column-index bounds (round-1 finding): now validated
  with a linear pre-loop scan in the CSR kernel. CSR SpMV is gather/latency-bound,
  so the O(nnz) linear validation is cheap relative to the random-access gathers
  it guards — the earlier "too expensive" deferral reasoning did not hold for this
  kernel. `spmv_csr` is now sound on adversarial input (negative/oversized indices
  rejected); covered by a `#[should_panic]` test. Sparse SpMV is otherwise a
  low-value SIMD target (gather-bound); the high-value sparse path, SpMM, already
  vectorizes via `axpy`. Consumers (e.g. leto) own format validation at their CSR
  construction boundary and now always route dense/SpMM ops through hermes (the
  `simd` cargo feature was removed downstream — SIMD is the unconditional path via
  hermes's runtime dispatch).

## Internal Audit - 2026-06-26 (round 6, closure) <a id="audit-2026-06-26-r6"></a>

Largely a verification/closure round — 5 prior passes + tool measurements show the
workspace is lean (lib IRs: `mnemosyne` ~3045, `hermes-simd` ~593, `leto-ops` ~167
lines; monomorphization deferred to leaf binaries as intended; no `dyn` on hot
paths; inner-fn extractions confirmed 1-copy). No substantive perf/memory/mono
change was warranted; manufacturing churn would violate the subtractive bias.

Resolved this sprint:
- [patch] Closed the lingering round-1 bitboard finding. `hermes_simd::{rook,
  bishop,queen}_attacks` are safe wrappers over the `Magic` `unsafe` kernel.
  Verified **not** an OOB/UB hole: the kernel indexes `[u64; 64]` magic/offset
  tables and the attack `Vec` with bounds-checked indexing and a computed mask, so
  `square >= 64` **panics**, never reads out of bounds. Added the missing
  `// SAFETY:` justification and `# Panics` docs on the wrappers and a
  `#[should_panic]` regression test. Evidence tier: source audit + test.

## Internal Audit - 2026-06-26 (round 5, monomorphization) <a id="audit-2026-06-26-r5"></a>

Evidence tier: value-semantic differential test (BCOO SIMD) + source-grounded
monomorphization analysis.

Resolved this sprint:
- [patch] hermes `spmv_bcoo` was hardcoded to `ScalarArch` (the only sparse op
  not runtime-dispatched), leaving the SIMD BlockedCoo kernels dead at runtime —
  a perf defect, not cleanup. Now routed through `#[runtime_dispatch]`
  `dispatch_spmv_bcoo`; differential test added for the SIMD branch.
- [patch] hermes `axpy_rows_batch`: extracted the type-independent extent
  validation to a non-generic `#[inline(never)]` fn (emitted once vs. per
  `(T, Arch)`). The validation is run-once-per-call (not the hot loop), so the
  dedup carries no hot-path cost — the correct application of the inner-function
  pattern here.

Verified clean / not pursued (monomorphization):
- Tiling const-generics (`<6,4>`/`<3,3>`/`<1,1>` …) are measured-win register
  blocking dispatched by `LANE_COUNT` — must NOT be collapsed to runtime params.
- Cross-crate inlining is complete (all `SimdKernel` methods `#[inline(always)]`);
  no `dyn`/`Box<dyn>` on any compute path; the one in-loop branch and
  `flush_limit_for::<T>()` are const-foldable (DCE handles per-instance).

Measured and closed (cargo-llvm-lines / cargo-bloat, 2026-06-26):
- Mnemosyne page-list ops (`push_page_front`/`unlink_page_from_list`/`move_*`) —
  **confirmed not worth deduping.** `cargo llvm-lines -p mnemosyne` does not list
  them at all (they are `#[inline(always)]`, fully inlined), and the whole
  `mnemosyne` crate is only ~3045 IR lines, so an `#[inline(never)]` inner-fn
  extraction would save negligible IR while adding a call on the hot free path.
  `#[inline(always)]` (as one agent suggested) dedups nothing. No change; the
  earlier deferral was correct. Tier: empirical (IR measurement).
- hermes monomorphization is lean — the round-5 inner-fn extractions
  (`check_axpy_rows_batch_extents`, `validate_gemm_sizes`) show as **1 copy** in
  `cargo llvm-lines -p hermes-simd` (deduped as intended); the lib's own IR is
  ~593 lines, and an example binary's `.text` is dominated by std runtime glue
  (`rust_eh_personality`), not hermes monomorphization. No bloat to attack.

## Internal Audit - 2026-06-26 (round 4) <a id="audit-2026-06-26-r4"></a>

Evidence tier: value-semantic tests (hermes numeric contract; mnemosyne
`take_all`) + source-grounded contention analysis.

Resolved this sprint:
- [patch] hermes `hermes-numeric`: signed-integer `NumericElement` impls collapsed
  into one `impl_numeric_element_signed!` macro; dead `min_scalar`/`max_scalar`
  integer overrides removed (identical to trait defaults). ~275 fewer lines.
- [patch] hermes `hermes-simd-intrinsics`: AMX raw tile wrappers no longer
  silently no-op on an out-of-range tile (`unreachable!`); `AmxGemm::amx_gemm`
  `# Safety` documents the AMX-availability precondition (already gated by the
  `has_amx()` dispatch probe — not an unguarded hole).
- [upstream, Mnemosyne `perf/segment-purge-batch-detach`] `purge`/`reset` segment
  sweeps batch-detach each node's chain under one lock (`NodeSegmentPool::take_all`)
  instead of one lock per segment — removes decay↔allocator serialization. Pool
  node arrays now built from the `NUMA_BUCKETS` SSOT.

Considered, deferred (recorded):
- NEON `neon_f32`/`neon_f64` (~92% overlap) is seam-level, not a clean macro: the
  divergent 8% (popcount reduction depth, `cmp_ne` u64 round-trip, `swap_adjacent`
  instruction, and mask construction) needs a complete backend-family design,
  not a thin suffix macro. Any future consolidation must be evaluated against
  the checked-in-source decision in ADR 005.
- scalar `f32`/`f64` kernels are a cleaner macro/const-generic candidate (no
  intrinsics); deferred to keep this round focused.

## Internal Audit - 2026-06-26 (round 3) <a id="audit-2026-06-26-r3"></a>

Evidence tier: compile-time invariant encoding + value-semantic tests (hermes);
value-semantic test + source-grounded retention analysis (Mnemosyne).

Resolved this sprint:
- [patch] hermes `view/vector_reg.rs` was the one module left out of the
  `MAX_SIMD_LANES` SSOT migration: 10 sites used `[_; 128]` buffers with dead
  `assert!(lane_count <= 128)` runtime checks. Migrated to `MAX_SIMD_LANES` (64)
  + compile-time `LANE_BOUND_CHECK`; magic `64` OOB guard → `u64::BITS`.
- [patch] hermes `tensor/view.rs` (601 lines) split into a vertical `tensor/view/`
  hierarchy: core (`mod.rs`), `rank_ops.rs`, `simd_bridge.rs` — SoC, pure
  relocation.
- [upstream, Mnemosyne `perf/huge-pool-byte-cap`] Huge-pool retention was bounded
  only by per-bucket block count (1024), allowing ~16 GiB/bucket of idle mappings;
  now byte-bounded per bucket (`bucket_block_cap`, ~256 MiB) while small-huge
  buckets keep the full count cap. Plus a redundant per-`pop` atomic reload removed.

## Internal Audit - 2026-06-26 <a id="audit-2026-06-26"></a>

Four-dimension sweep (safety, contention-free perf, memory, redundancy).
Evidence tier: compile-time invariant encoding and value-semantic tests for the
fixes below; source audit for the deferred item.

Resolved this sprint:
- [patch] Scalar-fallback buffer over-provisioning: `MAX_SIMD_LANES` was `128`,
  2× the true workspace maximum `LANE_COUNT` of `64` (AVX-512 `i8`). Lowered to
  `64`, halving every fallback stack frame; the two divergent local bounds
  (`reduction.rs::finalize` `MAX_LANE_COUNT = 64` debug-assert, and the
  bitmask-buffer `u64::BITS` guard) now fold onto the SSOT under the compile-time
  `LANE_BOUND_CHECK`. Evidence: const-eval catches a too-low bound (AVX-512 `i8`
  fails to build), so the value is the verified tight maximum.
- [patch] NUMA alloc-generation memory ordering: the cross-thread cache
  invalidation counter used `Relaxed` (no happens-before — a reader could trust a
  stale locality flag for a recycled address) and re-read the generation after
  the OS probe (a TOCTOU window stamping pre-bump data with the post-bump
  generation). Now `Release`/`Acquire` with a single pre-probe capture.
- [patch] `build_index_vector` layout invariant: the `&[i32] → IndexVector`
  unaligned read now carries a `const` size assert, so a layout-mismatched
  backend is a build error, not an OOB read.
- [patch] `#![forbid(unsafe_code)]` on `hermes-simd-macros` (no executable
  unsafe — only generated tokens). Magic-table init CAS success ordering relaxed
  to `Relaxed` (winner acquires no shared data).
- [patch] Redundancy: three byte-identical target-gated `SimdOps` impls collapsed
  to one `impl_simd_ops_methods!` macro (mod.rs 1217→845); `flush_limit` deduped
  to a `const fn` SSOT.

## Resolved

- [minor] Safe-code ISA fault surfaces (2026-07-05). `AmxSession::new` and
  `AmxBatchSession::begin` became fallible (`AmxSessionError::UnsupportedTarget`)
  and guard `ldtilecfg`; `AmxSession::release` guards `tilerelease`. `SimdArch`
  now owns `is_runtime_supported()`, with `TargetId` and safe vector/mask wrappers
  sharing that SSOT. `Vector::<T, Avx512>::try_zero`, `try_splat`,
  `try_from_array`, and checked slice wrappers return `SimdError::UnsupportedTarget`
  on unsupported hosts before entering AVX-512 code; infallible conveniences
  panic before ISA execution. Regression coverage asserts AMX session rejection
  and AVX-512 constructor/slice-wrapper rejection on unsupported x86 hosts.
  Evidence tier: type-level runtime-support seam + value-semantic regression.
- [patch] AVX-512 tile-kernel detection soundness (2026-07-02). The `Avx512Support`
  probes read raw CPUID without OSXSAVE/XCR0 (a host advertising the bit without OS
  XSAVE enablement would `#UD`) and, for bf16, tested the unrelated `avx512bf16`
  dot-product bit while the tile kernel enables `avx512f,avx512bw,avx512vl` and
  never issues `dpbf16` — both a fault window and a false skip on capable non-bf16
  parts. Now `is_x86_feature_detected!` for the exact enabled set per kernel
  (`avx512f,avx512bw,avx512vl`; `avx512f,avx512vnni,avx512vl`), which handles XCR0
  and the leaf-7 max-leaf internally. `widen_i8_to_i16`'s AVX-512 branch now gates
  on `avx512bw` (the `_mm512_cvtepi8_epi16` requirement) instead of the
  `avx512f`-only `TargetId::Avx512`, closing a KNL `#UD`. AMX detection is
  conservatively disabled in the same change (see round 8, MITIGATED). Evidence
  tier: source audit + toolchain feature-availability (AMX strings unstable).
- [patch] SELL-p vectorized SpMV / elementwise OOB read (2026-07-02). The
  `LANE_COUNT == C` fast path in `sparse/spmv.rs::sellp_spmv_vectorized` and
  `sparse/ops.rs::elementwise_mul_dense` gathered `x[col_idx]` and loaded
  `values[offset..]`/stored `out_values[offset..]` at full vector width with no
  bounds check, reachable from the safe `SparseView::<SellP<C>>::{spmv,
  elementwise_mul_dense}` — a proven OOB read from safe code on a
  caller-constructed matrix (`pub` fields, no-op `new`, opt-in `validate()`).
  Unlike the sibling CSR/BlockedCoo paths, SELL-p never validated. Fixed by
  routing both vectorized paths through the SSOT `SparseValidate::validate()` via
  the new `spmv::assert_sellp_validated` before the unsafe kernel (checks
  `col < ncols`, `col_indices.len() == values.len()`, and
  `slice_ptr[s] + slice_col_count[s]·C <= values.len()`), plus an
  `out_values.len() >= values.len()` guard on the elementwise store. Two
  `#[should_panic]` regressions drive the Scalar-backed vectorized path
  (`Scalar::LANE_COUNT 4 == C`, host-independent) with an out-of-range column and
  with over-long slice geometry. Evidence tier: type-level precondition + value
  regression. See [round 8](#audit-2026-07-02-r8).
- [patch] Scalar-fallback stack-buffer lane bound (2026-06-24). The default
  `SimdKernel` methods and `kernel_helpers` emulations stored a full vector into
  a fixed `[MaybeUninit<T>; 128]` buffer with the `LANE_COUNT <= 128` invariant
  unasserted (and misleadingly half-guarded by `LANE_COUNT.min(128)` on the read
  loop but not the unclamped `store_unaligned`). A backend with `LANE_COUNT > 128`
  (e.g. a future 2048-bit SVE `i8` at 256 lanes — see the SVE residual below)
  would silently overflow the stack. Now encoded at compile time:
  `MAX_SIMD_LANES` SSOT constant + `SimdKernel::LANE_BOUND_CHECK` asserted per
  backend at monomorphization (validated by a deliberate lower-the-bound build
  that fails AVX-512 compilation). `generic_mask_from_bitmask` gains the
  matching `LANE_COUNT <= u64::BITS` guard. Evidence tier: compile-time
  invariant encoding (strongest available).
- [patch] rust-1.95 clippy workspace lints resolved (redundant `as` casts,
  `iter().flatten()`, needless borrow/return, `enumerate()` range loops);
  `cargo clippy --workspace --all-targets -- -D warnings` is clean again.
- [minor] Masked-merge `SimdKernel` defaults (2026-06-28). Investigation of the
  remaining monomorphization gaps for SIMD capability expansion found the seam
  already mature: `rsqrt`, `popcount`, horizontal-bitwise reductions (NumKong
  P1/P2 — see [numkong-2026-06-17](#numkong-2026-06-17)), and the
  reduction/scan/unary op families are expressed as defaulted `SimdKernel`
  methods (`kernel_helpers`) or sealed ZST strategies, so each is one generic
  addition inherited by every backend. The single family still `required` on
  every impl was the masked-merge set (`masked_load_unaligned`,
  `masked_store_unaligned`, `masked_add`, `masked_mul`, `masked_fmadd`,
  `masked_sum_reduce`) — the NumKong P1 tail-free family — which now has
  scalar-emulated trait defaults (arithmetic via `blend(mask_to_vector(mask), …)`;
  load/store via new `kernel_helpers::generic_masked_{load,store}`). A new
  backend/type inherits the tail-masked family for free; the six redundant impls
  are removed from `impl_emulated_kernel!` (inherited by ~24 emulated backends).
  Bit-identical to the removed per-element loops, verified by a new cross-backend
  differential property test (Scalar/SveArch defaults vs AVX2/AVX-512 native
  overrides). Evidence tier: differential test across default and native paths.
  Not defaulted: `gather`/`compress`/`expand` stay `required` — no generic
  `IndexVector`/lane-introspection primitive exists to express them, and gather is
  latency-bound so the value is low. Remaining NumKong families (native `rsqrt`
  instruction override, sub-byte unpacking, Arm SME) are capability *additions*,
  not monomorphization debt — each is now a single defaulted-method or
  strategy-ZST addition rather than an N-impl change.

- [patch] Tiling dimension-product overflow → OOB SIMD load (2026-06-28). The
  GEMV/GEMM operand-length checks (`tiling/gemv.rs`, `gemv_transpose.rs`,
  `gemm.rs`) computed the required span with unchecked `usize` products as the
  only guard before `unsafe` SIMD loads/stores. An adversarial dimension from the
  public dispatch API (`lda = usize::MAX`, `nrows = 2`; or `m·k` etc.) overflowed:
  release (`overflow-checks = false`) wrapped → guard passed → OOB read; dev
  panicked undocumented. Fixed by an SSOT `tiling::dims` module
  (`checked_strided_span`/`checked_area`) returning `SimdError::LengthMismatch` on
  overflow — closes the OOB path in all profiles and consolidates the previously
  duplicated forward/transpose span math. Added `[profile.dev] overflow-checks =
  true` per the numerical-discipline mandate (release keeps default for hot-loop
  speed; the checked guard makes safety profile-independent). Evidence tier:
  value-semantic exact-variant regression tests on all three dispatchers passing
  in both dev and release (release pass proves the OOB load unreachable) +
  `tiling::dims` unit tests. Prior rounds closed per-element sparse-load overflow
  but never this dense dimension-product class.

- [patch] Integer `sqrt` f64-roundtrip precision loss (2026-06-28). Integer
  `NumericElement::sqrt` used `(self as f64).sqrt() as Self`, which rounds operands
  above 2⁵³ to `f64` before the root — wrong for large `i64`/`u64` (`u64::MAX`
  returned 2³² instead of 2³²−1; the result's square overflows). Replaced with
  exact `isqrt`; signed negatives keep the documented degenerate contract (→ 0;
  integers have no `NaN`), trait doc now states the contract. The audit's
  companion flag (`f16`/`bf16` `to_f64` via `to_f32`) was assessed **benign** —
  widening to a wider mantissa is lossless — and left unchanged. Evidence tier:
  value-semantic regression tests (large-operand exact cases, the
  `r² ≤ n < (r+1)²` invariant above 2⁵³, negative-input contract) over all eight
  integer types; integer `sqrt` previously had zero test coverage and no callers.

- [patch] `recip_sqrt` backend-inconsistent precision (2026-06-28). The f64 SIMD
  paths (avx2/avx512/neon) and NEON f32 reused the f32 single-Newton-step pattern,
  which under-refines a low-bit hardware `rsqrt` seed: `recip_sqrt::<f64>` ranged
  ~1e-16 (scalar) → ~4e-9 (avx512) → ~6e-8 (avx2) → ~1.5e-5 (neon), and NEON f32
  reached only ~16 bits — a native-precision violation masked by perfect-square
  test inputs (Newton converges exactly by luck) + magic `1e-4`/`1e-6` tolerances.
  Fixed to full native precision (~1 ulp) on all backends: f32 retains fast
  `rsqrt`+Newton (NEON now two steps for full 23-bit); f64 uses correctly-rounded
  hardware `sqrt`+divide (3 Newton steps would otherwise be needed). Contract
  documented on the trait method. Evidence: cross-backend differential test with
  analytically-derived relative bounds (`8·ε_f32`, `4·ε_f64`) over non-perfect-
  square inputs — x86 backends runtime-verified locally; **NEON is verified only on
  the aarch64 CI runner** (no local aarch64 target), though the new intrinsics are
  the same ones already used by the NEON `sqrt`/`div`/`splat` primitives.

## Hermes audit 2026-08-15 <a id="hermes-2026-08-15"></a>

- [patch] The exact integrated provider head `7343402a` had 23 source files
  over the 500-line hierarchy target; comparison with the preceding Atlas
  pointer showed this was stale conformance state, not a new regression. The
  546-line `ops/reduction.rs` mixed the reduction contract with the
  multiplicative strategy. Moved `Product` and its sealed generic
  `ReductionOp` implementation into the dedicated
  `ops/reduction/product.rs` leaf. The parent is now 442 lines, the new leaf
  is 60 lines, and the public export remains unchanged. Evidence: structural
  line-count audit plus provider `cargo fmt --check`, `cargo check`, clippy
  with `-D warnings`, and `cargo nextest run -p hermes-simd --all-features`
  (410/410 passed). This is a hierarchy/ownership cleanup; no performance
  claim is made without a controlled benchmark.

## Residual Risks

- AVX-512 and AMX runtime validation still depends on matching hardware.
- Native SVE remains a planned backend blocked by the pinned stable Rust
  toolchain: current `SveArch` coverage is an emulated value-semantic backend.
  `SveArch::is_native_hardware_supported` reports hardware capability separately
  and does not claim native execution. When a native SVE backend lands, `LANE_COUNT`
  may exceed `MAX_SIMD_LANES` for narrow element types; the new
  `LANE_BOUND_CHECK` will flag it at compile time so the scalar-fallback buffers
  are widened (or the backend overrides the affected methods natively) before it
  builds.
- The local `[patch]` graph warns that `mnemosyne-heap` is unused; this is not
  introduced by the Highway audit, but remains a supply-chain hygiene item.
- [minor, deferred] NUMA alloc-generation is a single global counter bumped on
  every dealloc/realloc. On a multi-NUMA + AMX host under heavy alloc churn this
  is a true-sharing serialization point and over-broad (a free on any node
  invalidates every node's thread-local cache). Sharding the generation per NUMA
  node would remove both, but requires threading the node through the allocator
  bump API; deferred as it only affects multi-node AMX hosts and needs careful
  node attribution. The ordering/TOCTOU correctness fix landed this sprint.
- `.config/nextest.toml` added this sprint (30s slow / 60s terminate), making the
  mandated test-time budget enforced rather than implicit. The suite currently
  runs in ~2.4s, well under the threshold.
