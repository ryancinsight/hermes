# Changelog

All notable changes to the hermes-simd workspace. Format: [Keep a Changelog]; versioning: SemVer 2.0.0 (pre-1.0: minor releases may break, documented under **Breaking**).

## [Unreleased]

### Added
- `hermes-numeric` [minor]: `NumericElement` and `CastFrom` coverage for `i64`
  and `u8`/`u16`/`u32`/`u64`, plus the crate's first test module — value-semantic
  contract tests for every integer impl cross-checked against std (bitops,
  popcount, wrapping fmadd, min/max, constants, `CastFrom` round-trips).

### Changed
- `hermes-simd-core` [minor]: give the six masked-merge `SimdKernel` methods
  (`masked_load_unaligned`, `masked_store_unaligned`, `masked_add`, `masked_mul`,
  `masked_fmadd`, `masked_sum_reduce`) scalar-emulated trait defaults — the
  arithmetic via `blend(mask_to_vector(mask), …)` and the load/store via new
  `kernel_helpers::generic_masked_{load,store}` — so a new backend or scalar type
  inherits the tail-masked family for free instead of hand-implementing it. These
  were the last capability family still `required` on every impl (rsqrt, popcount,
  horizontal-bitwise, reductions, and scans were already defaulted), the one
  paying an N-impl tax that gated cheap backend/type expansion. The six redundant
  hand-written impls are removed from `impl_emulated_kernel!` (~66 lines, inherited
  free by all ~24 emulated backends); native AVX2/AVX-512/NEON overrides are
  unchanged. Behavior is bit-identical to the removed per-element loops, verified
  by a new cross-backend differential property test (Scalar/SveArch defaults vs
  AVX2/AVX-512 native overrides). `gather`/`compress`/`expand` stay `required`
  (no generic `IndexVector`/lane-introspection primitive to default them).
- `hermes-simd` [patch]: extract `axpy_rows_batch`'s type-independent extent
  validation into a non-generic `#[inline(never)]` `check_axpy_rows_batch_extents`
  so it is emitted once instead of re-monomorphized into every `(T, Arch)`
  instantiation of the kernel (the validation runs once per call, not in the hot
  loop, so the dedup has no hot-path cost).
- `hermes-numeric` [patch]: consolidate the signed-integer `NumericElement`
  impls (`i8`/`i16`/`i32`/`i64` were four hand-copied blocks) into one
  `impl_numeric_element_signed!` macro mirroring the unsigned one, and drop the
  `min_scalar`/`max_scalar` overrides from every integer impl — they were
  byte-for-byte identical to the `PartialOrd`-based trait defaults (the float/half
  overrides stay for NaN semantics). Net ~275 fewer lines, behavior unchanged
  (verified by the existing integer-contract tests).
- `hermes-simd-core` [patch]: finish the `MAX_SIMD_LANES` SSOT migration in
  `view/vector_reg.rs` — the `Vector` scalar-fallback buffers (`Debug`,
  `PartialEq`, `to_bitmask`, `cast`, `extract`, `insert`, masked slice load/store)
  were still hardcoded `[_; 128]` with dead `assert!(lane_count <= 128)` runtime
  checks. They now use the named `MAX_SIMD_LANES` (64) const with the compile-time
  `LANE_BOUND_CHECK`, halving those stack frames and converting the dead runtime
  asserts into per-backend compile errors; the masked-slice OOB guard's magic `64`
  is now `u64::BITS`.
- `hermes-simd-core` [patch]: split the 601-line `tensor/view.rs` into a vertical
  `tensor/view/` hierarchy by concern — `mod.rs` (core N-D struct, constructors,
  rank-agnostic accessors), `rank_ops.rs` (rank-2/3 specialized views + transpose),
  and `simd_bridge.rs` (the rank-1 → `SimdView` seam). Pure relocation; behavior
  unchanged.
- `hermes-simd-core` [patch]: drop `adjust_layout_for_mnemosyne`, the small-alloc
  padding that inflated every `<=8KB` NUMA allocation to `8192+align` bytes to
  "bypass the thread-local cache". That routed small allocations into Mnemosyne's
  ~2 MiB-per-allocation huge path; the small thread-cache path is correct,
  NUMA-partitioned, and bounded. Combined with the Mnemosyne alignment-aware
  small-path fix (`Mnemosyne perf/aligned-small-alloc-tcache`), 512 live
  256-byte/64-aligned `AlignedVec` allocations drop from ~1056 MiB to ~4 MiB
  mapped (measured). Also removed the no-op NUMA thread bind in `dealloc_on_node`
  (a free routes by the pointer's owning segment, not the caller's node).
- `hermes-simd-core` [patch]: encode the scalar-fallback stack-buffer lane bound
  at compile time and tighten it to the true maximum. The default `SimdKernel`
  methods (`scan_vector`, `swap_adjacent`, `dup_even`/`dup_odd`) and the
  `kernel_helpers` scalar emulations store a full vector into a fixed
  `[MaybeUninit<T>; N]` stack buffer, so a backend whose `LANE_COUNT > N` would
  silently overflow it (UB). `N` is the named SSOT constant `MAX_SIMD_LANES`,
  now `64` (the workspace maximum, AVX-512 `i8`) rather than the previous
  over-provisioned `128` — halving every fallback frame. A defaulted associated
  const `SimdKernel::LANE_BOUND_CHECK` (referenced via inline `const {}` in each
  buffer method) asserts `LANE_COUNT <= MAX_SIMD_LANES` per backend at
  monomorphization, turning a would-be silent overflow into a compile error.
  `reduction.rs::finalize` (formerly a divergent `MAX_LANE_COUNT = 64` + debug
  assert) and `generic_mask_from_bitmask`'s bitmask buffer now both fold onto
  this SSOT under the compile-time check.
- `hermes-simd` [patch]: `dispatch_axpy` and `dispatch_scale` use a 4-accumulator
  unrolled SIMD body to break the store-to-load dependency chain, matching the
  throughput model used by `dot`.
- `hermes-simd` [patch]: the three target-gated `impl SimdOps for T` blocks
  (byte-identical 206-line bodies differing only in their `where` kernel bound)
  collapse into one `impl_simd_ops_methods!` macro (`dispatch/mod.rs` 1217 → 845
  lines); the per-call element-width flush limit in `view/reduce.rs` dedupes to a
  single `const fn flush_limit_for::<T>()`.
- `hermes-simd-macros` [patch]: `#![forbid(unsafe_code)]` (the crate executes no
  unsafe; the unsafe it emits lives in generated token streams).
- `hermes-simd-intrinsics` [patch]: magic-table init CAS success ordering relaxed
  from `Acquire` to `Relaxed` (the 0→1 winner acquires no shared data).

### Fixed
- `hermes-numeric` [patch]: integer `NumericElement::sqrt` computed
  `(self as f64).sqrt() as Self`, rounding operands above 2⁵³ to `f64` *before*
  taking the root — lossy for large `i64`/`u64` (e.g. `u64::MAX.sqrt()` returned
  4_294_967_296, whose square overflows `u64`; the correct floor root is
  4_294_967_295). Now uses exact integer `isqrt`; signed negatives keep the
  defined degenerate contract (return 0 — integers have no `NaN`). Trait doc states
  the integer/float/negative contract. Covered by new value-semantic tests: exact
  small roots for all eight integer types, the large-operand regression cases
  (`u64::MAX`, `i64::MAX`), the `r² ≤ n < (r+1)²` invariant above 2⁵³, and the
  negative-input contract.
- `hermes-simd-core` [patch]: **memory-safety** — the tiling GEMV/GEMM dimension
  checks computed the required operand span with unchecked `usize` arithmetic
  (`(nrows−1)·lda + ncols`, `m·k`, `k·n`, `m·n`) as the *sole* guard before
  `unsafe` SIMD loads/stores. An adversarial dimension reachable from the public
  dispatch API (e.g. `dispatch_gemv_strided(.., nrows=2, lda=usize::MAX)`)
  overflowed the product: under release `overflow-checks = false` it wrapped to a
  small value, the `a_len < a_needed` guard passed, and the kernel read out of
  bounds (and panicked undocumented in dev, where checks default on). Span math is
  now one SSOT module `tiling::dims` (`checked_strided_span`/`checked_area`,
  shared by the forward and transpose GEMV checkers — previously duplicated) that
  returns `SimdError::LengthMismatch` on overflow, closing the OOB path in every
  profile independently of `overflow-checks`; the checked bound also proves the
  kernels' own `row_idx·lda` index arithmetic cannot overflow. Added
  `[profile.dev] overflow-checks = true` (explicit per the numerical-discipline
  mandate; release keeps the default for hot-loop speed). Verified by exact-variant
  overflow regression tests on all three dispatchers, passing in **both** dev and
  release (the release pass is the proof the OOB load is unreachable), plus
  `tiling::dims` unit tests.
- `hermes-simd` [patch]: `spmv_bcoo` was hardcoded to `ScalarArch`, so the
  runtime-dispatched SIMD BlockedCoo kernels (and their bounds guards) were dead
  — every blocked-COO SpMV ran scalar regardless of host SIMD. It now routes
  through a `#[runtime_dispatch]` `dispatch_spmv_bcoo` like the CSR/SELL-P/
  dense-masked paths, selecting AVX-512/AVX2/NEON/scalar at runtime. Covered by a
  differential test exercising the SIMD branch against a scalar reference.
- `hermes-simd-core` [patch]: harden the NUMA alloc-generation cross-thread
  invalidation signal. The counter now publishes with `Release` and is read with
  `Acquire` (was `Relaxed`, which gave no happens-before, so a reader could trust
  a stale locality flag for a recycled address), and `verify_numa_locality`
  captures the generation once before the OS residency probe instead of
  re-reading it at store time — closing a TOCTOU window where a concurrent bump
  stamped pre-bump probe data with the post-bump generation.

### Safety
- `hermes-simd` [patch]: documented the `# Panics` contract (`square >= 64`) and
  added the `// SAFETY:` justification on the public `rook_attacks`/`bishop_attacks`/
  `queen_attacks` wrappers over the `Magic` `unsafe` kernel — verified the kernel
  uses bounds-checked table indexing (panics, never OOB), closing a round-1
  finding; backed by a `#[should_panic]` test.
- `hermes-simd-intrinsics` [patch]: the raw AMX tile wrappers
  (`tilezero`/`tileloadd`/`tilestored`/`tdpbf16ps`/`tdpbssd`) replaced their
  silent `_ => {}` fallthrough with `unreachable!` so an out-of-range tile index
  is a loud panic rather than a silently-dropped compute step; documented the
  AMX-availability precondition (CPU feature + OS tile-state enable) on
  `AmxGemm::amx_gemm`'s `# Safety` (it is reached only via the `has_amx()`-gated
  dispatch path).
- `hermes-simd-core` [patch]: CSR `spmv` now validates every column index is
  `< ncols` (linear pre-loop scan) before the unchecked SIMD gather `x[cols[j]]`,
  making the safe `spmv_csr` sound on malformed input (negative/oversized indices
  panic instead of reading out of bounds). The scan is cheap relative to the
  gather-bound kernel; covered by a `#[should_panic]` test.
- `hermes-simd-core` [patch]: the BlockedCoo `spmv` and `elementwise_mul_dense`
  kernels issued unchecked `load_unaligned` reads of `BN` lanes at each block's
  column base with no guarantee the span stayed within `x`/`dense`. Added an
  O(nblocks) pre-loop guard (every block's column span fits the input, row span
  the output) plus dense/output buffer-size checks, so a malformed block
  coordinate panics rather than reading out of bounds.
- `hermes-simd-core` [patch]: `build_index_vector` binds its `IndexVector` layout
  assumption with a `const` assert (`size_of::<IndexVector>() == LANE_COUNT *
  size_of::<i32>()`), so a layout-mismatched backend is a build error rather than
  an out-of-bounds unaligned read.

## [0.3.0] — 2026-06-21

### Added
- Public runtime-dispatched `gemv` (`y += A·x`, register-blocked level-2 BLAS
  matrix–vector product) plumbing the existing `TilingStrategy::gemv` /
  `gemv_impl` core through the `SimdOps` dispatch trait, in its own
  `dispatch/gemv.rs` leaf module. `TILE_M` row-blocking scales with the register
  file (8/4/1 by lane count); the operand-reuse theorem is documented inline.
  Value-semantic differential tests vs a scalar reference across shapes
  (incl. `TILE_M` remainder + column tail), accumulate semantics, and the
  length-mismatch error path; a `gemv_f32` Criterion benchmark vs a scalar
  row-by-row reference (measured ≈9× at 256² on the local AVX2 host).
- Public runtime-dispatched `gemv_transpose` (`y += Aᵀ·x`), the complement of
  `gemv`: a new register-blocked `gemv_transpose` core kernel
  (`tiling/gemv_transpose.rs` + `TilingStrategy::gemv_transpose`) plus the
  `dispatch/gemv_transpose.rs` leaf. Computes `Σᵢ xᵢ·A[i,:]` (sum of scaled rows),
  vectorizing across the `ncols` output lanes with **no horizontal reduction**;
  `TILE_N` blocks output lane-chunks for accumulator reuse across rows. Inline
  output-reuse theorem, value-semantic differential tests across shapes (incl.
  `TILE_N` remainder + column tail), accumulate semantics, error path; and a
  `gemv_transpose_f32` Criterion benchmark vs a scalar reference.
- Public runtime-dispatched `gemv_strided` (`y += A·x` over a row-major
  **sub-matrix** with leading dimension `lda ≥ ncols`). The core `gemv` kernel is
  generalized in place to `gemv_strided_impl(.., lda)` (DRY — packed `gemv` now
  delegates with `lda = ncols`, bit-for-bit unchanged, verified by a test), with
  `TilingStrategy::gemv_strided` and `dispatch/gemv_strided.rs`. Admits matvec
  over a trailing/leading block of a larger buffer (e.g. a reflector apply's
  column block) without copying it out. Differential test over a true sub-matrix
  (`lda > ncols`), a packed-equals-`gemv` equivalence test, an `lda < ncols` /
  short-span rejection test, and a `gemv_strided_f32` Criterion benchmark over a
  padded buffer (the gapped-row access path).
- Public runtime-dispatched `gemv_transpose_strided` (`y += Aᵀ·x` over a
  row-major sub-matrix, row stride `lda ≥ ncols`) — the transpose analogue of
  `gemv_strided`. The `gemv_transpose` core kernel is generalized in place to
  `gemv_transpose_strided_impl(.., lda)` (DRY — packed `gemv_transpose` delegates
  with `lda = ncols`, verified bit-for-bit equal by a test), with
  `TilingStrategy::gemv_transpose_strided` and a `dispatch/` leaf. Admits the
  `Aᵀ·x` reduction over a strided block (e.g. forming `Aw = Σⱼ wⱼ·colⱼ` in a
  reflector apply) without copying. Differential test over a sub-matrix, a
  packed-equals-`gemv_transpose` equivalence test, and an invalid-`lda` rejection.
- Generic absolute reductions: `AbsSum` / `AbsMax` reduction strategies plus
  runtime-dispatched `abs_sum` / `abs_max` APIs for Leto/Apollo norm paths,
  using transformed SIMD seeds and transform-free partial merges to avoid
  temporary absolute-value buffers.
- Criterion benchmark suite for the interleaved complex kernels (`benches/complex.rs`), runtime-dispatch vs `Scalar` series as a built-in regression reference.
- CI: `cargo miri test` job over hermes-simd-core (pointer/view/cow logic) and a `--no-default-features` build check.
- `#![deny(missing_docs)]` on all six public crates; remaining undocumented items (bitboard backends, AMX submodules, emulated-kernel macro, magic `OnceLock`) documented.
- Internal x86 VNNI asm instruction macro for `vpdpbssd`, keeping assembly behind the monomorphized tile-matmul backend contract without a hot-loop wrapper call.
- Benchmark reports now record benchmark-relevant host ISA features and the runtime dense-dispatch backend selected on the runner.
- Benchmark regression enforcement now uses `benchmarks_baseline.json` plus
  `run-benches --check-regressions`; the runner is split into CLI, Criterion
  parsing, host reporting, Markdown rendering, and threshold-check modules.
- Sparse SpMV benchmarks now sweep CSR, SELL-p, and Blocked-COO across 1K,
  10K, and 100K rows at 0.1%, 1%, and 10% non-zero density; Dense-with-mask is
  capped at 10K rows to bound local memory use.
- Packed4 COW unpacking now has a focused Criterion benchmark target over
  1K, 16K, and 256K logical elements for both public packed formats.
- Runnable doctests now cover the public complex kernels, sparse CSR
  `SparseCow` SpMV, and const-generic `TensorView` construction/access paths;
  `hermes-simd-core` doctests are enabled.
- Host-capability integration tests validate runtime dispatch, local AVX2 execution when available, and irregular-shape GEMM fallback coverage.
- Miri coverage now extends to the `hermes-simd-intrinsics` boundary: AMX session state is tested under Miri while hardware execution paths remain native-only.
- `parallel` and `mnemosyne-memory` are default features on every Hermes package; `mnemosyne-memory` routes `AlignedVec::with_capacity_numa` allocation and deallocation through Mnemosyne by default.
- `hermes-simd` re-exports `SveArch` with the other architecture markers, and
  the kernel property suite now exercises its mask, compress/expand, gather,
  and leading-tail invariants on every host.
- Core kernel, compute, and tiling Rustdoc examples now run as value-semantic
  doctests instead of compile-only `no_run` examples.
- `BitMask` native-mask conversion and active-lane iteration examples now run
  as value-semantic doctests.
- `TargetId`, `dispatch_view_to`, and `dispatch_view_mut_to` provide an
  explicit target-token surface for tests and benchmarks, rejecting unsupported
  targets before constructing architecture-specific views.
- `Vector<T, Arch>` now has safe one-vector slice wrappers for aligned and
  unaligned load/store, returning value-semantic `SimdError` variants for short
  or misaligned slices while preserving raw pointer kernels for hot loops.
- Host-capability tests now force every supported `TargetId` and compare dense
  facade sum, dot, elementwise arithmetic, gather, and select results against
  the scalar target.
- `axpy_rows_batch` adds one runtime-dispatched fused kernel for
  `out[row, i] += sum_k alphas[k, row] * x_panel[k, i]`, avoiding repeated
  facade dispatch when a consumer accumulates a dense row panel; the kernel
  accumulates each output lane across depth in registers and stores it once.
- Dense Criterion benchmarks now include `axpy_rows_batch_f32`, comparing the
  fused row-panel kernel against repeated public `axpy_rows` calls.
- Dense and AXPY length-mismatch tests now assert the exact
  `SimdError::LengthMismatch` contract instead of existence-only failures.
- Select, unary-map, and COW FMA error-path tests now assert exact
  `SimdError` variants for length and output-capacity failures.
- New operation, strategy, complex, gather, scan, and COW math error-path
  tests now assert exact `SimdError` variants for invalid shape, short output,
  and invalid index contracts.
- `SimdCow::map_unary` now asserts its internally constructed output-length
  invariant instead of silently discarding the impossible `map_unary` error.
- GEMM tiling module docs now avoid private intra-doc links, keeping workspace
  rustdoc warning-clean after the vertical tiling split.
- README/backlog now include an operation-family coverage map that distinguishes
  delivered SIMD families from consumer-demand pending families.
- Runtime FMA support probing now uses Rust's platform-aware feature detector
  behind a cached `has_fma3` helper and `FmaSupport` trait impls.
- GEMV dispatch docs now disambiguate function links from same-named modules,
  keeping rustdoc warning-clean.

### Changed
- `SveArch` is now a callable 512-bit-shape emulated backend for f32/f64
  (`16xf32`, `8xf64`) instead of a public marker with `unimplemented!()`
  kernel methods. Native SVE intrinsics remain a separate pending backend.
- Blocked-COO SpMV dispatch now uses one const-generic `spmv_bcoo::<T, BM, BN>`
  entry point, so tile shape monomorphizes from the call site instead of
  cloning fixed 4x4 and 8x8 public functions.
- SELL-p SpMV dispatch now uses one const-generic `spmv_sellp::<T, C>` entry
  point, preserving runtime architecture dispatch while removing fixed slice
  height functions.
- Interleaved complex `mul_assign` now processes two SIMD registers per loop
  iteration on SIMD backends and uses a direct four-pair scalar loop for large
  scalar inputs, reducing loop overhead in the measured complex benchmark
  range.
- `benchmarks_baseline.json` and `benchmarks_results.md` now include the
  packed4 COW unpack benchmark rows and refreshed complex `mul_assign`
  measurements from the local AVX2 host.

### Performance
- `SimdView::dot` middle SIMD loop now accumulates into the vector register via
  `fmadd` and reduces to scalar once at the end, instead of a horizontal
  `sum_reduce` per lane group. The per-group reduction serialized the loop on the
  ~5–7-cycle horizontal-reduction latency and dominated small/odd-length dot
  products (e.g. the length-`m−k` bidiagonal-SVD reflector applies in Leto). The
  unrolled head's vector accumulator now carries through the residual loop; only
  the final scalar tail reduces. Value-semantic (within the existing dot
  tolerance; 322 workspace tests green).

### Fixed
- `#[runtime_dispatch]` emitted `std::is_x86_feature_detected!` unconditionally, breaking `--no-default-features` builds; runtime-detection arms are now gated on the consuming crate's `std` feature (no_std keeps compile-time arms + scalar fallback).
- rkyv-exercising unit tests are ignored under Miri (rkyv 0.7 archived access violates Stacked Borrows inside the dependency); hermes's own unsafe passes Miri clean.
- INT4 unpack regression coverage now asserts the complete signed nibble domain.
- AMX context-pressure benchmarks no longer publish scalar fallback timings under an AMX-specific label on non-AMX hosts.
- Dense scalar benchmark baselines now black-box operands and accumulation so Criterion measures real iteration work instead of an optimized-away constant.
- Inline asm compute forms panic under Miri instead of returning fake values; Miri-valid AMX lifecycle operations are no-ops only for session-state verification.
- README now documents the Atlas SIMD/MIMD/GPU ownership boundary so consumers
  compose Hermes with Moirai and Hephaestus without duplicating responsibility.
- Packed4 COW unpacking delegates to the `Packable4` dispatcher, so the
  facade uses the existing AVX-512/AVX2/scalar runtime selection instead of an
  AVX-512-only x86 branch.
- README current-version metadata now reflects the released `0.2.0` workspace
  state.
- Added a Highway reference gap audit (`gap_audit.md`) and README baseline
  section, tracking Hermes-native follow-ups for target-token forced dispatch,
  safe slice wrappers, SSE2 feasibility, cross-target conformance, and
  operation-family coverage.
- `Cargo.lock` now matches the patched local Themis package version (`0.9.11`),
  keeping `cargo check --locked` coherent with the current Atlas checkout.

### Breaking
- Removed fixed Blocked-COO public dispatch functions `spmv_bcoo4x4` and
  `spmv_bcoo8x8`; use `spmv_bcoo::<T, BM, BN>`. Removed fixed
  `SparseView::from_blocked_coo_4x4` and `SparseView::from_blocked_coo_8x8`;
  use `SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo`.
- Removed fixed SELL-p public dispatch functions `spmv_sellp4` and
  `spmv_sellp8`; use `spmv_sellp::<T, C>`. Removed fixed
  `SparseView::from_sellp4` and `SparseView::from_sellp8`; use
  `SparseView::<T, SellP<C>, Arch>::from_sellp`.

## [0.2.0] — 2026-06-10

### Added
- Interleaved complex kernels (`interleaved_complex_dot`, `interleaved_complex_mul_assign`, runtime variants) over `[re, im, ...]` primitive slices, generic over `T: Scalar` and architecture with `const CONJ_B` conjugation (ADR-004).
- `SimdKernel` adjacent-pair primitives: `swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd` — default scalar emulation plus AVX2, AVX-512, and NEON intrinsic overrides.
- `CowFormat` trait and generic `SparseCow<'a, T, F, Arch>` clone-on-write sparse container.
- `SimdView::prefix_scan_in_place` (vectorized, single authoritative scan implementation).
- `SimdOps::interleaved_complex_mul_assign` / `interleaved_complex_dot` trait methods.
- Property-test suites: complex kernels with analytically derived rounding tolerances; differential AVX2/AVX-512-vs-Scalar tests on dyadic-exact inputs; `f16`/`bf16` complex differential tests (bitwise for elementwise multiply, reordering-bound for dot); kernel-level mask/compress/expand/gather/`leading_k_mask` invariants per backend.
- `complex_dot` example with throughput comparison.
- Exact NTT butterfly stage kernel (`dispatch/modular.rs`) with integration tests.
- CI pipeline (GitHub Actions): fmt, clippy `-D warnings`, tests on x86_64 and aarch64 (runtime NEON validation), warning-clean docs, aarch64 cross-compile check, `cargo-deny` supply-chain gate.
- `rust-toolchain.toml` pin (1.95.0) and workspace MSRV declaration (`rust-version = "1.95"`, verified by full build + test on 1.95.0).
- `deny.toml`: permissive-license allowlist, yanked-crate denial, source restrictions.
- PM artifacts: `backlog.md`, `checklist.md`, this changelog; README refreshed to the current architecture.

### Changed
- Complex kernel runtime dispatch unified onto `#[runtime_dispatch]` (replaces per-type `OnceLock` feature caching).
- Complex dot uses two independent accumulators (measured 47.3 → 31.7 ms on the example workload).
- Workspace-wide `cargo fmt` normalization; rustdoc builds warning-clean.

### Fixed
- `SimdCow::histogram_cow` computed bin indices through `f32` for every lane type, misbinning `f64` values near bin boundaries; indices now derive in `f64`.
- `SimdCow::prefix_scan_in_place` used a scalar loop; now delegates to the vectorized view-level scan.

### Breaking
- Removed `InterleavedComplexLane`; the runtime complex entry points now bound on `SimdOps` (same call syntax for `f32`/`f64`).
- Removed `CsrCow`, `SellPCow`, `BlockedCooCow`, `DenseWithMaskCow`; use `SparseCow<T, Csr | SellP<C> | BlockedCoo<BM, BN> | DenseWithMask, Arch>`. Constructors are unchanged per format (`borrowed`, `owned`, `from_vecs` via turbofish).

## [0.1.0]

Initial workspace: `SimdView` typestate views, `SimdKernel` trait with Scalar/AVX2/AVX-512/NEON backends, `#[runtime_dispatch]` macro, dense/masked/sparse (CSR, SELL-p, BCOO, Dense-with-Mask) kernels, `SimdCow`, precision ladder (`hermes-numeric`), Intel AMX + AVX-512 VNNI tile GEMM, SWAR chess bitboards, tiling, tensor views, criterion/divan benches.
