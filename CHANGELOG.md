# Changelog

All notable changes to the hermes-simd workspace. Format: [Keep a Changelog]; versioning: SemVer 2.0.0 (pre-1.0: minor releases may break, documented under **Breaking**).

## [Unreleased]

### Added
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
