# Changelog

All notable changes to the hermes-simd workspace. Format: [Keep a Changelog]; versioning: SemVer 2.0.0 (pre-1.0: minor releases may break, documented under **Breaking**).

## [Unreleased] — targeting 0.2.0

### Added
- Interleaved complex kernels (`interleaved_complex_dot`, `interleaved_complex_mul_assign`, runtime variants) over `[re, im, ...]` primitive slices, generic over `T: Scalar` and architecture with `const CONJ_B` conjugation (ADR-004).
- `SimdKernel` adjacent-pair primitives: `swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd` — default scalar emulation plus AVX2, AVX-512, and NEON intrinsic overrides.
- `CowFormat` trait and generic `SparseCow<'a, T, F, Arch>` clone-on-write sparse container.
- `SimdView::prefix_scan_in_place` (vectorized, single authoritative scan implementation).
- `SimdOps::interleaved_complex_mul_assign` / `interleaved_complex_dot` trait methods.
- Property-test suites: complex kernels with analytically derived rounding tolerances; differential AVX2/AVX-512-vs-Scalar tests on dyadic-exact inputs.
- `complex_dot` example with throughput comparison.

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
