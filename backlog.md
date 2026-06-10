# Backlog — hermes-simd

Strategic roadmap. Triage order: correctness → architecture → tests → docs → PM.
Tags: `[patch]` / `[minor]` / `[major]` / `[arch]` per SemVer change class.
Tactical breakdown of the active items lives in [checklist.md](checklist.md).

## Stage assessment (2026-06-10)

Phase: **Execution → Closure transition for 0.2.0.** Canonical trait surfaces
(`Scalar`, `SimdKernel`, `SparseFormat`/`CowFormat`, op-strategy ZSTs,
`#[runtime_dispatch]`) are defined with one-or-more concrete implementations
each; 278 workspace tests green; clippy/doc/fmt gates clean; aarch64
cross-compile verified. The dominant remaining risks are *infrastructure*
(no CI, no toolchain pin, no changelog) and *unverified hardware paths*
(AVX-512, NEON, AMX run compile-checked but not runtime-validated locally).

## P0 — Release engineering for 0.2.0 <a id="p0"></a>

- [ ] **[patch] CI pipeline** (highest risk reducer): GitHub Actions running
      fmt → clippy `-D warnings` → `cargo test --workspace` → doc build →
      `cargo check --target aarch64-unknown-linux-gnu`. Add an ARM runner
      (or QEMU) job to runtime-validate the NEON complex/dense paths, and an
      AVX-512-capable runner if available for the `Avx512` differential tests.
- [ ] **[patch] Toolchain pin + supply chain**: `rust-toolchain.toml`, declared
      MSRV, `cargo audit` + `cargo deny check` in CI.
- [ ] **[minor] 0.2.0 release**: CHANGELOG sections (Added/Changed/Breaking —
      includes `InterleavedComplexLane` removal and per-format sparse-Cow type
      removal), `cargo-semver-checks` run, version bump committed atomically.

## P1 — Correctness hardening <a id="p1"></a>

- [ ] **[patch] Reduced-precision complex coverage**: property/differential
      tests for `f16`/`bf16` interleaved complex kernels (currently exercised
      only via emulated defaults, asserted only for f32/f64).
- [ ] **[patch] Mask/gather/compress property suite**: proptest invariants —
      `compress`∘`expand` identity under fixed mask, `mask_to_bitmask` ∘
      `mask_from_bitmask` round-trip, gather with permuted indices vs scalar
      reference, `leading_k_mask` boundary cases (k=0, k=LANE_COUNT, k>LANE_COUNT).
- [ ] **[patch] `cargo miri` pass** over crates containing `unsafe`
      (intrinsics excluded where Miri lacks ISA support; cover the
      view/cow/sparse pointer logic in hermes-simd-core).
- [ ] **[patch] no_std + feature matrix**: verify `--no-default-features` and
      key feature combinations build and pass.

## P2 — Performance & memory <a id="p2"></a>

- [ ] **[minor] Criterion regression thresholds**: record baseline numbers for
      sum/dot/spmv/complex; a statistically significant slowdown blocks merge.
- [ ] **[minor] SpMV scalability sweep**: bench n ∈ {1K, 10K, 100K} rows ×
      sparsity {0.1%, 1%, 10%} across CSR/SELL-p/BCOO to characterize
      format-selection crossover points; document guidance in sparse module docs.
- [ ] **[minor] Packed4 unpack generalization**: `HardwareUnpack` currently
      routes x86_64 through the AVX-512 tiling path only; add an AVX2 unpack
      and runtime selection, and fold the Bf4/F4 impl pair into the dispatch
      abstraction if a third packed format lands.
- [ ] **[minor] Complex mul_assign unroll**: evaluate 2× unroll (as done for
      dot) once store-bound profile is confirmed; criterion data first.

## P3 — Architecture & maintenance <a id="p3"></a>

- [ ] **[arch] Per-type x86 kernel dedup**: avx2_f32/avx2_f64/avx512_f32/
      avx512_f64 share method-body shape differing only in intrinsic suffix and
      lane count. Evaluate build-time generation (`build.rs` generator preferred
      over `macro_rules!`) — maintenance payoff only; requires ADR.
- [ ] **[minor] SVE backend**: real implementations blocked on stable
      `core::arch::aarch64` SVE intrinsics; revisit on toolchain updates
      (stub documented in `aarch64/sve.rs`).
- [ ] **[minor] NUMA module status**: `hermes-simd-core/src/numa.rs` — audit
      for completeness/usage; either integrate with benches or document scope.

## P4 — Documentation <a id="p4"></a>

- [ ] **[patch] Doctest coverage**: `cargo doc` is warning-clean; extend
      runnable doctests to the complex, sparse-Cow, and tensor public surfaces.
- [ ] **[patch] `#![deny(missing_docs)]`** on all public crates (currently
      `warn` in hermes-simd-core).

## Completed (recent)

- [x] [minor] Generic vectorized interleaved complex kernels + runtime dispatch
      (ADR-004; commits 33ce1b8, 3aa963e).
- [x] [minor] NEON adjacent-pair primitive overrides, aarch64 compile-verified (3aa963e).
- [x] [arch] Sparse Cow consolidation → generic `SparseCow<T, F, Arch>` + `CowFormat` (3aa963e).
- [x] [patch] Native-precision histogram binning fix + regression test (8b4a796).
- [x] [patch] Vectorized in-place prefix scan, single authoritative impl (8b4a796).
- [x] [patch] Complex-kernel property tests with analytical tolerances (8b4a796).
- [x] [patch] Workspace fmt normalization; rustdoc warning cleanup (fc34e6a, 3aa963e).
