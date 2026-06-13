# Backlog — hermes-simd

Strategic roadmap. Triage order: correctness → architecture → tests → docs → PM.
Tags: `[patch]` / `[minor]` / `[major]` / `[arch]` per SemVer change class.
Tactical breakdown of the active items lives in [checklist.md](checklist.md).

## Delivered (2026-06-11)

- [x] [minor] AXPY provider: `SimdOps::axpy` / dispatched `axpy` free fn —
  fused row update `out[i] += alpha * x[i]` via the `fmadd` primitive with
  scalar tail, no temporaries, length-mismatch error. Driver: leto matmul SIMD
  dispatch (its Stage C2 gate). Value tests across all tail sizes, f32/f64,
  zero-alpha identity, mismatch rejection.

## Atlas in-house replacement roadmap — hermes slice [arch]

hermes is the Atlas **SIMD SSOT** (data-parallel lanes), replacing std::simd / packed_simd
and hand-rolled intrinsics. Scope boundary: hermes owns SIMD only; thread-level **MIMD**
is moirai's domain, GPU is the `hephaestus` substrate (wgpu + CUDA) via coeus/apollo. Work to make hermes the
complete SIMD substrate for leto-ops/coeus hot kernels:
- [ ] [minor] Stage C1: dedicated AVX-512 / AMX CI runners (currently self-skip on
  unsupported hosts), `no_std` feature matrix, committed criterion baselines.
  Partial delivered (2026-06-12): local AVX2 Criterion baseline refreshed with
  packed4 COW unpack and unrolled complex `mul_assign` rows; runner self-check
  covered 48 rows. Dedicated AVX-512/AMX runners remain open.
- [x] [patch] Stage C1: `SveArch` callable fallback (stub removal) — delivered
  2026-06-13 as a value-preserving 512-bit-shape emulated backend for f32/f64.
- [x] [minor] Stage C1: `SveArch` public marker + property coverage —
  delivered 2026-06-13 by re-exporting it from `hermes-simd` and adding it to
  the host-independent kernel property suite.
- [ ] [minor] Stage C1: native SVE intrinsic backend for AArch64 server targets.
- [ ] [minor] Stage C2: expand op/dtype coverage on demand from leto-ops/coeus
  (gather/scatter variants, additional reductions/scans, complex precisions) so every
  leto/coeus CPU hot kernel has a hermes path rather than a scalar fallback.
  Delivered (2026-06-12): abs-sum (`Σ|x|`) and abs-max (`max|x|`) slice
  reductions via `AbsSum`/`AbsMax` ReductionOp ZSTs + `SimdOps::{abs_sum,
  abs_max}` dispatch. The reduce loop's unrolled head previously seeded
  accumulators with raw loads and merged partials with `accumulate` — correct
  only for transform-free ops; it now seeds through `transform_vector` and
  merges through `combine_vectors`, fixing the latent defect for every
  transform-bearing reduction (the documented SquaredSum hook included).
- [x] [patch] Document the SIMD(hermes) vs MIMD(moirai) vs GPU(hephaestus: wgpu + CUDA)
  boundary in README so consumers compose the three deliberately. Delivered
  2026-06-12: README defines Hermes as the synchronous, slice-oriented SIMD
  substrate; Moirai owns thread-level partitioning; Hephaestus owns GPU
  resource lifetimes and device-resident kernels.

## Stage assessment (2026-06-10)

Phase: **Execution → Closure transition for 0.2.0.** Canonical trait surfaces
(`Scalar`, `SimdKernel`, `SparseFormat`/`CowFormat`, op-strategy ZSTs,
`#[runtime_dispatch]`) are defined with one-or-more concrete implementations
each; 278 workspace tests green; clippy/doc/fmt gates clean; aarch64
cross-compile verified. The dominant remaining risks are *infrastructure*
(no CI, no toolchain pin, no changelog) and *unverified hardware paths*
(AVX-512, NEON, AMX run compile-checked but not runtime-validated locally).

## P0 — Release engineering for 0.2.0 <a id="p0"></a>

- [x] **[patch] CI pipeline** (delivered 0.2.0; AVX-512 runner still open) (highest risk reducer): GitHub Actions running
      fmt → clippy `-D warnings` → `cargo test --workspace` → doc build →
      `cargo check --target aarch64-unknown-linux-gnu`. Add an ARM runner
      (or QEMU) job to runtime-validate the NEON complex/dense paths, and an
      AVX-512-capable runner if available for the `Avx512` differential tests.
- [x] **[patch] Toolchain pin + supply chain** (delivered 0.2.0; cargo-audit covered by cargo-deny in CI): `rust-toolchain.toml`, declared
      MSRV, `cargo audit` + `cargo deny check` in CI.
- [x] **[minor] 0.2.0 release** (semver-checks scoped per crate; see checklist): CHANGELOG sections (Added/Changed/Breaking —
      includes `InterleavedComplexLane` removal and per-format sparse-Cow type
      removal), `cargo-semver-checks` run, version bump committed atomically.

## P1 — Correctness hardening <a id="p1"></a>

- [x] **[patch] Reduced-precision complex coverage** (delivered 0.2.0): property/differential
      tests for `f16`/`bf16` interleaved complex kernels (currently exercised
      only via emulated defaults, asserted only for f32/f64).
- [x] **[patch] Mask/gather/compress property suite** (delivered 0.2.0): proptest invariants —
      `compress`∘`expand` identity under fixed mask, `mask_to_bitmask` ∘
      `mask_from_bitmask` round-trip, gather with permuted indices vs scalar
      reference, `leading_k_mask` boundary cases (k=0, k=LANE_COUNT, k>LANE_COUNT).
- [x] **[patch] `cargo miri` pass** (delivered post-0.2.0: core unit tests green under Miri; rkyv 0.7 tests excluded as upstream Stacked Borrows violations; CI job added) over crates containing `unsafe`
      (intrinsics excluded where Miri lacks ISA support; cover the
      view/cow/sparse pointer logic in hermes-simd-core).
- [x] **[patch] no_std + feature matrix** (delivered post-0.2.0: runtime_dispatch std-gating fixed, --no-default-features green + CI step; broader feature-combination sweep remains open): verify `--no-default-features` and
      key feature combinations build and pass.

## P2 — Performance & memory <a id="p2"></a>

- [x] **[minor] Criterion regression thresholds** (delivered 2026-06-12):
      `benchmarks_baseline.json` records structured Criterion point estimates;
      `run-benches --check-regressions` fails on missing baseline rows or rows
      exceeding the configured ratio threshold. The runner is split into
      cohesive modules for CLI parsing, result discovery, host metadata,
      threshold comparison, and Markdown report rendering.
- [x] **[minor] SpMV scalability sweep** (delivered 2026-06-12): bench row
      counts ∈ {1K, 10K, 100K} at structural non-zero density {0.1%, 1%, 10%}
      across CSR/SELL-p/BCOO with 1024 columns; Dense-with-mask is capped at
      10K rows because it stores full dense values and masks. Sparse module
      docs now record format-selection guidance.
- [x] **[minor] Packed4 unpack generalization** (delivered 2026-06-12):
      `Packed4CowExt` delegates to `Packable4::unpack_slice_packed`, reusing
      the hermes-numeric AVX-512/AVX2/scalar dispatcher and removing the
      facade-local Bf4/F4 hardware-unpack impl pair. Criterion now includes a
      focused packed COW unpack benchmark.
- [x] **[minor] Complex mul_assign unroll** (delivered 2026-06-12):
      `interleaved_complex_mul_assign` processes two SIMD registers per loop
      iteration before the single-register and scalar tails. Criterion
      validation on this host showed runtime improvement across 256, 1K, 4K,
      and 16K complex-pair inputs.

## P3 — Architecture & maintenance <a id="p3"></a>

- [x] **[patch] x86 VNNI asm form** (delivered post-0.2.0): factor repeated
      `vpdpbssd` inline assembly into one internal instruction macro with
      explicit target-feature contract. The portable surface remains
      `TileMatrixMultiply`/runtime dispatch; asm is not promoted to a separate
      public abstraction.
- [ ] **[arch] Per-type x86 kernel dedup**: avx2_f32/avx2_f64/avx512_f32/
      avx512_f64 share method-body shape differing only in intrinsic suffix and
      lane count. Evaluate build-time generation (`build.rs` generator preferred
      over `macro_rules!`) — maintenance payoff only; requires ADR.
- [x] **[patch] SVE callable fallback**: removed `unimplemented!()` SVE
      `SimdKernel` methods and routed `SveArch` f32/f64 through the existing
      lane-emulated kernel macro with value-semantic tests.
- [x] **[minor] SVE property coverage**: `hermes-simd` re-exports `SveArch`,
      and `kernel_property_tests` now exercises its mask round-trip,
      compress/expand, gather, and leading-tail invariants on every host.
- [ ] **[minor] Native SVE backend**: hardware intrinsic implementation remains
      blocked on stable `core::arch::aarch64` SVE vector types; revisit on
      toolchain updates.
- [x] **[minor] NUMA module status** (audited 2026-06-11): `numa.rs` IS
      integrated — `hermes-simd::dispatcher` uses `NumaTopologyService`/
      `verify_numa_locality`, `vec` uses `NumaAllocator`, and types_tests
      cover node count/distance. Finding: it reimplements platform NUMA
      detection (`GetNumaHighestNodeNumber` on Windows, sysfs on Linux) that
      **themis `CpuTopology` owns**, and its `MnemosyneNumaAllocator` names
      mnemosyne's allocation responsibility — a structural duplication across
      the stack SSOT map (themis=topology law, mnemosyne=allocation).
- [x] **[patch] Default provider feature policy**: every Hermes package
      defaults `parallel` and `mnemosyne-memory`; the default
      `MnemosyneNumaAllocator` path now uses Mnemosyne allocation instead of a
      name-only std/platform allocator branch. The broader Themis topology
      replacement below remains open.
- [x] **[arch] NUMA consolidation onto themis/mnemosyne** (delivered
      2026-06-12): `numa.rs` detection now delegates to themis —
      `current_numa_node` → `themis::try_current_numa_node` (Option-honest,
      added in themis 0.7.0), `NumaTopologyService::{current_cpu,total_nodes,
      node_distance}` → `themis::current_processor` / process-cached
      `CpuTopology::detect()` distance tables. The duplicated libnuma /
      GetNumaHighestNodeNumber / sched_getcpu platform blocks are deleted.
      Allocation already routes through mnemosyne (`MnemosyneNumaAllocator`
      with `NumaBinding`). Kept in hermes by design: `NumaAllocator` trait,
      `NumaBinding` thread-affinity RAII, and `verify_numa_locality` —
      SIMD-specific concerns the topology SSOT should not own. Public query
      surface unchanged; dispatcher/vec/tests untouched.

## P4 — Documentation <a id="p4"></a>

- [x] **[patch] Doctest coverage**: `cargo doc` is warning-clean; extended
      runnable doctests to the complex, sparse-Cow, and tensor public surfaces.
- [x] **[patch] `#![deny(missing_docs)]`** (delivered post-0.2.0: all six public crates) on all public crates (currently
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
