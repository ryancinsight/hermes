# Checklist — active sprint

**Target version: 0.2.0** · Strategy: [backlog.md](backlog.md) · Gap register: [gap_audit.md](gap_audit.md) · Phase: Closure (release)

## Sprint scope: ship 0.2.0 with CI

- [x] [patch] `.github/workflows/ci.yml`: fmt-check, clippy `-D warnings`,
      `cargo test --workspace` (x86_64 + native aarch64 runner), warning-clean
      docs, aarch64 cross-check, cargo-deny. → green (run 27296233212; required three fixes: libnuma feature gating, license inheritance, AMX bench import cfg).
- [x] [patch] `rust-toolchain.toml` (1.95.0) + `rust-version = "1.95"` in the
      workspace manifest, inherited by all members. MSRV verified empirically:
      full workspace build + 295-test pass on rustc 1.95.0.
- [x] [patch] `deny.toml` (advisories/licenses/bans/sources) + CI job.
      Local `cargo audit` blocked by an outdated local advisory parser
      (CVSS 4.0); CI cargo-deny is the authoritative gate.
- [x] [patch] `f16`/`bf16` interleaved-complex differential tests — bitwise
      equality for elementwise multiply (lane-emulated backends share the op
      sequence); dot compared under the analytical reordering bound
      `(n+8)·ε_T·Σ magnitudes`.
- [x] [patch] Kernel property suite (`kernel_property_tests.rs`): bitmask
      round-trip, compress∘expand identity, gather vs scalar reference,
      `leading_k_mask` boundaries — per backend, feature-gated.
- [x] [minor] Version bump 0.1.0 → 0.2.0; CHANGELOG 0.2.0 section dated;
      `cargo-semver-checks` pass on hermes-simd and hermes-simd-core vs the
      previous rev (196 checks each, no regression).
- [x] [minor] Tag `v0.2.0` — created after CI run 27296233212 (all four jobs green, including runtime NEON validation on a native aarch64 runner).

## Residual risks

- AVX-512 and AMX paths: differential tests self-skip on unsupported hosts;
  no AVX-512 CI runner yet ([backlog → P0](backlog.md#p0)).
- `cargo-semver-checks --workspace` cannot doc-build `hermes-numeric` under
  its feature-combination probing (rkyv `size_*` feature requirement);
  per-crate scoped runs are the working procedure.
- `panic = "abort"` in the release profile: CI tests the dev profile.
- Full Themis topology consolidation remains open: Hermes still owns the
  public `NumaTopologyService` facade and platform binding fallback, while
  default NUMA-vector allocations now route through Mnemosyne.

## Post-0.2.0 increment (2026-06-10)

- [x] [patch] `cargo miri` over hermes-simd-core: unit tests green; rkyv 0.7
      tests `#[cfg_attr(miri, ignore)]` (upstream Stacked Borrows violations);
      CI `miri` job added.
- [x] [patch] no_std: `#[runtime_dispatch]` std-gating fixed;
      `--no-default-features` check green locally and in CI.
- [x] [patch] `#![deny(missing_docs)]` on all six public crates; 12 items documented.
- [x] [minor] Complex criterion bench suite + recorded baselines
      (benchmarks_results.md); threshold automation delivered below.
- [x] [patch] x86 VNNI asm cleanup: `vpdpbssd` factored into one internal
      asm macro with `nostack`/`nomem`/`preserves_flags`; both AVX-512 tile
      kernels expand it inside the target-feature-gated loop. Added complete
      signed-nibble INT4 unpack regression coverage and documented the asm
      scope boundary.
- [x] [patch] local-capable test/bench hardening: host dispatch tests cover
      the locally detected dense backend, AVX2 direct execution when present,
      and irregular INT8 GEMM against scalar reference. Benchmark report
      generation records detected ISA features and suppresses AMX
      context-pressure rows on non-AMX hosts; dense scalar baselines now
      black-box operands/accumulation to prevent optimized-away work.
- [x] [patch] Miri intrinsics boundary: VNNI/AMX compute asm panics under
      Miri instead of returning synthetic values; AMX configuration/release
      instructions are no-ops only for session-state tests; CI now runs
      `cargo +nightly miri test -p hermes-simd-intrinsics`.
- [x] [patch] runnable doctest coverage: enabled doctests for
      `hermes-simd-core` and added value-semantic examples for complex
      multiplication/dot, sparse CSR `SparseCow` SpMV, and const-generic
      `TensorView`.
- [x] [patch] SVE callable fallback: `SveArch` f32/f64 now use the
      monomorphized lane-emulated kernel macro (`16xf32`, `8xf64`) with
      value-semantic tests. Native SVE intrinsics remain tracked separately.
- [x] [minor] SVE property coverage: `hermes-simd` re-exports `SveArch`, and
      the shared kernel property suite now runs its mask/compress/expand,
      gather, and leading-tail invariants on every host.
- [x] [patch] runnable core doctests: kernel, compute, and tiling examples now
      execute value assertions under `cargo test --doc --workspace` instead of
      compile-only `no_run`.
- [x] [patch] runnable BitMask doctests: native-mask conversion and
      active-lane iteration examples now execute value assertions.
- [x] [patch] Default provider features: every Hermes package now defaults
      `parallel` and `mnemosyne-memory`; `hermes-simd-core` pins Mnemosyne
      `938d0c2` and routes `AlignedVec::with_capacity_numa` allocation and
      deallocation through `mnemosyne::Mnemosyne` under the default feature.
      Verification: fmt, clippy all-targets/all-features, workspace tests,
      warning-clean docs, and no-default-features check.
- [x] [minor] Absolute reductions: `AbsSum` / `AbsMax` and dispatched
      `abs_sum` / `abs_max` provide Hermes-owned L1 and infinity norm
      accumulators for Leto/Apollo consumers without temporary buffers.
      Evidence tier: value-semantic tests plus full workspace gate (`fmt`,
      `check`, `test`, `clippy -D warnings`, docs).
- [x] [minor] Criterion threshold automation: `run-benches` now writes
      `benchmarks_baseline.json`, enforces baseline rows with
      `--check-regressions`, and is split into SRP modules (`cli`,
      `criterion_results`, `host`, `regression`, `report`) instead of a
      542-line mixed-concern entrypoint. Evidence tier: value-semantic unit
      tests for CLI/regression/report parsing, local dense Criterion run, and
      baseline self-check over 36 rows.
- [x] [minor] SpMV scalability sweep: sparse Criterion bench now covers
      CSR/SELL-p/BCOO over 1K, 10K, and 100K rows at 0.1%, 1%, and 10%
      structural non-zero density with bounded Dense-with-mask cases through
      10K rows. Sparse module docs now state format-selection guidance.
- [x] [patch] Atlas compute boundary docs: README states Hermes owns SIMD
      lane-parallel kernels and slice-oriented dispatch, Moirai owns MIMD
      scheduling, and Hephaestus owns GPU/device lifetimes.
- [x] [minor] Packed4 unpack generalization: `Packed4CowExt` now calls the
      canonical `Packable4` packed dispatcher, so the facade inherits AVX-512,
      AVX2, and scalar runtime selection without a duplicate x86 branch.
      Coverage: odd-length full-nibble COW unpack regression plus a focused
      Criterion benchmark target.
- [x] [minor] Complex `mul_assign` unroll: in-place interleaved complex
      multiply now processes two SIMD registers per loop iteration before the
      single-register and scalar tails, with a direct four-pair scalar loop for
      large scalar buffers. Evidence: focused Criterion runs plus refreshed
      48-row local AVX2 baseline.
- [x] [patch] README current-version metadata corrected from `0.1.0` to
      `0.2.0`.
- [x] [patch] Benchmark baseline refresh: `run-benches --parse-only
      --write-baseline --check-regressions` regenerated
      `benchmarks_baseline.json` and `benchmarks_results.md` from local
      Criterion output, including packed4 COW unpack and the unrolled complex
      `mul_assign` rows. Regression self-check covered 48 rows.
- [x] [minor] Const-generic Blocked-COO dispatch: removed fixed public
      `spmv_bcoo4x4`/`spmv_bcoo8x8` dispatch functions and fixed
      `SparseView::from_blocked_coo_4x4`/`from_blocked_coo_8x8` constructors
      in favor of one `spmv_bcoo::<T, BM, BN>` API and the existing generic
      `from_blocked_coo` constructor. Evidence tier: type-level const-generic
      shape encoding plus value-semantic sparse tests and benchmark parser
      regression self-check.
- [x] [minor] Const-generic SELL-p dispatch: removed fixed public
      `spmv_sellp4`/`spmv_sellp8` dispatch functions and fixed
      `SparseView::from_sellp4`/`from_sellp8` constructors in favor of one
      `spmv_sellp::<T, C>` API and the existing generic `from_sellp`
      constructor. Evidence tier: type-level const-generic slice-height
      encoding plus value-semantic sparse tests and benchmark parser
      regression self-check.
- [x] [patch] Highway comparison audit: audited
      `https://github.com/NikoMalik/highway.git` at
      `0984271e74db124cf5e200de542e745348eb0b9e` and recorded Hermes-native
      follow-ups in `gap_audit.md`, `backlog.md`, and README. Evidence tier:
      source audit plus local code search.

## Next sprint focus (from [gap_audit](gap_audit.md#highway-2026-06-14))

- [x] [minor] Target-token forced dispatch: define the Hermes-native
      `TargetId`/forced-dispatch test surface and prove unsupported targets
      reject safely. Evidence tier: type-level architecture view construction
      plus value-semantic host capability tests.
- [x] [minor] Safe one-vector slice wrappers: add bounds/alignment-checked
      wrappers over `SimdKernel` load/store primitives and value-semantic
      tests for success and failure paths. Evidence tier: value-semantic
      integration tests over public `Vector<T, Arch>` methods.
- [x] [minor] Public dense cross-target matrix: compare public facade outputs
      against Scalar across every host-supported target. Evidence tier:
      value-semantic differential tests over forced `TargetId` views.
- [x] [minor] Batched AXPY rows: add `axpy_rows_batch` to the sealed
      `SimdOps` facade and runtime-dispatch it through the existing AXPY
      kernel family. Evidence tier: value-semantic differential test against
      repeated `axpy_rows`, exact invalid-extent error assertions, and Miri
      coverage of the unsafe pointer loop. Memory model: each output lane is
      loaded once, accumulated across depth in registers, and stored once.
      Benchmark coverage: `axpy_rows_batch_f32` compares the fused path with
      repeated public `axpy_rows` calls.
- [x] [patch] Dense/AXPY error-contract hardening: length-mismatch tests now
      assert exact `SimdError::LengthMismatch` values instead of only
      asserting that an error exists.
- [x] [patch] Select/unary error-contract hardening: select, unary-map, and
      COW FMA tests now assert exact `SimdError` variants for length mismatch
      and insufficient output capacity.
- [x] [patch] Operation-family error-contract hardening: new operation,
      strategy, complex, gather, scan, and COW math tests now assert exact
      `SimdError` variants for invalid shape, short output, and invalid index
      cases.

## Next sprint candidates (from [backlog](backlog.md))

- [minor] 0.3.0 release for the additive absolute-reduction API.
- [arch] Per-type x86 kernel dedup ADR (P3).
