# Checklist — active sprint

**Target version: 0.2.0** · Strategy: [backlog.md](backlog.md) · Phase: Closure (release)

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

## Next sprint candidates (from [backlog](backlog.md))

- [minor] SpMV scalability sweep (P2).
- [minor] Packed4 AVX2 unpack path (P2).
- [minor] 0.3.0 release for the additive absolute-reduction API.
