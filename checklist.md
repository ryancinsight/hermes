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

## Next sprint candidates (from [backlog](backlog.md))

- [patch] `cargo miri` over hermes-simd-core pointer logic (P1).
- [patch] no_std / feature-matrix CI job (P1).
- [minor] Criterion baselines + regression thresholds (P2).
- [minor] SpMV scalability sweep (P2).
