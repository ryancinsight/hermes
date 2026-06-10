# Checklist — active sprint

**Target version: 0.2.0** · Strategy: [backlog.md](backlog.md) · Phase: Closure (release engineering)

Each item lists its observable completion condition.

## Sprint scope: ship 0.2.0 with CI

- [ ] [patch] Add `.github/workflows/ci.yml`
      → green run on push: fmt-check, clippy `-D warnings`, `cargo test --workspace`,
        `cargo doc --no-deps` warning-clean, aarch64 cross-check. ([backlog → P0](backlog.md#p0))
- [ ] [patch] Add `rust-toolchain.toml` + MSRV in workspace manifest
      → `cargo +<pinned> test --workspace` green; MSRV documented in README.
- [ ] [patch] Add `cargo audit` / `cargo deny` config and CI step
      → both report no violations or each exception is documented in `deny.toml`.
- [ ] [patch] f16/bf16 interleaved-complex differential tests
      → property tests vs scalar reference with analytically derived tolerance
        pass for both types, both conjugation variants. ([backlog → P1](backlog.md#p1))
- [ ] [patch] Mask/gather/compress proptest suite
      → round-trip and reference invariants pass on all compiled backends.
- [ ] [minor] CHANGELOG 0.2.0 section + `cargo-semver-checks` + version bump
      → Cargo.toml `0.2.0` == checklist target; Breaking subsection lists
        `InterleavedComplexLane` and per-format sparse-Cow type removals;
        tag `v0.2.0` created only after all items above are green.

## Done this cycle (2026-06-10)

- [x] README updated to current architecture (crates, complex kernels, Cow
      containers, verification policy, PM artifact links).
- [x] backlog.md / checklist.md / CHANGELOG.md initialized.

## Residual risks

- AVX-512 and AMX paths: compile-verified, differential tests self-skip on
  unsupported hosts — runtime validation requires capable CI hardware (P0).
- NEON: compile-verified for aarch64-unknown-linux-gnu; no runtime run yet.
- `panic = "abort"` in release profile: incompatible with `cargo test --release`
  harness on some setups; CI should test the dev profile or override.
