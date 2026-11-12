# Milestone ladder ([ADR 026](../docs/adr/026-release-scope-1-0.md))
<a id="M-core-kernels"></a> **M-core-kernels** (0.10.0) needs: (none) — scalar fallbacks and core kernels on the admitted provider cohort.
- exit: pass core kernel tests (`cargo test -p hermes-simd-core --locked --all-features` passes).
- exit: pass facade kernel tests (`cargo test -p hermes-simd --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-backends-isa"></a> **M-backends-isa** (0.11.0) needs: M-core-kernels — ISA backends, prefetch, and permutes.
- exit: pass intrinsics backend tests (`cargo test -p hermes-simd-intrinsics --locked --all-features` passes).
- exit: pass types and macros tests (`cargo test -p hermes-simd-types --locked --all-features` and `cargo test -p hermes-simd-macros --locked --all-features` pass).
- standing: no open correctness item.
<a id="M-1.0"></a> **M-1.0** (1.0.0) needs: M-backends-isa — stable kernel release (authority-gated).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.
