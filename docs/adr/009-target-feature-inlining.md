# ADR 009: Monomorphized Target-Feature Gate Helpers for Zero-Overhead Inlining

## Status
Accepted

## Context
In Rust, `#[target_feature]` attributes restrict compiler inlining. A function marked with `#[target_feature(enable = "avx2")]` cannot be inlined into a function that lacks that attribute, because doing so could leak AVX2 instructions into context where they might execute on non-AVX2 hardware (causing `SIGILL`).

In the original dispatch design, generic `SimdView::sum()` was called directly from the public dynamic dispatch functions (`sum_f32` etc.). Because the public functions themselves do not have `#[target_feature]` attributes (since they are the entry point and must run on all CPUs), the compiler was forced to monomorphize `SimdView::sum` without AVX2/AVX-512 target features enabled. Consequently, calls to `Avx2::load_aligned`, `Avx2::add`, etc. (which are annotated with target features) could not be inlined into the loop body, introducing heavy function call overhead inside the critical inner loops and completely neutralizing SIMD throughput advantages.

## Decision
We introduced monomorphized local helper functions for each target architecture and annotated them with target features (e.g. `sum_f32_avx2` with `#[target_feature(enable = "avx2")]`). Inside these helpers, the view is constructed and `view.sum()` is called. 

The public dispatch functions first check compile-time features (via `cfg!(target_feature = "...")`), which has zero runtime cost. If compile-time features do not match, it checks runtime features (via `std::is_x86_feature_detected!`) and calls the corresponding helper function safely.

## Consequences
- **Perfect Loop Inlining**: Because the helper function has the target feature enabled, the compiler monomorphizes `view.sum()` with that feature enabled as well, permitting 100% inlining of `load_aligned`, `add`, and `fmadd`. The inner loop compiles down to a single contiguous block of vectorized instructions without any call boundaries.
- **Zero Runtime Check Overhead**: Under static compilation scenarios (e.g. compiling with `-C target-cpu=native`), the compile-time `cfg!` checks completely bypass runtime feature checking, translating to zero CPUID branching latency.
- **Safety Gating**: Target-specific instructions are strictly enclosed inside unsafe helper functions, preserving safe, panic-free execution boundaries for the public library API.
