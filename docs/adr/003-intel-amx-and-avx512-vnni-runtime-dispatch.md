# ADR 003: Intel AMX Session Cache and AVX-512 VNNI Runtime Dispatch on Stable Rust

## Status
Approved

## Context
High-performance machine learning (ML) and neural network inference workloads require high-throughput matrix multiplication engines. Intel AMX (Advanced Matrix Extensions) and AVX-512 VNNI (Vector Neural Network Instructions) provide hardware-accelerated GEMM operations. However, implementing these backends in a safe, generic, and stable Rust library introduces several architectural challenges:
1. **Compiler Channel Constraints**: Rust's standard library AMX intrinsics are unstable and require `#![feature(x86_amx_intrinsics)]`, which would force the workspace to build only on nightly compilers.
2. **AMX Configuration Latency**: The `ldtilecfg` instruction loaded to configure the AMX tile registers has a high latency cost. Loading it on every GEMM invocation degrades performance.
3. **Sub-byte Quantization**: Support for INT4 elements requires fast, bit-parallel unpacking to INT8 arrays before feeding the data into AMX (`tdpbssd`) or VNNI (`vpdpbssd`) execution units.
4. **Zero-Overhead Runtime Dispatch**: Path selection (AMX → AVX-512 VNNI → AVX-512 → Scalar fallback) must occur dynamically at runtime without adding pointer dereferences or branching overhead to the hot loop execution path.

## Decision
We implemented a self-contained, stable, and amortized hardware acceleration architecture:
- **Stable Inline Assembly**: We replaced all nightly intrinsics with stable `core::arch::asm!` blocks matching the AMX and VNNI instructions (`ldtilecfg`, `tilerelease`, `tilezero`, `tileloadd`, `tilestored`, `tdpbf16ps`, `tdpbssd`, and `vpdpbssd`). VNNI inline assembly is factored through one internal instruction macro so it expands inside the target-feature-gated tile loop; the portable contract remains the monomorphized `TileMatrixMultiply`/dispatch surface, not the assembly text.
- **RAII-based `AmxSession`**: Implemented a thread-local RAII guard `AmxSession` that tracks whether AMX is already configured on the current thread. Entering a compute phase is fallible: `AmxSession::new` and `AmxBatchSession::begin` return `AmxSessionError::UnsupportedTarget` before issuing `ldtilecfg` unless runtime support is confirmed. When a supported session goes out of scope, it automatically releases tile registers via `tilerelease`. Subsequent GEMM calls within the same session execute with zero configuration overhead.
- **Runtime Detection**: AVX-512 paths use stable `is_x86_feature_detected!` probes for the exact enabled target-feature sets. AMX dispatch remains disabled until a permission-aware probe verifies hardware bits, XCR0 tile state, and Linux XTILEDATA process permission; raw CPUID alone is not sufficient.
- **Quantization Helpers**: Implemented a bit-parallel `unpack_int4_to_int8` helper leveraging bitwise shifts to convert packed INT4 vectors into INT8 elements.
- **2x2 Register Blocking**: Designed the core AMX matrix multiplication loops using a 2x2 register blocking layout (loading 4 accumulator tiles simultaneously) to saturate processor FMA execution ports.

## Consequences
- **Stable Compiler Support**: The entire `hermes-simd` workspace compiles, checks, tests, and benchmarks successfully on stable Rust compilers.
- **Asm Scope Boundary**: Inline assembly is an implementation detail for unavailable or selected instruction contracts. New asm wrappers require target-feature gating, value-semantic scalar differential tests, and a measured or stability-driven reason over `core::arch` intrinsics.
- **Amortized Configuration Latency**: The configuration cost of AMX tile setup is paid once per thread phase, achieving near-theoretical hardware compute throughput on hot execution paths.
- **Safe Fallbacks**: Slices are dynamically matched, validated for dimensions, and routed to the most optimal instruction set extension available at runtime. If the CPU lacks AMX or AVX-512, dispatch falls back to AVX2 or Scalar loops. Safe vector/mask wrappers also check `SimdArch::is_runtime_supported()` before executing target-feature kernels.
