# ADR 007: Arm SME Backend Feasibility Study

## Context

The NumKong comparison audit ([gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17)) identified that modern AArch64 systems (including Apple M4/M5 and Arm Neoverse-V2/V3 server platforms) implement the **Scalable Matrix Extension (SME)**. SME provides hardware-accelerated outer-product operations and a dedicated matrix accumulator tile array (`ZA`) to accelerate matrix multiplication (GEMM) kernels.

As the Single Source of Truth (SSOT) for data-parallel lanes in the Atlas stack, Hermes needs to evaluate the architectural feasibility, compile-time/runtime type constraints, context-switching overheads, and toolchain stability of integrating a native AArch64 SME backend.

## Evaluation

### 1. Toolchain Stability and Type-System Constraints
The AArch64 SME architecture operates under unique CPU regimes:
- **Streaming SVE Mode (`PSTATE.SM`)**: The processor shifts from standard NEON/SVE execution to a streaming regime. This shift changes the active vector length from standard Vector Length (`VL`) to Streaming Vector Length (`SVL`).
- **Stable Toolchain Constraints**: Stable Rust (rustc 1.95.0) lacks compiler intrinsics and vector representation types for AArch64 SVE and SME. There are no stable target feature markers (e.g., `#[target_feature(enable = "sme")]`) or built-in ACLE (Arm C Language Extensions) mappings.
- **Nightly Status**: SME support in Rust is in early design and experimental phases, requiring nightly features for inline assembly register constraints (`q` registers) and ABI attributes.

### 2. State Management & ABI Context Switching
SME introduces significant runtime complexity regarding registers:
- Entering streaming mode via `SMSTART SM` or `SMSTART ZA` has a non-negligible latency penalty.
- The `ZA` accumulator tile storage has a variable size of $(\text{SVL}/8) \times (\text{SVL}/8)$ bytes. Transitioning between standard execution and streaming mode clears or corrupts standard SVE register files (`Z0-Z31`), necessitating context save/restore guards.
- Standard calling conventions (AAPCS64) do not natively preserve the `ZA` state across standard library function boundaries. Custom assembly wrappers or specialized compiler ABI bindings are required to ensure safety.

### 3. Compute Dispatch & Algorithmic Mapping
SME is optimized for tiled matrix multiplication using outer products:
- Operations like `FMOPA` (Floating-point Outer Product Accumulate) take SVE vector registers and accumulate the outer product directly into the `ZA` tile.
- While highly efficient for dense GEMM/GEMV, this outer-product tiled accumulator register model does not map cleanly onto the standard 1D vector abstraction defined by `SimdKernel<T>` (e.g., elementwise lane ops, horizontal reductions).
- Attempting to force the `ZA` tile storage model into `SimdKernel<T>` requires extensive emulation and context swaps, which negates the performance benefit.

## Decision

We decide **to defer the implementation of a native Arm SME backend in Hermes** and handle it via the following phased approach:

1. **Defer Native Intrinsic Backend**: We will not introduce inline assembly SME kernels or custom ABI context-switching wrappers on the current stable toolchain.
2. **Expose CPU Feature Probes**: Expose SME target-feature check capability in the runtime dispatcher once AArch64 architecture detection is fully supported.
3. **GEMM Tiling Interface Alignment**: Since SME's execution model is intrinsically tiled, any future native SME backend should bypass the 1D `SimdKernel` abstraction and bind directly to the `TileMatrixMultiply` trait seam, where the `ZA` matrix accumulator state can be mapped to a persistent tile context.

## Consequences

- SVE/SME emulation remains mapped under `SveArch` for portable compile-checking, avoiding nightly compiler requirements.
- The AArch64 target feature footprint remains safe, stable-only, and warning-free.
- Core 1D SIMD operations remain performant without the context-switch latency overhead of entering and exiting `PSTATE.SM`.
