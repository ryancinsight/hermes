# ADR 006: SSE2 Backend Feasibility Study

## Status
Accepted

## Context

The initial Highway comparison audit ([gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#highway-2026-06-14)) identified that Hermes currently lacks an intermediate 128-bit x86_64 SIMD backend (SSE2/SSE4). On older x86_64 hardware, VMs, or conservative CI targets lacking AVX2 support, Hermes falls back to the portable `Scalar` backend. 

Adding a native SSE2 backend would bring 128-bit vector hardware acceleration to these environments. However, SSE2 lacks support for key features defined by the `BackendKernel<T>` seam and exposed through operation-family facets, including:
1. **Fused Multiply-Add (FMA)**: Only introduced in FMA3/AVX2.
2. **Gather/Scatter**: Only introduced in AVX2/AVX-512.
3. **Advanced Integer/Byte Shuffle**: `_mm_shuffle_epi8` requires SSSE3.
4. **Float Min/Max IEEE 754-conformance**: SSE2 `maxpd`/`minpd` do not fully conform to IEEE 754 propagation of NaNs.

We need to evaluate the feasibility, CI value, and maintenance cost of implementing an SSE2 backend.

## Evaluation

### 1. Trait Compatibility and Emulation Slop
Implementing `BackendKernel<T>` for SSE2 requires emulating missing hardware primitives:
- **FMA**: Must be emulated as `(a * b) + c`, paying double rounding penalties and instruction latency.
- **Gather**: Must be emulated via scalar pointer extraction and loading loops, yielding no performance benefit over the `Scalar` backend.
- **Adjacent-Pair Primitives**: Interleaved complex multiplication swaps and duplicates require complex shuffles that are slow in pure SSE2.

### 2. CI Value and Host Availability
GitHub Actions VM runners (standard `ubuntu-latest` x86_64) support up to AVX2. Therefore, our CI environment is already capable of executing and validating AVX2/AVX-512 (via emulation or supported subsets) and NEON. Adding SSE2 does not unlock any new target validation capabilities in CI.

### 3. Maintenance Cost
Adding an SSE2 backend introduces:
- A new set of vector wrappers (`Sse2F32Vec`, `Sse2F64Vec`, etc.) and mask layouts.
- Substantial code duplication across the `hermes-simd-intrinsics` crate.
- Added complexity in the runtime dispatch macro `#[runtime_dispatch]` and target feature detection.

## Decision

We decide **not to implement a native SSE2 backend** in Hermes. The maintenance cost and architectural complexity outweigh the performance benefits on legacy x86_64 hardware. 

Instead, we recommend:
1. **Compiler Auto-Vectorization**: Rely on the Rust compiler's LLVM backend to auto-vectorize the portable `Scalar` backend loops into SSE2 instructions when targeting older CPUs (via `-C target-feature=+sse2` or `-C target-cpu=x86-64`).
2. **Evaluate SSE4.1/SSSE3 as Baseline**: If an intermediate 128-bit x86_64 hardware backend becomes consumer-demand justified, we should implement it using **SSE4.1 + SSSE3** as the baseline rather than SSE2. SSE4.1 provides vector round, blend, dot product, and SSSE3 provides byte-level shuffles (`_mm_shuffle_epi8`), significantly reducing emulation slop.

## Consequences

- Legacy x86_64 hosts continue to execute via the portable `Scalar` fallback, which LLVM compiles to optimized SSE2/SSE3 instructions where appropriate.
- The `BackendKernel` and operation-family interfaces remain clean, avoiding the pollution of SSE2-specific emulation branches.
- The next step for 128-bit x86_64 targets remains defined as SSE4.1/SSSE3 rather than legacy SSE2.
