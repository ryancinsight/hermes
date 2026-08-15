# hermes-simd — SIMD/ISA Dispatch for Atlas

`hermes-simd` is the SIMD/ISA dispatch layer of the Atlas compute stack.  It
sits above `hermes` (SIMD/ISA dispatch) and below `leto` (host arrays/CPU
linear algebra) in the Atlas layer hierarchy.

## Design goals

- **Runtime ISA selection** — kernel selection happens once at call time via
  a cached `OnceLock` probe; the selected path stays in the L1 instruction
  cache for subsequent calls.
- **Zero-overhead abstraction** — `SimdView<T, Arch, Align>` is a typed
  wrapper around a `&[T]`; the arch and alignment parameters are ZSTs that
  vanish after monomorphization.
- **All-path coverage** — every operation has a `Scalar` fallback that
  compiles on every platform; ISA-specific paths (`Avx2`, `Avx512`, `Neon`,
  `SveArch`) layer over it when the CPU reports support.

## What this book covers

**Part I — Dispatch Model**

1. ISA detection and the runtime capability probes.
2. Runtime vs. compile-time dispatch and the scalar floor.
3. The `SimdArch` / operation-facet seams and their backends.

**Part II — Core Operations**

4. Horizontal reductions: `sum`, `min`, `max`, `abs_sum`, `argmin`, `argmax`.
5. Dot product, AXPY, and the register-blocked GEMV/GEMM family.
6. Masked execution and the `ExecutionMode` typestates.

**Part III — Data Structures**

7. `AlignedVec<T, Align>` and aligned, NUMA-aware allocation.
8. `SimdView` and register-blocked tiling.
9. Sparse formats: CSR, SELL-P, Blocked COO, Dense-with-mask.

**Part IV — The Atlas Stack**

10. `hermes`'s position in the Atlas provider graph.
11. The verification ladder: differential, property, and coverage tests.
