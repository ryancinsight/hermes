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

1. ISA detection: `has_fma3()`, `FmaSupport`, `Avx512Support`.
2. Runtime vs. compile-time dispatch and the `SimdKernel` trait.
3. Core reductions: `sum`, `min`, `max`, `abs_sum`, `argmin`, `argmax`.
4. Dot product, AXPY, and FMA-accelerated kernels.
5. Masked execution: `masked_dot` and the `ExecutionMode` ZSTs.
6. `AlignedVec<T>` and aligned allocation via mnemosyne.
7. `SimdView` and tile-matrix multiplication.
8. Sparse formats: CSR, SELL-P, Blocked COO.
