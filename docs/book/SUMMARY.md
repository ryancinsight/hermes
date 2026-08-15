# Summary

[Introduction](README.md)

# Part I — Dispatch Model

- [1. ISA Detection](isa_detection.md)
  - [Example: ISA Dispatch](examples/isa_dispatch.md)
- [2. Runtime vs. Compile-Time Dispatch](dispatch_model.md)
- [3. SimdArch and SIMD kernel facets](simd_arch.md)

# Part II — Core Operations

- [4. Horizontal Reductions](reductions.md)
  - [Example: Reductions](examples/reductions.md)
- [5. Dot Product and AXPY](dot_axpy.md)
- [6. Masked Operations](masked_ops.md)

# Part III — Data Structures

- [7. AlignedVec](aligned_vec.md)
- [8. SimdView and Tiling](simd_view.md)
  - [Example: Tiling](examples/tiling.md)
- [9. Sparse Formats](sparse.md)
  - [Example: Sparse](examples/sparse.md)

# Part IV — The Atlas Stack

- [10. Position in the Stack](stack_position.md)
- [11. Verification: How We Know the Kernels Are Right](verification.md)
