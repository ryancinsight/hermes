# hermes-simd

A high-performance, zero-overhead Rust SIMD abstraction library focusing on data-parallel computing, Intel AMX tiling, AVX-512 VNNI, SWAR Chess Bitboards, and Sparse SpMV kernels.

The workspace is designed for extreme runtime efficiency, using traits, ZST markers, and full compiler monomorphization to generate machine code identical to hand-optimized assembly. It compiles entirely on stable Rust with no unstable nightly compiler prerequisites.

## Workspace Structure

The project is structured as a multi-crate workspace:
- **`crates/hermes-simd-core`**: Core abstractions, align typestates, reference type-state parameterization (`SimdView`), mask wrappers (`BitMask`), execution modes (`Unmasked`/`Masked`), and unified `ComputeView` trait.
- **`crates/hermes-simd-intrinsics`**: Low-level, architecture-specific vector kernels (Scalar, AVX2, AVX-512, NEON), Intel AMX engine, AVX-512 VNNI tile multipliers, and sliding attack bitboards.
- **`crates/hermes-simd-macros`**: Procedural macros (`#[runtime_dispatch]`, `#[derive(SparseData)]`) for compile-time generation of dispatch boilerplate.
- **`crates/hermes-simd`**: The public safe API facade that handles dynamic CPUID runtime dispatch and safe client interactions.
- **`crates/hermes-simd-examples`**: Demo applications showing bitboard computations, dot products, and copy-on-write SIMD utilities.
- **`crates/hermes-simd-benches`**: Matrix benching suite with Criterion and a custom parser to compile results.

## Key Features

1. **Intel AMX Acceleration**:
   - Self-contained, stable inline assembly support for AMX registers and instructions (`tdpbf16ps`, `tdpbssd`).
   - RAII `AmxSession` cache manager to load tile configurations once and amortize setup latency.
   - 2x2 register blocking for high-throughput matrix multiply kernels.
2. **AVX-512 & VNNI Optimization**:
   - VNNI tile matrix multiplication (`gemm_int8`) utilizing bit-parallel unpacking of sub-byte INT4 elements to INT8.
   - Vector mask registers (`__mmask16`/`__mmask8`) mapped to branchless conditional logic.
3. **SWAR Chess Bitboards**:
   - Sliding Rook/Bishop attack generator backends: Kogge-Stone (parallel direction vectorization), Hyperbola Quintessence, Fancy Magic Bitboards, and Hybrid SWAR-Magic.
   - Batch attack queries using a unified `BitBoardView`.
   - Pure SWAR bit utilities (byte-wise popcounts, bit scans, MSB/LSB isolation).
4. **Sparse SIMD (SpMV)**:
   - Format-parameterized views for CSR, Sliced ELLPACK (SELL-p), Blocked COO, and Dense-with-Mask layouts.
5. **Type-State Reference Parameterization**:
   - Unified `SimdView<'a, T, Arch, Align, Ref>` where `Ref` can be `&'a [T]` or `&'a mut [T]`. Enforces covariance/invariance and aliasing safety at compile time with zero runtime layout overhead.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Enables runtime CPU feature detection |
| `sparse` | Enables `SparseView` SpMV layouts and computation |
| `tiling` | Enables register-blocked tiling dot products and GEMV |
| `bytemuck` | Enables safe type-casting via the `bytemuck` crate |
| `wide` | Enables the `wide` crate backend fallback |
| `portable-simd` | Enables nightly standard library `std::simd` |

---

## Quickstart

### Dense Sum Reduction (Dynamic Dispatch)
```rust
use hermes_simd::sum;

let data = vec![1.0f32; 1024];
let result = sum(&data);
assert_eq!(result, 1024.0);
```

### Masked Dot Product
```rust
use hermes_simd::masked_dot;

let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
let b = vec![1.0f32; 5];
let mask = vec![true, false, true, false, true];

let result = masked_dot(&a, &b, &mask).unwrap(); // computes 1*1 + 3*1 + 5*1
assert_eq!(result, 9.0);
```

### High-level GEMM (AMX / VNNI Fallback)
```rust
use hermes_simd::gemm_int8;

let m = 32;
let n = 32;
let k = 64;
let a = vec![1i8; m * k];
let b = vec![2i8; k * n];
let mut c = vec![0i32; m * n];

// Automatically dispatches to Intel AMX (if available), AVX-512 VNNI, or Scalar loops
unsafe {
    gemm_int8(m, n, k, &a, k, &b, n, &mut c, n).unwrap();
}
```

---

## Running Verification

### Automated Tests
Run the unit, integration, and property tests across the workspace:
```powershell
cargo test --workspace
```

### Examples
Run the SWAR bitboard simulator:
```powershell
cargo run -p hermes-simd-examples --example swar_bitboards
```

### Benchmarks
Generate the performance overview report:
```powershell
# Compiles benchmarks
cargo check --benches

# Runs benchmarks and updates benchmarks_results.md
cargo run -p hermes-simd-benches
```
