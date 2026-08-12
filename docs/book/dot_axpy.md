# 5. Dot Product and AXPY

The dot product and the AXPY family are the workhorses of dense linear
algebra. hermes-simd provides both as runtime-dispatched slice functions, and
both share the two defining properties of the crate's dense kernels: **fused
arithmetic** (the multiply–add collapses into one FMA instruction where the
host has it) and **no temporaries** (every kernel writes straight through to
its output).

## Dot product

```rust,ignore
use hermes_simd::dot;

let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
let ones: Vec<f32> = vec![1.0; 1024];
let d = dot::<f32>(&a, &ones).expect("equal lengths");
```

`dot` computes `Σ aᵢ·bᵢ` and returns `Result<T, SimdError>`: unequal lengths
are a caller-visible failure and are reported as
`SimdError::LengthMismatch`, never silently truncated. The empty slice has a
well-defined identity (`T::ZERO`).

Internally the accumulation is FMA-accelerated. The `Dot` reduction strategy
overrides the default `mul`+`add` pair with `fma_pair_accumulate`, which folds
the lane product into the accumulator in a single `fmadd(a, b, acc)`
instruction with one rounding. On an FMA3-capable host the kernel is
`vfmadd231ps`; on hosts without FMA the same generic body compiles to the
separate multiply and add. The final ragged tail is routed through the
provider's masked-FMA seam so the masked-memory contract stays valid on every
backend and only live lanes contribute to the final reduction.

Floating-point dot, like floating-point sum (Chapter 4), reorders its
reduction across SIMD lanes. Callers comparing against a sequential reference
assert within the documented reassociation envelope — never bitwise equality.

## AXPY: fused scaled row updates

```rust,ignore
use hermes_simd::axpy;

let mut out: Vec<f32> = vec![1.0; 4];
axpy(2.0_f32, &[3.0, 1.0, -2.0, 5.0], &mut out).expect("equal lengths");
// out[i] = 1.0 + 2.0·x[i]   =>   [7.0, 3.0, -3.0, 11.0]
```

- `axpy(alpha, x, out)` — `out[i] += alpha·x[i]`, one fused update with no
  temporary allocation.
- `axpy_mul(alpha, a, b, out)` — `out[i] += alpha·a[i]·b[i]`, the fused ternary
  form for computing a scaled product into an accumulator without a
  temporary product vector (used, for example, for residual accumulation
  `c += multiplier·a·b`).
- `axpy_rows(alphas, x, out, row_stride, rows, cols)` — the multi-row form
  `out[row, i] += alphas[row]·x[i]` over a row-major strided output window.
- `axpy_rows_batch(alphas, x_panel, out, row_stride, rows, depth, cols)` —
  the depth-major batched accumulation
  `out[row, i] += Σ_k alphas[k, row]·x_panel[k, i]` for dense row-panel
  accumulation in one kernel.

Each returns `Result<(), SimdError>` and reports `LengthMismatch` on unequal
lengths. Every tail (the partial final vector of any non-dyadic length) runs
through the provider-owned masked-FMA seam, preserving the full-width masked
memory contract on scalar, AVX2, AVX-512, NEON, and the emulated SVE backend.

## GEMV: register-blocked matrix–vector product

```rust,ignore
use hermes_simd::gemv;

// A is row-major nrows × ncols; y += A·x, so zero y first for y = A·x.
let mut y = vec![0.0_f32; nrows];
gemv(&a, &x, &mut y, nrows, ncols)?;
```

GEMV is memory-bound: it performs `2·nrows·ncols` flops over `nrows·ncols`
matrix elements, an arithmetic intensity of 2 flops per element, so
throughput is governed by operand reuse, not FLOP rate. The kernel blocks
`TILE_M` rows: each `x` vector is loaded once and applied to all `TILE_M`
rows held in independent register accumulators, cutting `x` traffic by
`TILE_M×` and breaking the per-row FMA dependency chain. `TILE_M` scales with
the register file — wider ISAs block more rows before spilling. The
`nrows mod TILE_M` remainder runs as a single-row cleanup, so every shape is
supported.

The family is completed by `gemv_transpose` (`y += Aᵀ·x`), and the strided
forms `gemv_strided` / `gemv_transpose_strided`, which take a leading
dimension `lda ≥ ncols` for sub-matrix views; `lda = ncols` is the packed
form. All four validate their spans with overflow-safe dimension arithmetic —
an adversarial `lda` cannot wrap a length check and reach a raw SIMD load.

## The accumulate convention

Every GEMV and GEMM kernel **accumulates** into its output: `y += A·x`. This
matches the AXPY convention, so a caller wanting `y = A·x` zeroes `y` first.
The convention is uniform across the dense facade and is documented on each
entry point.

## What to notice

- **Fusion is the point.** `fmadd` collapses multiply+add into one rounding;
  the ternary and row forms eliminate temporaries entirely.
- **Tails are first-class.** The final partial vector uses the masked-FMA seam
  on every backend — no element-at-a-time cleanup, no masked-memory contract
  violation.
- **Validation is exact and overflow-safe.** Length mismatch is a typed error;
  dimension-product overflow is rejected at the boundary, not wrapped.
- **Blocking follows the bound.** GEMV's `TILE_M` exists because the kernel is
  memory-bound and the register file is the reuse cache.
