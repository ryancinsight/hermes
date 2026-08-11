# hermes-simd

A high-performance, zero-overhead Rust SIMD abstraction workspace covering dense, sparse, complex, and packed sub-byte data-parallel kernels — plus Intel AMX tiling, AVX-512 VNNI, and SWAR chess bitboards.

The workspace is designed for extreme runtime efficiency, using traits, ZST markers, const generics, and full compiler monomorphization to generate machine code identical to hand-optimized assembly. It compiles entirely on stable Rust with no unstable nightly compiler prerequisites.

## Workspace Structure

The project is structured as a multi-crate workspace (dependencies flow strictly downward):

- **Numeric vocabulary (external)**: the precision ladder (`Bf4`/`F4`/`Bf8`/`F8`/`F16`/`Bf16`/`F32`/`F64`/`I8`/`I16`/`I32`), packed 4-bit storage, and cast traits live in the [`eunomia`](https://github.com/ryancinsight/eunomia) crate — the Atlas numeric SSOT — and are re-exported through `hermes-simd`. (The former `hermes-numeric` member crate was migrated upstream.)
- **`crates/hermes-simd-core`**: Core abstractions — `SimdView<'a, T, Arch, Align, Mode, Ref>` typestate views, the `SimdKernel<T>` operation trait, `SimdCow` dense copy-on-write, generic `SparseCow<T, Format, Arch>`, `BitMask<N>`, reduction/element/scan op strategy ZSTs, const-generic tiling, and N-D tensor views.
- **`crates/hermes-simd-intrinsics`**: Architecture-specific kernels (`Scalar`, `Avx2`, `Avx512`, `AvxVnni`, `Neon` ZST markers implementing `SimdKernel<T>` / tile traits), Intel AMX engine, AVX-512 VNNI and 256-bit AVX-VNNI tile multipliers, packed 4-bit hardware unpacking, and sliding-attack bitboard backends.
- **`crates/hermes-simd-types`**: Monomorphized convenience aliases and the compile-time `PreferredArch` selection.
- **`crates/hermes-simd-macros`**: Procedural macros — `#[runtime_dispatch]` generates compile-time-gated plus runtime-detected dispatchers from one generic kernel function.
- **`crates/hermes-simd`**: Public facade — the sealed `SimdOps` extension trait, runtime-dispatched free functions (`sum`, `dot`, `spmv_*`, `interleaved_complex_*`, …), and `dispatch_view` CPUID routing.
- **`crates/hermes-simd-examples`**: Demos — bitboards, dot products, copy-on-write, interleaved complex throughput.
- **`crates/hermes-simd-benches`**: Criterion + divan benchmark suite with report generation.

## Key Features

1. **Generic runtime dispatch**: one `<T: Scalar, A: SimdKernel<T>>` kernel per operation; `#[runtime_dispatch(avx512f, avx2, neon, scalar)]` emits the per-ISA `#[target_feature]` wrappers and the detection ladder. No per-type kernel clones, no type names in identifiers.
2. **Interleaved complex kernels**: `interleaved_complex_dot` / `interleaved_complex_mul_assign` over `[re, im, ...]` primitive slices, fully register-resident via adjacent-pair `SimdKernel` primitives (`swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd`) with AVX2/AVX-512/NEON overrides and a `const CONJ_B` conjugation flag (see `docs/adr/004`).
3. **Copy-on-write containers**: `SimdCow` (dense, with map/zip/reduce/scan/norm extensions) and one generic `SparseCow<T, F, Arch>` covering every sparse format through the `CowFormat` trait — zero-copy reads, single-allocation promotion.
4. **Sparse SIMD (SpMV)**: format-parameterized views for CSR, Sliced ELLPACK (SELL-p), Blocked COO, and Dense-with-Mask layouts.
5. **Intel AMX acceleration**: stable inline-assembly AMX (`tdpbf16ps`, `tdpbssd`), fallible RAII `AmxSession` tile-config caching guarded by runtime support, 2×2 register blocking; VNNI tile GEMM uses a single internal `vpdpbssd` asm macro plus bit-parallel INT4→INT8 unpacking.
6. **SWAR chess bitboards**: Kogge-Stone, Hyperbola Quintessence, Fancy Magic, and Hybrid SWAR-Magic sliding-attack backends behind one `BitBoardView`.
7. **Typestate safety**: alignment (`Aligned<A>`/`Unaligned`), execution mode (`Masked`/`Unmasked`), and reference mutability are compile-time parameters with zero layout overhead.
8. **Precision ladder**: 4-bit through 64-bit numeric types with packed storage and hardware-accelerated unpacking into `SimdCow`.

## Atlas Compute Boundaries

Hermes is the Atlas SIMD substrate: it owns lane-parallel CPU kernels, scalar
fallbacks, ISA dispatch, packed-lane representation, and zero-copy SIMD views.
Thread-level MIMD scheduling belongs to Moirai, and GPU execution belongs to
the Hephaestus substrate consumed through Coeus/Apollo. Consumers compose these
layers by selecting Hermes for per-core vector work, Moirai for partitioning
independent work across cores, and Hephaestus for device-resident kernels.
Hermes APIs therefore remain synchronous, slice-oriented, and monomorphized;
they do not own task scheduling or GPU resource lifetimes.

## External SIMD Reference Baseline

Hermes tracks external SIMD libraries as coverage references, not as API
authorities. The current external audit compares Hermes with
[`NikoMalik/highway`](https://github.com/NikoMalik/highway) at commit
`0984271e74db124cf5e200de542e745348eb0b9e`; findings live in
[`gap_audit.md`](gap_audit.md#highway-2026-06-14).

Actionable gaps from that audit are Hermes-native: target-token forced
dispatch for tests/benchmarks, safe one-vector slice wrappers over raw
`SimdKernel` load/store primitives, an SSE2 feasibility ADR, a public dense
cross-target conformance matrix, and a finer operation-family coverage map.
The audit does not replace Hermes' sealed `SimdKernel` facade, sparse/packed
domain kernels, AMX tiling, COW containers, tensor views, or Atlas compute
boundaries.

The target-token increment is available as `TargetId`,
`dispatch_view_to`, and `dispatch_view_mut_to`. Unsupported targets return
`None` before constructing a target-specific view, so benchmark and test
harnesses can force a backend only when the host can execute it.

Safe one-vector slice wrappers are available on `Vector<T, Arch>` as
`load_unaligned_from_slice`, `load_aligned_from_slice`,
`store_unaligned_to_slice`, and `store_aligned_to_slice`. These wrappers check
slice length and vector-width alignment before calling the raw `SimdKernel`
load/store primitives.

Dense target conformance is covered by host-capability tests that force every
supported `TargetId` and compare sum, dot, elementwise arithmetic, gather, and
select against the scalar target. Scatter is covered by per-backend property
tests instead, since it needs a mutable view per backend rather than the shared
read-only view the `TargetId` matrix forces.

The operation-family coverage map is tracked in
[`backlog.md`](backlog.md#operation-family-coverage-map). Delivered families
include arithmetic, reductions, masks/select, memory views/wrappers, consumer
shuffle primitives, float-specialized kernels, and indexed gather/scatter.
Scatter was admitted as the write-side dual of the already-public `gather`
rather than from a consumer request: a one-directional lane-addressing model
forces any scatter-shaped caller out of the vector domain entirely. Pending
families remain admitted only from consumer demand: compress-store,
comparison predicates,
standalone conversions, broad bitwise public facades, and crypto/hash
primitives. This keeps Hermes as the SIMD SSOT without cloning Highway's full
catalog or claiming unsupported operations.

The public AXPY facade includes `axpy`, `axpy_rows`, and `axpy_rows_batch`.
`axpy_rows_batch` fuses a depth-major panel accumulation into one
runtime-dispatched kernel, so dense row-panel consumers avoid repeated facade
dispatch, allocate no temporaries, and store each output lane once after
accumulating across depth in registers.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Enables runtime CPU feature detection |
| `mnemosyne-memory` (default) | Routes aligned vector allocation through Mnemosyne; topology queries belong to Themis |
| `libnuma` | Enables Linux affinity and residency probes; Hermes does not expose topology query facades |
| `sparse` | Enables `SparseView` SpMV layouts and computation |
| `tiling` | Enables register-blocked tiling dot products and GEMV |
| `bytemuck` | Enables safe type-casting via the `bytemuck` crate |
| `wide` | Enables the `wide` crate backend fallback |
| `portable-simd` | Enables nightly standard library `std::simd` |

---

## Quickstart

### Dense Sum Reduction (Runtime Dispatch)
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

### Interleaved Complex Dot (Runtime Dispatch)
```rust
use hermes_simd::interleaved_complex_dot_runtime;

// [re0, im0, re1, im1]: (1 + 2i), (3 + 4i)
let a = [1.0f64, 2.0, 3.0, 4.0];
let b = [5.0f64, 6.0, 7.0, 8.0];

// sum(a[k] * conj(b[k])) — CONJ_B selects conjugation at compile time
let (re, im) = interleaved_complex_dot_runtime::<f64, true>(&a, &b).unwrap();
assert_eq!((re, im), (70.0, 8.0));
```

### High-level GEMM (AMX / VNNI Fallback)
```rust
use hermes_simd::gemm;

let (m, n, k) = (32, 32, 64);
let a = vec![1i8; m * k];
let b = vec![2i8; k * n];
let mut c = vec![0i32; m * n];

// Automatically dispatches down the ladder: Intel AMX → AVX-512 VNNI →
// 256-bit AVX-VNNI (client CPUs without AVX-512) → scalar tiles.
unsafe {
    gemm::<i8, i8, i32>(m, n, k, &a, k, &b, n, &mut c, n).unwrap();
}
```

---

## Verification

```powershell
# Unit, integration, differential, and property tests (proptest)
cargo nextest run --workspace

# Lint and format gates
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# Cross-target compile check (NEON kernels)
cargo check --workspace --target aarch64-unknown-linux-gnu

# Benchmarks → updates benchmarks_results.md
cargo run -p hermes-simd-benches

# Parse existing Criterion output, refresh the JSON baseline, and enforce it
cargo run -p hermes-simd-benches -- --parse-only --write-baseline --check-regressions

# Enforce an existing baseline without updating it (default threshold: 1.10x)
cargo run -p hermes-simd-benches -- --parse-only --check-regressions
```

Differential testing policy: every optimized backend (AVX2, AVX-512, NEON, AMX) is verified against the always-available `Scalar` backend — bitwise on dyadic-exact inputs, within analytically derived rounding bounds on arbitrary inputs.

Benchmark regression policy: `benchmarks_baseline.json` is the structured
Criterion baseline. `--check-regressions` fails when a committed baseline row is
missing from the current run or when the current point estimate exceeds the
baseline by the configured threshold.
The dense suite includes `axpy_rows_batch_f32`, which compares fused
depth-major row-panel accumulation against repeated public `axpy_rows` calls.

## Project Management

- Design decisions: [`docs/adr/`](docs/adr/)
- Strategic roadmap: [`backlog.md`](backlog.md)
- Active sprint tactics: [`checklist.md`](checklist.md)
- Version history: [`CHANGELOG.md`](CHANGELOG.md)

Current version: **0.2.0** (pre-release; canonical trait surfaces defined, breaking changes documented per minor release).
