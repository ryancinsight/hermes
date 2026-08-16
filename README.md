# hermes-simd

A high-performance, zero-overhead Rust SIMD abstraction workspace covering dense, sparse, complex, and packed sub-byte data-parallel kernels — plus AVX-512 VNNI tile GEMM and SWAR chess bitboards.

The workspace is designed for extreme runtime efficiency, using traits, ZST markers, const generics, and full compiler monomorphization to generate machine code identical to hand-optimized assembly. It compiles entirely on stable Rust with no unstable nightly compiler prerequisites.

## Workspace Structure

The project is structured as a multi-crate workspace (dependencies flow strictly downward):

- **Numeric vocabulary (external)**: the precision ladder (`Bf4`/`F4`/`Bf8`/`F8`/`F16`/`Bf16`/`F32`/`F64`/`I8`/`I16`/`I32`), packed 4-bit storage, and cast traits live in the [`eunomia`](https://github.com/ryancinsight/eunomia) crate — the Atlas numeric SSOT — and are re-exported through `hermes-simd`. (The former `hermes-numeric` member crate was migrated upstream.)
- **`crates/hermes-simd-core`**: Core abstractions — `SimdView<'a, T, Arch, Align, Mode, Ref>` typestate views, the `SimdKernel<T>` aggregate and operation-family facets over the `BackendKernel<T>` implementation seam, `SimdCow` dense copy-on-write, generic `SparseCow<T, Format, Arch>`, `BitMask<N>`, reduction/element/scan op strategy ZSTs, const-generic tiling, and N-D tensor views.
- **`crates/hermes-simd-intrinsics`**: Architecture-specific kernels (`Scalar`, `Avx2`, `Avx512`, `AvxVnni`, `Neon` ZST markers implementing the sealed `BackendKernel<T>` seam / tile traits), AVX-512 VNNI and 256-bit AVX-VNNI tile multipliers, packed 4-bit hardware unpacking, sliding-attack bitboard backends, and the Intel AMX engine (gated on a permission-aware runtime probe; see [Intel AMX status](#intel-amx-status)).
- **`crates/hermes-simd-types`**: Monomorphized convenience aliases and the compile-time `PreferredArch` selection.
- **`crates/hermes-simd-macros`**: Procedural macros — `#[runtime_dispatch]` generates compile-time-gated plus runtime-detected dispatchers from one generic kernel function.
- **`crates/hermes-simd`**: Public facade — the sealed `SimdOps` extension trait, runtime-dispatched free functions (`sum`, `dot`, `spmv_*`, `interleaved_complex_*`, …), and `dispatch_view` CPUID routing.
- **`crates/hermes-simd-examples`**: Demos — bitboards, dot products, copy-on-write, interleaved complex throughput.
- **`crates/hermes-simd-benches`**: Criterion + divan benchmark suite with report generation.

## Key Features

1. **Generic runtime dispatch**: one `<T: Scalar, A: SimdKernel<T>>` aggregate-bound kernel per operation, with narrow operation-family facets where applicable; `#[runtime_dispatch(avx512f, avx2, neon, scalar)]` emits the per-ISA `#[target_feature]` wrappers and the detection ladder. No per-type kernel clones, no type names in identifiers.
2. **Interleaved complex kernels**: `interleaved_complex_dot` / `interleaved_complex_mul_assign` over `[re, im, ...]` primitive slices, fully register-resident via adjacent-pair operation-family facets (`SimdPermute`, `SimdArith`) with AVX2/AVX-512/NEON overrides and a `const CONJ_B` conjugation flag (see `docs/adr/004`).
3. **Copy-on-write containers**: `SimdCow` (dense, with map/zip/reduce/scan/norm extensions) and one generic `SparseCow<T, F, Arch>` covering every sparse format through the `CowFormat` trait — zero-copy reads, single-allocation promotion.
4. **Sparse SIMD (SpMV)**: format-parameterized views for CSR, Sliced ELLPACK (SELL-p), Blocked COO, and Dense-with-Mask layouts.
5. **VNNI tile GEMM**: AVX-512 VNNI and 256-bit AVX-VNNI tile multipliers behind a single internal `vpdpbssd` asm macro, plus bit-parallel INT4→INT8 unpacking. Intel AMX kernels exist in the tree and dispatch only where the CPUID / `XCR0` / OS-permission chain holds, which no CI machine satisfies — see [Intel AMX status](#intel-amx-status).
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
`SimdLoadStore` load/store primitives, an SSE2 feasibility ADR, a public dense
cross-target conformance matrix, and a finer operation-family coverage map.
The audit does not replace Hermes' sealed `SimdKernel` aggregate/facets, sparse/packed
domain kernels, AMX tiling, COW containers, tensor views, or Atlas compute
boundaries.

The target-token increment is available as `TargetId`,
`dispatch_view_to`, and `dispatch_view_mut_to`. Unsupported targets return
`None` before constructing a target-specific view, so benchmark and test
harnesses can force a backend only when the host can execute it.

Safe one-vector slice wrappers are available on `Vector<T, Arch>` as
`load_unaligned_from_slice`, `load_aligned_from_slice`,
`store_unaligned_to_slice`, and `store_aligned_to_slice`. These wrappers check
slice length and vector-width alignment before calling the raw
`SimdLoadStore` load/store primitives.

Dense target conformance is covered by host-capability tests that force every
supported `TargetId` and compare sum, dot, elementwise arithmetic, gather, and
select against the scalar target. Scatter is covered by per-backend property
tests instead, since it needs a mutable view per backend rather than the shared
read-only view the `TargetId` matrix forces.

The operation-family coverage map is tracked in
[`backlog.md`](backlog.md#operation-family-coverage-map). Delivered families
include arithmetic, reductions, masks/select, memory views/wrappers, consumer
shuffle primitives, cross-lane permutes (`reverse`, `interleave`,
`deinterleave`, all on the flat lane sequence), float-specialized kernels, and
indexed gather/scatter.
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

Sparse SpMV, register-blocked tiling, and the packed sub-byte kernels are
unconditional parts of the library, not opt-in features.

## Intel AMX status

**AMX dispatches only where the full permission chain holds — which is no
machine currently in CI.** The hardcoded `false` is gone: the tile kernels
(`tdpbf16ps`, `tdpbssd`), the `AmxConfig` tile descriptors, and the fallible
RAII `AmxSession`/`AmxBatchSession` guards are now gated on a real runtime
probe (`crates/hermes-simd-intrinsics/src/x86_64/amx/probe.rs`), which
`crates/hermes-simd/src/cpu.rs` consumes as the capability SSOT.

The probe refuses unless all three of these hold, because each is checked by a
different mechanism and none implies the others:

1. **Silicon** — `CPUID.(EAX=7,ECX=0).EDX` bit 24 (`amx-tile`), plus bit 22
   (`amx-bf16`) or bit 25 (`amx-int8`) for the respective kernel.
2. **OS state** — `XCR0` bits 17 (`XTILECFG`) and 18 (`XTILEDATA`), read via
   `XGETBV` after confirming `OSXSAVE`. Clear bits mean `#UD` no matter what
   CPUID says.
3. **Process permission** — `XTILEDATA` is XFD-gated (8 KiB of state, so the OS
   traps first use rather than growing every signal frame). XCR0 advertises the
   component system-wide while `IA32_XFD` withholds it per thread, so step 2
   does *not* subsume this. Executing without it raises `#NM`.

Step 3 is platform-specific. On **Linux** the probe calls
`arch_prctl(ARCH_GET_XCOMP_SUPP)`, then `arch_prctl(ARCH_REQ_XCOMP_PERM,
XFEATURE_XTILEDATA)`, then re-reads `ARCH_GET_XCOMP_PERM` to confirm the grant.
On **Windows** it requires both AMX bits in `GetEnabledXStateFeatures()`, then
calls `EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA)` and
verifies with `GetThreadEnabledXStateFeatures()`; both entry points are resolved
with `GetProcAddress` because they do not exist before Windows 11 / Server 2022,
where a static import would make the binary fail to load. Every other OS
refuses. Note that Rust's own `is_x86_feature_detected!("amx-tile")` implements
steps 1 and 2 only, so it is unsound for this purpose even once it stabilizes.

Probing has a **side effect**: learning whether permission is obtainable
requires requesting it, so the first call performs a process-wide, irreversible
opt-in that enlarges the XSAVE area for every thread. The result is cached, so
this happens at most once.

**AVX-512 executes under emulation; real-silicon timing is best-effort.** The
`test-avx512-sde` job runs the suite under Intel SDE emulating Sapphire
Rapids, so the AVX-512 paths execute deterministically on every push and the
coverage step asserts `scalar,avx2,avx512,sve` without requiring silicon.
Performance evidence is a separate claim SDE cannot make, so the
`test-avx512-hosted` job additionally records the machine class of the
GitHub-hosted x86 runner and, when that host happens to carry AVX-512, asserts
`scalar,avx2,avx512` and captures the permute A/B benchmark natively.
GitHub's hosted x86 pool is heterogeneous — some Intel parts have AVX-512 and
others (AMD, older Intel) do not — so native timing is opportunistic: on hosts
without the silicon the coverage report shows AVX-512 as `NOT COVERED` and the
benchmark is skipped loudly, never silently. AMX remains a possibility rather
than a claim: the probe's third condition is the real
`arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA)` syscall, which the runner kernel
may or may not grant, so `amx` is deliberately absent from both jobs'
`HERMES_EXPECTED_TARGETS`. Where it is admitted, the AMX GEMM dispatches and
must match the `scalar/tiling.rs` reference within a derived bound.

**Remaining trigger** ([`backlog.md`](backlog.md) → Open): a Sapphire-Rapids (or
later) Linux runner on which the probe returns `true` — i.e. the runner kernel
admits `XTILEDATA` — the AMX GEMM dispatches, and its result matches the
`scalar/tiling.rs` reference within a derived bound. A `test-avx512-hosted`
host whose kernel admits XTILEDATA may satisfy this; it is not asserted
because kernel admission varies.

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

### High-level Tile GEMM (VNNI)
```rust
use hermes_simd::gemm;

let (m, n, k) = (32, 32, 64);
let a = vec![1i8; m * k];
let b = vec![2i8; k * n];
let mut c = vec![0i32; m * n];

// Automatically dispatches down the ladder: AVX-512 VNNI → 256-bit AVX-VNNI
// (client CPUs without AVX-512) → scalar tiles.
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

Differential testing policy: the AVX2, AVX-512, and NEON backends are verified against the always-available `Scalar` backend — bitwise on dyadic-exact inputs, within analytically derived rounding bounds on arbitrary inputs. AVX-512 executes under Intel SDE emulating Sapphire Rapids (`test-avx512-sde`), with native timing captured best-effort on hosted silicon whenever it is present (`test-avx512-hosted`), and NEON runs on a native aarch64 runner (`test-aarch64`), so no backend is carried by a capability-gated skip. The AMX kernels are compile-checked only; they are not runtime-validated, because no available machine satisfies the permission chain — the SDE job cannot stand in, since `arch_prctl` reaches the host kernel rather than the emulator (see [Intel AMX status](#intel-amx-status)).

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

Current version: **0.6.0** (pre-release; canonical trait surfaces defined, breaking changes documented per minor release).
