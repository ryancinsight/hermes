# 2. Runtime vs. Compile-Time Dispatch

hermes-simd uses *both* dispatch times, at different layers, and the boundary
between them is deliberate:

- **Compile time** resolves *which kernel exists for which ISA* — the
  monomorphized specialization of a generic kernel for each `(scalar type,
  architecture marker)` pair. Compile-time dispatch costs nothing at runtime:
  every specialization is a plain function, and the selection is a single
  comparison.
- **Runtime** resolves *which specialization to call* — a CPU-feature probe
  picks the widest ISA the host supports, once per call site's first use, and
  the selected kernel then runs hot without any per-element dispatch.

The two layers never cross. A compile-time-selected kernel is a fixed function;
a runtime-dispatched call is a fixed comparison. There is no per-lane or
per-element capability check anywhere in the hot path.

## The public surface is slice functions

The primary API reads as plain slice functions:

```rust,ignore
use hermes_simd::sum;

let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
let total = sum::<f32>(&data);
```

`sum`, `dot`, `min`, `max`, `abs_sum`, `argmin`, `argmax`, `scale`, `axpy`,
`axpy_mul`, `axpy_rows`, `axpy_rows_batch`, `elementwise_*`, `gemv`, `gemm`,
the masked operations, and the sparse `spmv_*` family all share this shape:
the caller passes slices (or validated sparse data) and gets a scalar, a
`Result`, or writes into an output slice. Each monomorphizes at the call site
to a concrete scalar type — `sum::<f32>` is a real `f32` specialization, not a
generic function dispatched at runtime.

The signature is the contract: `dot` returns `Result<T, SimdError>` because
length mismatch is a caller-visible failure; `sum` returns `T` because an empty
slice has a well-defined identity. Functions that can fail say so in their
return type; nothing fails silently.

## The dispatch ladder

Each operation is written once as a *generic kernel* over an architecture
marker, then wrapped by the `#[runtime_dispatch(avx512f, avx2, neon, scalar)]`
attribute. The macro expands the generic body into one specialization per
listed ISA and emits a wrapper that:

1. performs compile-time applicability checks first,
2. probes the host ISA features at runtime (through the exact capability gates
   from [Chapter 1](isa_detection.md)),
3. calls the widest specialization the host supports, falling back to `scalar`
   on every platform.

Because `scalar` is always last, every operation has a path that compiles and
runs on *any* target — the ISA-specific kernels layer over it, never replace
it. A build for an architecture with none of the listed features simply keeps
the scalar specialization.

The list order is the priority order, not an exhaustive matrix: `avx512f`
before `avx2` before `neon` on x86-64, with `scalar` as the floor. The runtime
probes guarantee the selected specialization is *provably* callable on the
host (the `HS-405` invariant): constructing a typed view or entering a
dispatched kernel for an unsupported target is a defect, never a "best
effort".

## Forced dispatch for tests and benchmarks

Runtime auto-selection is correct for production, but it makes testing awkward:
a probe that returns `false` skips the code it guards. For deterministic
coverage, `TargetId` names the closed set of targets — `Scalar`, `Avx2`,
`Avx512`, `Neon` — and `dispatch_view_to` / `dispatch_view_mut_to` force a view
onto a named target after a host-capability check. The CI backend-coverage
matrix uses exactly this surface: it asserts which targets *executed* on each
runner rather than trusting that a green run exercised any particular ISA.
`TargetId::supported_on_host()` distinguishes "architecture applies but this
CPU lacks the feature" from "not applicable", so a coverage report reads
truthfully.

## Monomorphization is the abstraction, not an implementation

Every op is parameterized by `T: Scalar` (via the sealed `SimdOps` trait for
the slice facade, or `SimdArch` plus a narrow operation-family facet such as
`SimdReduce<T>` at the kernel level) and is
compiled to machine code identical to a hand-written specialization. The
arch markers and op-strategy ZSTs (Chapter 3) are zero-sized and erase at
codegen. The dispatch wrapper is `#[inline(always)]`: once runtime selection
has picked a kernel, the call chain inlines down to the selected monomorphic
function — the same code a direct call would produce.

## What to notice

- **Selection happens once per operation call, not per element.** The cached
  probe plus one comparison chooses the kernel; the kernel body is straight-line
  monomorphic code.
- **The scalar floor is universal.** Every feature layer is optional; nothing
  requires an ISA to exist.
- **Force is explicit.** Auto-selection never guesses; testing a specific ISA
  uses `TargetId` + `dispatch_view_to`, and the coverage matrix records
  whether that target actually ran.
