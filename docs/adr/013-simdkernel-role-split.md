# ADR 013: Split `SimdKernel` into Operation-Family Facets

## Status

Accepted

## Context

`SimdKernel` previously combined the complete low-level SIMD operation surface:
load/store, dense arithmetic, bitwise operations, comparisons, reductions,
masking, gather/scatter, scans, and cross-lane permutations. Every consumer
therefore named the full backend contract even when it used one operation
family. The implementation set is sealed across the Scalar, AVX2, AVX-512,
NEON, and SVE backends; splitting every implementation block would multiply
method-bearing blocks without changing the machine code.

The existing associated types (`Vector`, `Mask`, `IndexVector`) and constants
(`LANE_COUNT`, `LANE_BOUND_CHECK`, `UNROLL_FACTOR`, and `SUPPORTS_NT_STORE`)
also belong to the closed backend implementation seam. Duplicating those
associations in every public role would create equality constraints and a
second source of backend truth.

## Decision

Use two layers with one implementation source:

1. `BackendKernel<T>` is the sealed, implementation-facing operation seam. It
   retains the canonical default bodies, native overrides, associated types,
   constants, and unsafe contracts. The name is `#[doc(hidden)]`; backend
   crates implement it, but consumers do not use it as their capability bound.
   `SimdStorage<T>` mirrors only the shared register and lane associations for
   role consumers through one blanket implementation, preserving that seam as
   the only backend source of truth.
2. Eight zero-sized, statically dispatched role facets are exposed from the
   kernel module: `SimdLoadStore`, `SimdArith`, `SimdBitwise`, `SimdCompare`,
   `SimdReduce`, `SimdMask`, `SimdGather`, and `SimdPermute`. Each facet is a
   public operation contract with a blanket forwarding implementation over
   `BackendKernel` and documents its operation family. A consumer can therefore
   state the family it requires without naming the full aggregate; the bound no
   longer exposes unrelated backend methods.
3. `SimdKernel<T>` is the public aggregate bound. It is a blanket aggregate of
   all role facets for callers that intentionally use several families.

The facets carry no data, use no dynamic dispatch, and forward through
monomorphized trait calls with no runtime control path. The existing backend
method bodies remain the single source of truth; monomorphization and
target-feature dispatch are unchanged. The operation-family modules live under
`kernel/roles/`, while the implementation seam lives in `kernel/backend.rs`.

The direct implementation-level split described in the original proposal is
rejected for this closed backend set. It would duplicate the associated-type
contract across eight traits and multiply impl blocks without reducing code or
changing consumer behavior. If a second
independently implemented backend family is admitted, this decision must be
revisited before opening that seam.

## API and migration impact

The old public implementation name `SimdKernel` is replaced by the public
aggregate with the same name; backend implementations and explicit qualified
method projections use `BackendKernel`. In-repository callers are migrated in
this change. External code that directly implements or explicitly projects
the old implementation trait requires the pre-1.0 migration to the aggregate
or the relevant role facet. No compatibility re-export is retained.

## Consequences

- Consumer bounds can name `SimdReduce<T>`, `SimdGather<T>`, or another role
  facet, improving interface segregation and documentation.
- Backend implementations remain one canonical generic contract, so no
  duplicated ISA kernels, fallback algorithms, or dynamic dispatch are added.
- The public aggregate remains available for multi-family algorithms.
- The `BackendKernel` name is intentionally hidden from normal rustdoc; the
  role facets and aggregate are the supported consumer surface.

## Verification

- `cargo fmt --all -- --check`.
- Core, intrinsics, and facade `cargo check --all-targets`.
- Role-facet value test through the Scalar backend.
- Focused nextest for the kernel property suite and the sum dispatcher.
- Full workspace Clippy, nextest, doctest, and Rustdoc gates before merge.
- Codegen comparison on a representative sum kernel; the facets must not add
  a runtime branch or allocation.

### Verification record (2026-08-14)

The exact branch head passed the focused check, full Hermes workspace gates,
and the role-facet value test. `cargo llvm-lines -p hermes-simd --release
--test kernel_property_tests` showed backend implementations such as
`<Scalar as BackendKernel<f32>>::sum_reduce` and
`<Avx2 as BackendKernel<f32>>::sum_reduce` as the emitted monomorphized
functions; no `SimdReduce` forwarding symbol was emitted. This is codegen
evidence that the forwarding facet is inlined and adds no separate runtime
dispatch symbol. It is not a benchmark claim.

`cargo semver-checks` reported the expected removed old `SimdKernel` methods,
associated types, and associated constants against `origin/main`; the same
check passed when classified as a major migration. The full workspace gates
were `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo nextest run --workspace --no-fail-fast` (465 passed),
`cargo test --doc --workspace`, and `RUSTDOCFLAGS=-D warnings cargo doc
--no-deps --workspace`.

### Revision record (2026-08-28)

HS-MASKED-TAIL-PARTIAL-LOAD adds active-prefix masked load/store to the
existing `BackendKernel` implementation seam and `SimdLoadStore` role. It does
not add a ninth role, a backend-specific public type, or another implementation
source: scalar, NEON, SVE, and x86 F16 inherit the canonical active-lane
default, while AVX2 and AVX-512 f32/f64 override only the instruction
selection. The existing full-width masked-memory methods retain their contract.

The revision is additive because a tail at an allocation or page boundary is
a distinct load/store capability. Generic conformance tests cover arbitrary
merge masks and every accessible-prefix length; Windows guard-page tests cover
the active and zero-mask boundary cases. Exact AVX2 f32/f64 code generation
verifies native mask-move instructions and removal of the former caller-tail
staging frames.

## References

- HS-436 in `backlog.md`.
- `crates/hermes-simd-core/src/kernel/backend.rs`.
- `crates/hermes-simd-core/src/kernel/roles/`.
- `docs/adr/005-per-type-x86-kernel-dedup.md`.
