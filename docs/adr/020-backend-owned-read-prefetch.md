# ADR 020: Backend-Owned Read Prefetch

## Status

Accepted.

## Context

`HS-SPMV-GATHER-PREFETCH-2026-08-29` measures CSR multiplication with a 64 MiB
dense operand and 128 structurally nonzero entries per row. The operand exceeds
the development host's last-level cache, while four independent SIMD gathers
already expose memory-level parallelism. One-unroll-distance software prefetch
reduced median time by 8.5%/9.3% in the first paired run and 26.2%/10.1% in the
second at 1,048,576/2,097,152 nonzeros. Every candidate confidence interval was
disjoint from its paired production interval.

The generic CSR kernel cannot own an x86 or AArch64 intrinsic. Target-feature
safety and instruction selection belong to the sealed backend seam. A read hint
also belongs to the existing load/store capability family rather than a second
parallel memory trait.

## Decision

Add a default no-op `prefetch_read` operation to the sealed `BackendKernel`
contract and forward it through `SimdLoadStore`. The default preserves every
backend that lacks measured evidence. The AVX2 f32 provider overrides it with a
temporal read hint because that is the measured monomorphization; other scalar
and backend combinations retain the no-op until their own measurements justify
an override.

CSR schedules each hint one four-gather unroll group before use. A steady loop
handles only groups that have a future group to prefetch, and an epilogue uses
the same inlined unroll body without a conditional hint. The final machine loop
must contain the intended hints and no added branch, call, spill, or bounds
check. Failure of that code-generation gate rejects the implementation even if
wall-clock samples improve.

The hint address is derived only from `Validated<Csr>` columns after the public
SpMV entry check proves `x.len() >= ncols`. The unsafe provider contract still
requires a readable address even though current hardware treats the operation
as a non-faulting hint. Prefetch changes neither arithmetic order nor observable
results and performs no allocation.

## Rejected alternatives

1. Call architecture intrinsics directly from the generic CSR module. This
   bypasses provider ownership and duplicates target-specific policy in core.
2. Override every backend and scalar combination. Only AVX2 f32 has controlled
   evidence; widening the optimization would substitute inference for data.
3. Add a separate prefetch capability trait. Read prefetch is a memory-access
   operation and a second sealed forwarding hierarchy would duplicate
   `SimdLoadStore` and `BackendKernel`.
4. Retain a conditional hint in the unrolled loop. The branch is predictable,
   but the accepted code-generation oracle requires it to be structurally
   absent rather than relying on prediction.

## Consequences

- The public sealed capability surface grows additively and is classified
  `[minor] [arch]`; no caller or first-party implementation migration is
  required because both new methods have no-op defaults.
- AVX2 f32 CSR adds read traffic only for values the following unrolled group
  consumes. No operand, result, or scratch allocation changes.
- The permanent gather-bound Criterion rows become the regression instrument.
  They validate exact dyadic results before timing and report structural
  nonzeros rather than dense logical elements.
- AVX-512, AArch64, reduced-precision, and integer override decisions remain
  measurement-gated.

## Verification

- Independent dyadic scalar equality at both benchmark sizes.
- Two pinned-core paired Criterion comparisons with disjoint 95% confidence
  intervals and at least 5% median reduction at both sizes.
- Exact release code generation for hint, branch, call, spill, and bounds-check
  counts.
- Warning-denied all-target and AArch64 no-std compilation, focused/workspace
  Nextest, doctests, Rustdoc, SemVer classification, and benchmark smoke.

## Revision history

- 2026-08-29: Accepted for `HS-SPMV-GATHER-PREFETCH-2026-08-29` after the
  measurement threshold selected the provider-owned AVX2 f32 override.
