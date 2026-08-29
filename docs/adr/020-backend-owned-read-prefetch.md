# ADR 020: Backend-Owned Read Prefetch

## Status

Rejected.

## Context

`HS-SPMV-GATHER-PREFETCH-2026-08-29` measures CSR multiplication with a 64 MiB
dense operand and 128 structurally nonzero entries per row. The operand exceeds
the development host's last-level cache, while four independent SIMD gathers
already expose memory-level parallelism. The predeclared gate requires at least
a 5% median reduction at both 1,048,576 and 2,097,152 nonzeros in two paired
runs, with disjoint 95% confidence intervals.

The initial record mislabeled Criterion's mean change estimates as median
reductions. The retained second-pair samples have candidate medians of
4.3268/8.8900 ms and production medians of 5.4795/10.8422 ms, reductions of
21.04%/18.01%. Criterion overwrote the first candidate sample when the second
pair ran, so the claimed first-pair medians are not recoverable. Comparing the
retained final candidate sample with the retained first production sample gives
1.49%/18.29%, but that cross-run comparison is not a paired measurement and the
smaller row is below the threshold. The artifact therefore does not establish
the required two paired median wins.

## Decision

Reject software prefetch and remove the candidate backend capability, AVX2
intrinsic, and CSR loop changes. Production retains the original four-gather
kernel and public surface.

Retain the gather-bound Criterion rows and their explicit `harness = false`
registration. The instrument constructs and validates the 64 MiB fixture
outside the timed region, checks an independent exact dyadic scalar result, and
reports structural nonzeros. Future candidates must preserve both candidate
sample sets under distinct baseline names so the stated median gate is
reproducible after the run.

## Rejected alternatives

1. Change the selection statistic to Criterion's mean estimate after seeing the
   result. That changes the instrument's acceptance oracle after measurement.
2. Retain the candidate from the recoverable second pair alone. One pair cannot
   satisfy the two-run stability requirement.
3. Call architecture intrinsics directly from the generic CSR module. This
   bypasses provider ownership and duplicates target-specific policy in core.
4. Prefetch two unroll groups ahead. The controlled candidate regressed the
   1,048,576-nonzero row set by 30.3% and produced unstable 2,097,152-nonzero
   measurements.

## Consequences

- Production code, arithmetic, memory traffic, allocation behavior, and public
  API remain unchanged.
- The permanent gather-bound Criterion rows become the regression instrument.
- Software prefetch remains rejected until a new candidate satisfies a
  predeclared, durably retained two-run oracle.

## Verification

- Independent dyadic scalar equality at both benchmark sizes.
- Median recomputation from each retained `sample.json` using per-iteration
  times; the missing first candidate sample is an evidence failure.
- Exact source restoration against the production CSR/backend files.
- Warning-denied all-target compilation, workspace Nextest, benchmark smoke,
  formatting, and standalone lock validation.

## Revision history

- 2026-08-29: Rejected after independent review identified mean estimates
  mislabeled as medians and the retained samples failed to prove both required
  paired median comparisons. The initially accepted source candidate was
  removed before merge.
