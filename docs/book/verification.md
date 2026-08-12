# 10. Verification: How We Know the Kernels Are Right

Every kernel in this book claims speed *and* correctness. This chapter is the
accounting: what evidence stands behind a `hermes_simd` operation, and what
each test tier can and cannot establish. The invariant hierarchy is the
backbone — each claim is checked by the layer that can check it, and no
category of evidence is allowed to stand in for another.

## The invariant ladder

- **Host support (the `HS-405` invariant).** A kernel is callable only when
  the host executes its architecture. Enforced at the type level: constructing
  a `SimdView` / `AlignedVec::view` checks that the `Arch` marker runs on this
  host, so holding the view proves callability. No kernel can be reached
  without passing the check.
- **No uninitialized access.** No `&mut [T]` ever spans uninitialized memory;
  growth flows through `MaybeUninit` spare capacity and `unsafe set_len`
  publishes only the initialized prefix. This is a type/API shape invariant,
  checked by the constructors.
- **Dispatch coherence.** The capability check that gates a kernel is the same
  check the view construction performed — the runtime ISA detection and the
  type marker cannot disagree.
- **Masked-path correctness.** Predicated operations produce the same values
  as their unmasked counterparts on the active lanes and leave inactive lanes'
  destinations at the masked-out identity.

The first two are established by construction; the last two by tests.

## Differential tests against a scalar reference

Each dispatch module carries a `#[cfg(test)]` scalar reference — written as a
plain loop, `ln` (linear naive) — and the central regression test is named
for it: `matches_ln_across_tail_sizes`. This is deliberate. The canonical SIMD
defect is the **tail**: a kernel that processes `LANE_COUNT`-sized chunks and
mis-handles the `n % LANE_COUNT` remainder either reads past the buffer or
computes nothing for the final partial chunk. Running every kernel against the
scalar reference at every tail size (`n` just below, at, and just above a
multiple of the lane count) makes that defect class impossible to ship
silently.

The comparison is **value-semantic**: the SIMD result must equal the scalar
reference within a derived epsilon where reordering is possible, bitwise where
evaluation order provably matches. This matters because reduction order — not
correctness — legitimately differs between the sequential `ln` reference and
a tree-reduced SIMD accumulation; asserting bitwise equality there would be a
false oracle. The tolerance is derived from the machine epsilon and the
reduction depth, not tuned to pass.

## Property tests

`proptest` covers the algebraic contracts a fixed input cannot reach: round-
trip and idempotence laws, shape and length invariants, and adversarial
ranges. Property suites live in dedicated test files
(`property_tests.rs`, `kernel_property_tests.rs`), and their
`proptest-regressions` files are committed — so a counterexample that ever
finds a real defect is replayed deterministically on every CI run until the
defect is fixed. A regression file that silently disappears is itself a
defect signal.

The kernel properties are generic-instantiation properties where they can be:
a law that must hold for `f32` and `f64` is written once and run for every
shipped scalar type and backend, so a newly admitted type inherits the full
suite (and a newly admitted backend inherits the differential guarantees).

## Backend coverage and host capability

Because dispatch is runtime ISA detection (Chapter 3), there are two
verification fronts:

- `backend_coverage_tests` — every kernel family runs under every backend the
  host can provide, asserting the dispatch path executes and agrees with the
  scalar reference. On a host without AVX-512, the AVX-512 backend is
  exercised via its own differential suite where the ISA is emulatable, and
  the coverage limit is recorded rather than fabricated.
- `host_capability_tests` — the capability probes themselves are tested: the
  runtime detection (`has_fma3` and friends) is exercised against the compiled
  target features, so the dispatch gate that chooses the kernel is itself a
  tested function, not an untested assumption.

## Benchmarks as measurement instruments

The benchmark suites (`dense`, `sparse`, `complex`) are not demonstrations —
they are instruments with fixed inputs and a measured baseline, so a regression
is a statistically significant change against the stored baseline, never a
single-shot number. The tiling guidance in Chapter 8 is derived from the same
evidence: `TILE_M` follows the architecture's FMA throughput hint (4 for AVX2
and NEON, 8 for AVX-512), which is a property of the machine's dependency
latency and port count, not a tuning guess.

Two benchmark disciplines carry over from the kernels themselves:

- **The reference is the oracle.** A fast path is "fast" only relative to a
  measured slower path with the same inputs; an optimization claim without a
  baseline comparison is unverified.
- **Same-arch runs are reproducible.** The kernels here are single-threaded
  and iterate in a fixed order, so the same kernel on the same inputs is
  bitwise reproducible across runs. What varies between backends is
  instruction-set width and the `TILE_M` accumulator layout (Chapter 8), which
  changes summation order — so cross-backend differential tests assert
  agreement within Chapter 5's reassociation envelope, never bitwise
  equality. A bitwise mismatch against the scalar reference is a
  defect; an in-envelope match is the expected reordering.

## What the ladder establishes

Put together: **construction** proves support and initialization, **typed
errors** prove input safety at boundaries (the sparse `ValidatedData`
certificate from Chapter 9), **differential and property tests** prove value
semantics against independent references, and **baselines** prove performance.
Each claim is checked by the strongest evidence category that applies to it —
and no claim is passed off as a stronger one than its evidence warrants.
