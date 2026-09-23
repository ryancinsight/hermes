# ADR 024: Const-Generic Pair Decimation and Interleave

## Status

Accepted (2026-09-23)

## Context

`BackendKernel`, `SimdPermute`, and `Vector` exposed the lane-pair gather
and scatter of interleaved complex data as six methods named for their arity:
`deinterleave_pairs`, `deinterleave_pairs4`, `deinterleave_pairs8`,
`interleave_pairs`, `interleave_pairs3`, and `interleave_pairs5`. They are two
operations, the stride-`N` pair decimation and its inverse. Each new radix a
consumer needed added a trait method, a role forward, a `Vector` wrapper, and
per-backend overrides. The arity sat in the name as a structural constant.
Coverage was uneven: no `interleave_pairs4`, and no decimation at 3 or 5. The
defaults for 4 and 8 were hand-written chains of the pairwise method. Apollo
kept its own copy of that chain for 8 (`triple_small_groups.rs`).

## Decision

There is one method per direction, generic over the register count:
`deinterleave_pairs<const N: usize>([V; N]) -> [V; N]` and
`interleave_pairs<const N: usize>([V; N]) -> [V; N]`. Output `k` of the
decimation holds flat pairs `N q + k`. The interleave sends pair `q` of operand
`k` to flat pair `N q + k`.

The defaults live in `kernel::pair_permute`:

- For a power of two `N > 2`, they run `log2 N` levels of the backend's own
  `N = 2` arm. The decimation runs the levels in increasing width, and the
  interleave runs them in reverse.
- For other arities, and for `N = 2` itself, they use scalar lane emulation.
  This keeps the composition from recursing into a backend's `N = 2` arm.

A backend overrides the generic method once. It branches on `N`
(`if N == 4 { .. }`), names the registers of each specialized arm through
`pair_permute::cast_arity`, and forwards every other arity to
`pair_permute::deinterleave`/`interleave`. `N` is a constant, so each
instantiation keeps one arm and `cast_arity`'s length check folds away. The
hand-written networks move into the arms unchanged: AVX2 at 2, 4, 8 and 2, 3,
5; AVX-512 at 2; NEON `f32` at 2 and interleave 3. NEON `f64` holds one pair a
register, so both directions are the identity at every `N`.

## Rejected alternatives

- **Recursive default** (`Self::deinterleave_pairs::<{ N / 2 }>`) needs
  `generic_const_exprs`, which is nightly-only. The level loop computes the same
  composition on stable.
- **One trait per arity** (`PairDecimation<const N>`). The default for all `N`
  would be a blanket impl, and without specialization no backend could then
  override a single arity.
- **A closed-form half-concatenation override** for the backends with two pairs
  a register (AVX2 `f64`, NEON `f32`). Each output would pick its halves by
  loop index. The immediates would then depend on unrolling, and helpers
  indexed that way have compiled outside the dispatcher's `#[target_feature]`
  frame before. The explicit arms keep the instruction sequences already
  proven in codegen.

## Consequences

- Breaking change to `BackendKernel`, `SimdPermute`, and `Vector` (0.7 → next
  minor). Apollo migrates in the same co-evolution unit, as a tuple → array
  change at each call site.
- Every arity is available on every backend. Backends that specialize only
  `N = 2` (AVX-512, NEON `f32`) get the level composition at 4 and 8, where
  they previously took the hand-written default chain. The interleave at
  powers of two now composes shuffles where no method existed before.
- `transpose_interleaved_square` calls the one decimation at 2, 4, or 8.
- Evidence: `check_pair_permute::<T, A, N>` runs for `N` in {1, 2, 3, 4, 5, 6,
  8} at `f32` and `f64` on every host backend. It checks each output against
  the flat specification and each direction as the other's inverse. Swapping
  one output of the AVX2 `f64` `N = 8` arm fails it
  (`deinterleave_pairs::<8> output 4 mismatch`).

Item: `HERMES-PAIR-PERMUTE-ARITY`
([`backlog.md`](../../backlog.md#hermes-pair-permute-arity)).
