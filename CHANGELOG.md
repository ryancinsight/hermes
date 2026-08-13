# Changelog

All notable changes to the hermes-simd workspace. Format: [Keep a Changelog]; versioning: SemVer 2.0.0 (pre-1.0: minor releases may break, documented under **Breaking**).

## [Unreleased]

### Fixed

- [patch] The `codegen` binary propagates its file-I/O failures instead of
  panicking through four `unwrap()`s, and carries the crate documentation it
  previously lacked.

- [patch] `test_packed4_cow_state_accessors_preserve_packed_borrow` asserted
  `F4` values where it means raw nibble codes. Upstream eunomia made
  `PartialEq` on the sub-byte float wrappers float-semantic instead of bitwise
  (correctly: the derived bitwise ordering was sign-inverted), and code 7 is a
  NaN, so `F4(7) == F4(7)` is now properly false. The test pins copy-on-write
  promotion round-tripping the exact stored nibble, so it compares `.0`
  directly — which also keeps the NaN codes covered rather than avoiding them.

- [patch] `compress_bench` aborted on its scalar rows: it built a
  `BitMask::<1>` for a backend whose f32 lane count is 4, tripping
  `SimdView::compress`'s lane-count assertion. Broken since 2026-07-07 and
  unnoticed because the benchmark job runs only on pull requests and manual
  dispatches. The mask width now derives from the backend's `LANE_COUNT`
  instead of a literal.

### Changed

- [patch] Lint policy is now a single `[workspace.lints]` table inherited by
  every member, replacing three overlapping per-crate `#![allow(..)]` blocks
  and four copies of `#![deny(missing_docs)]`. `clippy::pedantic` is the floor;
  `unwrap_used`, `dbg_macro`, `print_stdout`, and `print_stderr` are denied, and
  `allow_attributes` drives the `#[allow]` -> `#[expect]` migration. The
  allowed pedantic lints are listed once, each with the domain reason it does
  not apply to vector kernels.
  Notably `clippy::missing_safety_doc` is no longer suppressed: it had been
  allowed across the whole of `hermes-simd-intrinsics`, the crate holding
  roughly 1270 `unsafe` sites, so an `unsafe` public function could ship
  without a `# Safety` section unnoticed.
  The floor is set against 2152 remaining library-src pedantic findings, which
  are recorded as a non-increasing ratchet (backlog HS-435) rather than
  silently allowed.

- [patch] The native aarch64 CI job now measures the NEON cross-lane permute
  overrides against the generic store/permute/load defaults. It saves a
  `neon-native` Criterion baseline, rebuilds with the explicit
  `hermes_benchmark_generic_default` comparison configuration, and reruns the
  same bounded benchmark rows. The workflow records evidence on real ARM
  silicon; it makes no speed claim until the comparison is adjudicated. AVX-512
  performance remains gated on HS-429 real silicon because Intel SDE cannot
  provide timing evidence.

### Added

- [minor] Native cross-lane permute overrides on AVX-512 (`vpermps`/`vpermpd`,
  `vpermi2ps`/`vpermi2pd`) and NEON (`rev64`+`ext`, `zip1`/`zip2`,
  `uzp1`/`uzp2`), verified by the existing differential and round-trip tests on
  the SDE and aarch64 runners. `benches/permute.rs` is the committed regression
  baseline. Note the AVX2 interleave/deinterleave overrides were written,
  measured, and **removed**: `unpack` + `permute2f128` runs 37% slower than the
  generic default at L1-resident size, because LLVM already lowers the
  default's stack round-trip into good shuffle sequences. AVX2 `reverse` was
  kept on measurement (10.4% faster at 1024 f32). AVX-512 override performance
  is not yet measured. On native aarch64, the HS-430 A/B gate found
  NEON reverse neutral against the generic default and removed both f32/f64
  overrides; large f32 interleave and deinterleave improved 1.27% and 1.40%
  respectively and remain. The exact hosted run was 31694336159; AVX-512
  timing remains gated on HS-429 real silicon.
- [patch] CI now executes the capability-gated ISA paths instead of skipping
  them. AVX-512 branches are `is_x86_feature_detected!`-guarded, and the
  capability report added with the new `test-avx512-sde` job proved the x86
  runner has no AVX-512 or AMX flags — so those paths, including the AVX-512
  scatter override, BF16 tile dispatch, VNNI, and AMX, had never run in CI. The
  job executes the suite under Intel SDE emulating Sapphire Rapids (444/444,
  ~11x native) via the cargo target runner, so compilation stays native. A
  dedicated `sde` nextest profile carries the emulation budget; the native 30s
  budget is unchanged.
- [minor] Cross-lane permutes: `SimdKernel::reverse`, `interleave`, and
  `deinterleave` join the trait as defaulted methods, so general lane
  reordering no longer requires leaving the vector domain and no existing
  backend impl changed. Previously the only lane shuffles were the complex
  adjacent-pair primitives (`swap_adjacent`, `dup_even`, `dup_odd`), which
  express interleaved complex arithmetic and nothing else. All three are
  specified on the flat lane sequence rather than per 128-bit sub-lane;
  `deinterleave` is the exact inverse of `interleave`, and `reverse` is an
  involution. AVX2 overrides `reverse` natively (`vpermps` for f32, `vpermpd`
  for f64). Verification: per-backend differential tests against an external
  slice reference plus both round-trip identities. Native AVX-512 and NEON
  overrides remain open — see HS-427; they are deliberately not shipped
  unverified, since wrong permute indices return plausible wrong lanes rather
  than failing loudly.
- [minor] Indirect indexed store (scatter), the write-side dual of `gather`.
  `SimdKernel::scatter` / `scatter_masked` join the trait as defaulted methods,
  so every existing backend gains them without an impl change; AVX-512 f32/f64
  override them with native `vscatterdps`/`vscatterdpd`, while AVX2 and NEON
  keep the lane-sequential default because neither ISA has a scatter
  instruction. `SimdView::scatter` is the public surface, mirroring
  `SimdView::gather`: indices are fully validated before any write, so the
  error path leaves the view untouched, and the final partial vector routes
  through `scatter_masked` rather than a scalar tail loop. Duplicate indices
  resolve last-writer-wins, matching the hardware rule on both paths.
  Verification: per-backend differential property tests against the scalar
  reference, a gather∘scatter round-trip identity, duplicate-index and
  error-contract tests. The native AVX-512 path is executed under the Intel SDE
  job, so the earlier "runner-gated" caveat on this entry is discharged.

### Changed

- [arch] Native AVX-512 BF16 tile dispatch is now available when Rust's exact
  `avx512bf16` runtime probe succeeds. The `Bf16 × Bf16 → F32` tile uses
  `DPBF16PS` while retaining the AVX-512F/BW/VL conversion/FMA fallback for
  non-BF16 AVX-512 hosts. Differential coverage validates nonzero `C += A·B`;
  hardware-specific runtime and benchmark evidence remains runner-gated.
- [minor] Native SVE remains explicitly blocked on the pinned stable Rust
  toolchain. `SveArch` continues to provide a safe lane-emulated backend, and
  `is_native_hardware_supported` reports hardware capability separately without
  claiming native execution.
- [patch] Mutable `SimdView::transform_in_place` tails now stage both
  operands in initialized provider-local buffers, use the generic `ElementOp`
  vector seam, and copy back only live result lanes. Add/Sub/Mul/Div share the
  same bounds-safe tail implementation, with forced emulated-SVE odd-length
  coverage.
- [patch] `SimdView::zip_reduce(Dot)` now routes its final pairwise vector
  through two initialized provider-local buffers and the generic masked
  reduction seam, removing the scalar dot tail while preserving full-width
  masked-memory safety. Forced emulated-SVE non-dyadic f32 coverage records
  the expected reassociation tolerance; multiplicative reductions retain their
  scalar contract.
- [patch] Generic `Sum`/`Min`/`Max` reductions and reusable view kernels now
  route final partial vectors through initialized provider-local buffers and
  leading masks. `SimdView::sum` delegates to `reduce(Sum)`, while masked
  add/multiply/FMA, elementwise multiply, and generic `zip_into` avoid scalar
  cleanup loops without reading beyond the live slice. Eunomia min/max NaN and
  signed-zero behavior remains the contract; floating sums retain the existing
  SIMD reduction-order envelope.
- [patch] Popcount reductions now route their final partial vectors through
  `SimdKernel::masked_sum_reduce`, including the shared bitwise binary path.
  Initialized local lane buffers preserve bounds safety for blend-based backends,
  and each masked tail count is exact; the existing whole-reduction accumulator
  contract is unchanged. Generic reduction and broader view tails are covered
  by the HS-416 increment; other hot kernels remain separate follow-ups.
- [patch] Absolute reductions (`AbsSum`/`AbsMax`) now route their final partial
  vector through a generic masked reduction seam. The view copies live elements
  into initialized local lanes, applies the absolute transform once, and merges
  inactive lanes with the neutral identity. Generic sum/min/max and other
  reduction tails remain separate follow-ups.
- [patch] Hermes row-update tails now route `axpy_rows` and `axpy_rows_batch`
  through provider-owned masked fused multiply-add. Fully initialized local lane
  buffers preserve the AVX2 blend-based bounds proof, and the batched path keeps
  its existing depth accumulation order. Reductions, views, and other scalar
  tails remain separate follow-ups.
- [patch] Transposed GEMV column tails now use initialized provider-local lane
  buffers plus the existing masked-FMA seam. The full-width masked-memory
  contract is preserved for every backend, including blend-based AVX2; only live
  tail elements are copied back to the caller. Non-dyadic f32 coverage records
  the documented tolerance for fused-operation rounding.
- [patch] Dense dot-product tails now use initialized provider-local lane buffers
  and masked FMA before the final horizontal reduction. This removes the scalar
  remainder loop without widening caller pointers; odd non-dyadic f32 coverage
  records the expected fused-rounding tolerance.
- [patch] Enabled crates.io publication for the five reusable workspace crates,
  using the registry package identities `mnemosyne-memory` and
  `themis-topology` while preserving their Rust-facing dependency names.
- [patch] Kept the core crate's cyclic workspace-only intrinsics test dependency
  path-only so Cargo excludes it from the registry archive.

### Fixed

- [patch] Bind the `themis` crate alias to the renamed `themis-topology`
  package so fresh Git dependency resolution follows the provider identity.
- [patch] Bind the `mnemosyne` crate alias to the renamed `mnemosyne-memory`
  package while preserving Rust imports.
- [patch] Align CI with the pinned Rust 1.97.0 toolchain and make the AArch64
  cross-check install its target explicitly before compilation, while retaining
  Rust 1.95 as the workspace MSRV contract.

- [patch] Corrected stale `"0.5.0"` version requirements in `hermes-simd-benches`
  and `hermes-simd-examples` path dependencies; the workspace is `0.4.1`, so
  these must be `"0.4.0"` for Cargo to resolve the workspace graph. Tests and
  benchmarks now discover their targets.

## [0.5.0] - 2026-07-24

### Added

- [minor] `SimdView::gather_into_uninit` and `prefix_scan_into_uninit` fill a
  caller's `&mut [MaybeUninit<T>]` and return the initialized prefix, and
  `AlignedVec::spare_capacity_mut` exposes the reserved-but-uninitialized tail.
  Together they let a routine fill freshly reserved capacity without a prior
  zero-fill. `gather`/`prefix_scan` (initialized-slice) now delegate to the
  uninit forms, keeping one implementation each.
- [minor] `SimdKernel::vector_to_mask` converts a comparison-result vector into
  the backend's native mask — the inverse of `mask_to_vector`. Composed with
  `mask_to_bitmask` it reduces any `cmp_*` result to one bit per lane, so a lane
  search resolves through `trailing_zeros` without leaving vector registers.
  Every backend implements it natively: a register rewrap on AVX2, a
  bit-preserving reinterpretation on NEON, `cmplt_epi{32,64}_mask` sign
  extraction on AVX-512 (AVX512F-only, since `movepi*_mask` needs AVX512DQ), and
  a `SIGN_MASK` bit test on the lane-emulated backends. Adding it to the sealed
  `SimdKernel` breaks no implementor: `cargo semver-checks` against `main`
  reports no required update.

### Breaking

- [major] `BitBoardKernel::rook_attacks`, `bishop_attacks`, and `queen_attacks`
  are now safe `fn`s. Their documented obligation — "caller must ensure target
  feature flags are active" — matched no implementation: four backends are plain
  integer arithmetic over bounds-checked tables, and `KoggeStone` selects its
  AVX-512/AVX2/NEON fill inside its own `is_x86_feature_detected!` guard. The
  `unsafe` now sits on those ISA fills, where the argument is real and local,
  instead of on every call. `cargo-semver-checks` classifies this as major
  (`trait_method_unsafe_removed`); see
  [ADR 011](docs/adr/011-bitboard-kernel-safe-surface.md). The methods also gain
  a `# Panics` section for an out-of-range square, their actual precondition.

### Migration

- Delete `unsafe` blocks wrapping `BitBoardKernel` calls; they now raise
  `unused_unsafe`, which is an error under `-D warnings`. Implementors of the
  trait drop `unsafe` from the three method signatures.

### Fixed

- [patch] `SparseView<Csr>::elementwise_mul_dense` could read out of bounds from
  safe code. It gathers `dense[col_indices[j]]` with an unchecked `Arch::gather`,
  but `SparseView<Csr>` is the *unvalidated* type — constructible from arbitrary
  `CsrData` via the public `from_csr` (and reachable through `SimdCow`) — so
  nothing guaranteed `col_indices[j] < dense.len()`. miri confirms the
  undefined behavior on an out-of-range column: "in-bounds pointer arithmetic
  failed: attempting to offset pointer by 36 bytes, but ... only 16 bytes from
  the end". The kernel now validates the CSR structure (`col_indices[k] <
  ncols`) and requires `dense.len() >= ncols` before the gather, matching the
  guards the SELL-p and Blocked-COO paths in the same file already had. The
  dense-with-mask path likewise gains the `dense.len() >= values.len()` assert
  its unchecked loads need. Both convert a reachable UB into a defined panic.
- [patch] `cmp_ne` reported NaN operands as *equal* on AVX2 and AVX-512. Those
  backends used the ordered `_CMP_NEQ_OQ` predicate, which yields false when
  either operand is NaN, while the trait documents Rust's `a != b` and both the
  scalar default and NEON (`vmvnq_u32 ∘ vceqq_f32`) return true. The x86
  backends now use the unordered `_CMP_NEQ_UQ`, the exact complement of the
  `_CMP_EQ_OQ` used by `cmp_eq`, making `cmp_ne` the lane-wise negation of
  `cmp_eq` on every backend. A cross-backend property test pins that complement
  with NaN-versus-NaN and NaN-versus-finite lanes; it reproduces the old
  behavior on AVX2 hardware, where the two NaN lanes were reported as equal.
- [patch] AVX-512 `blend` selected `false_val` for every active lane. It tested
  the mask by comparing it against zero, but an active lane carries `ALL_ONES` —
  a NaN bit pattern — which the ordered predicate rejects; `-0.0` was likewise
  misread, since it carries a sign bit yet compares equal to zero under any
  predicate. It now extracts the mask with `vector_to_mask`, matching the
  documented sign-bit contract and the AVX2 `blendv` behavior. Pinned by a
  cross-backend property test over canonical masks. Found while tracing the
  `cmp_ne` predicate, which `blend` shared; not reproducible on the development
  host, which lacks AVX-512.

- [patch] Safe code could execute an unsupported instruction set. A view or
  sparse/copy-on-write container named an `Arch` marker directly — `Avx512` on a
  host without AVX-512, say — and every operation on it called
  `#[target_feature]`-gated kernels, which is undefined behavior off that
  target. Reproduced as a hard `SIGILL`: constructing the view succeeded and the
  first reduction died with an illegal instruction, from a program containing no
  `unsafe`. `SimdView::new`/`new_mut` now return `None` for an architecture the
  host cannot execute — matching how they already reject bad alignment — and the
  eight `SparseView` constructors plus the owned `SimdCow` constructors assert
  the same condition, so holding one of these values is itself the proof that
  its kernels are callable. Runtime dispatch was never affected: it only ever
  selects a detected target. The probe caches its CPUID result, and an A/B on
  `dense/argmin_f32` measured no regression (−5.8% / −1.2% / −4.4% / −1.1%,
  i.e. within layout noise). Pinned by tests asserting that view availability
  tracks the platform feature probe for every marker.

### Changed

- [patch] `tiling` — the register-blocked `dot`, `gemv`, `gemv_transpose`, and
  `gemm` kernels gain module-level `# Safety` sections, and their fragmented
  single-call `unsafe {}` blocks (65 total, 11 documented) consolidate to 32,
  each carrying a `SAFETY` comment. The audit confirmed all four already validate
  their caller-supplied dimensions against the actual operand lengths — with
  arithmetic-overflow rejection (`checked_area`/`checked_strided_span`) closing
  the OOB path under release `overflow-checks = false` — before any unchecked
  access, so no soundness gap was found (unlike `sparse/ops`). The
  `dot`/`gemv`/`gemv_transpose` restructure is behavior-preserving code motion; a
  new differential test drives all four kernels through the `Scalar` backend
  under miri against scalar references, which also gives `gemm_register_tile`'s
  2D tile indexing its first miri coverage.
- [patch] `view::reduce` — the `reduce` and `zip_reduce` hot loops each wrapped
  every kernel call and `ptr.add` in its own `unsafe {}` block (66 total, 1
  documented). Each loop region is now one `unsafe` block with a `SAFETY` comment
  stating the pointer-bounds argument (offsets bounded by `unrolled_len`/
  `simd_len`, so every `LANE_COUNT` window stays within the slice); the two
  `reduce_popcount` families and the `load` closures gain the same. Blocks drop
  to 35. The `reduce`/`zip_reduce` restructure is behavior-preserving code motion,
  verified codegen-neutral: on `dense/sum_f32` and `dot_f32` the cross-version
  deltas fall within this host's measured run variance (`dot_f32/256` shows ±31%
  between successive runs of *identical* code). A new differential test drives
  `reduce`/`zip_reduce` through the `Scalar` backend under miri across lengths
  spanning the unrolled body, the vector tail, and the scalar remainder.
- [patch] `view::vector_reg` — the `Vector<T, Arch>` register wrapper gains a
  module-level `# Safety` section documenting its two disciplines (safe methods
  gate on `assert_runtime_supported`/`runtime_support_result` before the kernel
  call; the `pub unsafe fn` register loads/stores push both the target-feature
  requirement and pointer validity to the caller). The six `pub unsafe fn` docs,
  which stated only pointer validity, now also state the target-feature
  requirement. Per-site `SAFETY` comments cover the blocks with a further
  obligation — the `MaybeUninit` store-then-read in `Debug`/`PartialEq`/`to_array`/
  `to_bitmask`/`cast`/`extract`/`insert`, the compile-time lane-count/index guards
  behind `from_array`/`extract`, and the bounds asserts in the view-chunk
  loads/stores. Documentation only; no behavior change (SAFETY comments 6 to 23).
- [patch] `sparse::ops` — the elementwise-multiply and sum kernels each wrapped
  every target-feature call in its own `unsafe {}` block (35 total, none
  documented). Each kernel region is now one `unsafe` block with a `SAFETY`
  comment stating its bounds argument; blocks drop to 7. A differential miri
  test drives CSR, dense-with-mask, and SELL-p through the `Scalar` backend
  against a dense reference, and asserts the new CSR guards reject out-of-range
  columns and short dense buffers.
- [patch] The `sparse::spmv` kernels — CSR, dense-with-mask, and Blocked-COO —
  each wrapped every target-feature kernel call in its own `unsafe {}` block,
  around thirty per file, only three of them documented. Each kernel's inner
  region is now one `unsafe` block carrying one `SAFETY` comment that states the
  bounds and provenance argument (`Validated<...>` proves `col_indices[k] <
  ncols`, `validate_spmv_sizes` gives `x.len() >= ncols`, and the windowed
  offsets stay within `values`/`col_indices`). `sellp_spmv_vectorized`, an
  `unsafe fn` that had no `# Safety` doc at all, now documents its contract. The
  dense-with-mask kernel additionally hoists the loop-invariant masked-off fill
  vector out of its unrolled loop. Behavior is unchanged — the value-semantic
  integration tests pass and a new differential test drives all four formats
  through the `Scalar` backend under miri, checked against a dense reference, so
  the restructured pointer arithmetic is exercised by the interpreter.
- [patch] The owning `SimdCow::gather` and `prefix_scan` fill reserved capacity
  through the new `*_into_uninit` view methods instead of zeroing it first. The
  zero-fill (introduced with the HS-407 soundness fix) cost 12-59% on this
  path — benchmarked with a new `cow_f32` group in the dense suite — and the
  uninit fill removes it: `gather` -18%/-5%/-12%/-11% and `prefix_scan`
  -6%/-9%/-10%/-31% across 256/1024/4096/16384 elements versus the zeroing
  version, back to pre-HS-407 throughput with the soundness kept. The
  short-lived `AlignedVec::with_capacity_zeroed`, added and superseded within
  this Unreleased cycle, is removed.
- [patch] The copy-on-write constructors no longer form a `&mut [T]` over
  uninitialized elements. They reserved capacity, raised the length, then handed
  the buffer out as a slice while its tail was still unwritten — every element
  was initialized before anything read it, but that reference is not one the
  language permits, and `AlignedVec` allocates with `alloc`, not `alloc_zeroed`.
  `map_cow`, `fma_cow`, `splat_fill`, and the scalar-broadcast kernel behind
  `add`/`sub`/`mul`/`div_scalar_cow` now write their tail through the same raw
  pointer as their vector body and raise the length only once every element is
  written. That removes work rather than adding it: the tail stores are no
  longer bounds-checked. `gather` and `prefix_scan` fill their reserved capacity
  directly through the view's new `gather_into_uninit` / `prefix_scan_into_uninit`
  methods over `AlignedVec::spare_capacity_mut`, so they too never zero the
  buffer. Covered by a test that runs every constructor under miri across
  lengths straddling the vector body, where a missed element surfaces as an
  uninitialized read.
- [patch] `SimdCow::map_unary` and `SimdCow::map_cow` were the same operation
  implemented twice — one delegating to the view kernel, one with its own SIMD
  loop. `map_unary` now delegates to `map_cow`, leaving a single implementation.

- [patch] The unsafe-block audit continues (HS-406): `bitboard.rs` is fully
  documented, having gone from seven `unsafe` blocks to two — the raw pointer
  reborrows, each now carrying a `SAFETY` comment deriving the aliasing and
  lifetime argument from the view's reference typestate. The six `cow` modules
  gain module-level `# Safety` sections and per-site comments on the
  `with_capacity`/`set_len` buffers.

- [patch] The arch-generic modules (`view::reduce`, `view::ops`, `sparse::spmv`,
  `sparse::ops`, `iter::chunks`, `iter::zip`) document the target-feature
  obligation once, as a module-level `# Safety` section, now that construction
  enforces it. Per-site `SAFETY` comments are reserved for the obligations that
  go beyond it — pointer provenance, bounds, and alignment — rather than
  restating the shared invariant at every call.

- [patch] `argmin`/`argmax` locate their extremum with a vectorized scan instead
  of a scalar pass. Each vector tests NaN with `cmp_eq(v, v)` and matches the
  extremum with `cmp_eq(v, target)`, reducing both to bitmasks and taking the
  first hit with `trailing_zeros`. Measured on `dense/argmin_f32` (quiescent
  host, zero competing builds, p < 0.05): **−88.1% (256) / −91.8% (1024) /
  −92.7% (4096) / −92.9% (16384)**, an 8.4× to 14.1× speedup, lifting the scan
  from ~0.93 Gelem/s to ~12.8 Gelem/s. The NaN test used `cmp_eq(v, v)` to avoid
  `cmp_ne`, whose NaN result differs between the scalar default and the
  `_CMP_NEQ_OQ` hardware backends, since fixed under HS-404. Behavior is
  unchanged: the rejection, first-occurrence, and signed-zero contracts hold,
  now also covered at lengths that exercise the vector body, not only the tail.

- [patch] The AVX-512 VNNI signed-int8 GEMM tile now uses the ISA-supported
  unsigned-byte × signed-byte dot product with exact 128-bias correction. This
  removes the unsupported signed-byte ZMM instruction that raised `SIGILL` on
  AVX-512 VNNI hosts while preserving bitwise wrapping-`i32` semantics.
- [patch] `argmin` and `argmax` now reject every NaN-containing input and return
  the first matching slice element. This makes
  NaN and signed-zero behavior identical across scalar and SIMD backends.
  CI also smoke-runs each workspace Criterion binary under 60 seconds and
  provides a dispatchable full-suite job that bounds each binary at 300
  seconds after precompilation. The 48-ID dense suite retains every regime and
  uses flat sampling at Criterion's 10-sample floor with explicit 100 ms
  warm-up and 500 ms measurement budgets.
- [patch] SELL-p SpMV's scalar fallback (taken when `Arch::LANE_COUNT != C`) drops
  its dead `if c_idx < x.len()` guard and gathers `x[c_idx]` unchecked, matching
  the vectorized path — both rest on the `Validated<SellP>` invariant
  (`col_indices[k] < ncols`) plus `validate_spmv_sizes` (`x.len() >= ncols`).
  Eliminating a branch *and* a bounds check per entry: **−19% (rows_1024/1.0%) /
  −30% (rows_1024/10%) / −37% (rows_10000/0.1%) / −24% (rows_10000/1.0%)**
  (`sparse_bench` sellp4; measured against a quieter baseline, so understated).
  Results are unchanged — the guard was unreachable under the invariant, and had
  it ever been false it would have *silently dropped* the term rather than
  surfacing the violation, so this also removes a silent-wrong-answer path.
  Covered by the SellP differential (scalar vs vectorized) and property tests;
  all 38 sparse tests pass. Completes the scalar-tail work begun in 0.4.1's CSR
  fix. Miri cannot execute the SIMD-bearing module, so the `unsafe` rests on the
  SAFETY proof plus those tests.

## [0.4.1] - 2026-07-21

### Changed

- [patch] Refresh the locked numeric provider to Eunomia 0.6 after its
  production raw-half trait surface was retired; Hermes continues to use the
  native Eunomia reduced-precision vocabulary.
- [patch] CSR SpMV's scalar remainder loop now gathers `x[col]` unchecked
  (`get_unchecked`), matching the SIMD body's `Arch::gather` and the SellP
  vectorized path, which already trust the same `Validated<Csr>` invariant
  (`col < ncols`) plus `validate_spmv_sizes` (`x.len() >= ncols`). Rows with
  `nnz < LANE_COUNT` run *entirely* through this tail, so the previously
  inconsistent bounds-check + panic branch cost every nonzero. Fully-scalar
  short-row CSR SpMV (`sparse_bench`, quiet host, 1 nnz/row): **−20% (1024 rows)
  / −25% (10000 rows)**. Results are unchanged (same gathered value, no reorder);
  all 38 sparse tests pass. Miri cannot execute the SIMD-bearing kernel, so the
  `unsafe` is covered by the SAFETY proof (strictly weaker than the existing
  vectorized gather) and the short-row tests rather than miri.

## [0.4.0] - 2026-07-18

### Changed
- [arch] Advance Eunomia to 0.5.0 and replace raw `half::f16`/`half::bf16`
  throughout scalar, F16C, AVX-512, NEON, AMX, tiled GEMM, tests, and benchmarks
  with `eunomia::F16`/`Bf16`/`F32`. Remove every direct Hermes `half`
  dependency and delete the duplicate raw-half AMX and tiled-GEMM families.
- Resolve Themis, Mnemosyne, and Eunomia from their default branches and remove
  workspace-local patch overrides so downstream Atlas consumers share provider
  identities. The supply-chain allowlist now names those reviewed Git sources,
  and CI no longer checks out redundant sibling repositories.

### Fixed
- Bind adaptive tile dispatch to the operand type's AMX and AVX-512 capability
  probes so an int8 GEMM cannot select AVX-512 VNNI from the unrelated Bf16
  feature set.
- Size masked-gather fixtures from their lane-derived maximum index, covering
  the 64-lane AVX-512 int8 case without out-of-bounds oracle access.

### Migration
- Replace raw `half::f16`/`half::bf16` kernel and tiled-GEMM operands with
  `eunomia::F16`/`eunomia::Bf16`; use `eunomia::F32` for the typed Bf16
  accumulation surface.

### Breaking
- `hermes-simd-intrinsics`/`hermes-simd` [arch]: remove SIMD and tiled-GEMM
  implementations specialized for raw `half` types. Reduced-precision callers
  now use the Eunomia numeric vocabulary.
- `hermes-simd-core`/`hermes-simd-intrinsics` [minor]: `SimdArch` now requires
  `is_runtime_supported()`, the SSOT runtime probe used by safe vector/mask
  wrappers and forced target dispatch. `AmxSession::new` and
  `AmxBatchSession::begin` now return `Result<_, AmxSessionError>` so safe code
  cannot enter `ldtilecfg` on unsupported or OS-disabled AMX hosts.
- `hermes-simd-core`/`hermes-simd` [minor]: remove Hermes' public
  `NumaTopologyService`, `numa_node_count`, and `numa_node_distance` facades.
  Consumers that need node counts, distances, processor maps, or current
  topology snapshots must use `themis::CpuTopology` and Themis current-locality
  queries directly. Hermes still exposes SIMD-local `current_numa_node`,
  `refresh_numa_node`, `verify_numa_locality`, `NumaBinding`,
  `NumaAllocator`, and `MnemosyneNumaAllocator`.
- `hermes-simd-core`/`hermes-simd` [minor]: CSR, SELL-p, and Blocked-COO SpMV
  now require `ValidatedData` sparse storage. Build validated storage with
  `ValidatedData::new(...)`, `SparseView::<_, Validated<_>, _>::try_from_*`, or
  `SparseCow::<_, Validated<_>, _>::try_borrowed`/validated `from_slices`.
  Malformed sparse structures fail at construction instead of being rescanned
  inside hot `spmv` calls.

### Changed
- `hermes-simd-core`/`hermes-simd-intrinsics` [minor]: safe vector and mask
  wrappers check `SimdArch::is_runtime_supported()` before executing
  target-feature kernels. Fallible vector constructors (`try_zero`, `try_splat`,
  `try_from_array`) and checked slice wrappers return
  `SimdError::UnsupportedTarget` on unsupported AVX-512 hosts before any
  AVX-512 instruction; infallible vector conveniences panic before ISA
  execution. AMX `release` now avoids `tilerelease` unless a supported active
  session exists. Regression tests cover unsupported-host AVX-512 constructor
  rejection and AMX session rejection.
- `hermes-simd-core` [patch]: remove the direct `libnuma`
  `numa_alloc_onnode`/`numa_free` and Windows `VirtualAllocExNuma` allocation
  branches from `MnemosyneNumaAllocator`. Hermes now uses explicit affinity
  binding plus Mnemosyne/the configured allocator path for node-associated
  allocation instead of owning platform allocation fallbacks.
- `hermes-simd-core` [minor]: added the sparse `Validated<F>` typestate and
  `ValidatedData<S>` wrapper. `SparseSpMv` is implemented for validated
  CSR/SELL-p/Blocked-COO views only, so repeated solver calls trust
  construction-time validation and avoid per-call structural index scans.

### Added
- `hermes-simd-intrinsics`/`hermes-simd` [minor]: **256-bit AVX-VNNI int8 tile
  GEMM backend** (`AvxVnni` arch marker + `x86_64/avx_vnni_tiling.rs`). Client
  CPUs (Intel Alder Lake+, AMD Zen 5) have VEX-encoded `vpdpbusd` on YMM
  registers but no AVX-512, so int8 GEMM previously fell all the way to scalar
  tiles on that hardware. The new 16×16×64 kernel slots into the dispatch ladder
  as AMX → AVX-512 VNNI → **AVX-VNNI** → scalar (new `DispatchDecision::AvxVnni`
  variant; probe = `is_x86_feature_detected!("avxvnni")`, cached). Base AVX-VNNI
  has no signed-signed `vpdpbssd` (that is `avxvnniint8`), so the kernel computes
  signed×signed exactly via the unsigned-signed instruction with a bias
  identity — `Σ a·b = Σ (a XOR 0x80)·b − 128·Σ b`, the correction accumulated
  in-register per column group (`vpdpbusd(bias, splat(0x80), b_vec)`) and
  subtracted once after the K loop; wrapping i32 semantics make the identity
  exact mod 2^32, so results are **bitwise-equal** to the wrapping scalar
  reference. Register-blocked 2×(8 rows × 8 cols): 8 accumulators + bias + 
  operands fit the 16 YMM registers spill-free. Verified: kernel-level
  differential tests (full-range signed tile incl. −128 wraparound extremes,
  exact equality) + an end-to-end dispatched-GEMM differential on a
  non-multiple shape (37×29×130) vs an independent scalar triple loop — all
  executed on real `avxvnni` hardware. The int8 GEMM dispatch body and its
  4×-copy-pasted scalar remainder consolidated into one shared
  `gemm_i8_dispatched`/`gemm_i8_remainder` (the `I8`/`I32` newtype impl now
  delegates via `#[repr(transparent)]` casts); bench gains a forced
  `avx_vnni_tiles` row via a backend-generic `forced_backend_int8_gemm` helper.
  Measured (criterion, 100 samples, avxvnni client host): 64³ scalar tiles
  53.98 µs → AVX-VNNI 3.12 µs (**17.3×**, 84.1 Gelem/s); 128³ 437.1 µs →
  21.6 µs (**20.2×**, 97.1 Gelem/s); dispatched `gemm::<i8,i8,i32>` 22.7 µs at
  128³ (was scalar-routed before this change).
- `hermes-numeric` [minor]: `NumericElement` and `CastFrom` coverage for `i64`
  and `u8`/`u16`/`u32`/`u64`, plus the crate's first test module — value-semantic
  contract tests for every integer impl cross-checked against std (bitops,
  popcount, wrapping fmadd, min/max, constants, `CastFrom` round-trips).

### Added
- `hermes-simd-core`/`hermes-simd-intrinsics` [minor]: **non-temporal
  (streaming) stores for out-of-LLC elementwise writes.** New `SimdKernel`
  seam — `SUPPORTS_NT_STORE` (const gate, default `false`), `store_streaming`
  (default = `store_aligned`; x86 f32/f64 override to `vmovntps`/`vmovntpd` via
  the codegen template), and `stream_write_barrier` (default no-op; x86 =
  `sfence`). `SimdView::zip_into` (the elementwise `Add`/`Sub`/`Mul`/`Div` SSOT)
  routes outputs ≥ `NT_STORE_MIN_BYTES` (8 MiB, past every consumer L2) through
  a prefix-peeled-to-alignment streaming path that bypasses the cache, avoiding
  the read-for-ownership traffic a normal write-allocate pays. The store
  instruction is the only change, so results are byte-identical to the regular
  path — verified by a facade differential at 8.4 MiB with a mis-aligned output
  (forces the peel head) asserting `to_bits()` equality. Motivated and gated by
  the measured 1.71× (`streaming_bench`: 18.3 → 31.3 GiB/s on 64 MiB AVX2 f32).
  Backends without a non-temporal store (NEON, scalar, integer/half) inherit the
  safe default and never take the path.
- `hermes-simd-core` [minor]: `AlignedVec::reserve(additional)` and
  `extend_from_slice(&[T])` (`T: Copy`). `reserve` grows to at least the request
  (never below doubling, preserving geometric-growth amortization) in a single
  reallocation via the new SSOT `grow_to(new_cap)` that `grow`/`reserve` share;
  `extend_from_slice` is one reserve + one `copy_nonoverlapping`. `Extend for
  SimdCow` now reserves the iterator's `size_hint().0` up front, replacing a
  push loop's ⌈log₂ n⌉ reallocations (each copying the live prefix) with one —
  the `AlignedVec` growth-churn gap flagged in the memory audit. Verified:
  reserve satisfies-request + pointer-stable-within-capacity + no-op-when-
  sufficient, extend_from_slice value/empty/pre-sized/ZST paths.

### Changed
- `hermes-simd-core` [patch]: `SimdView::compress` no longer re-zeroes a
  `[T::ZERO; 64]` scratch buffer on every chunk. The store writes `lane_count`
  lanes and the copy reads only `pop ≤ lane_count`, so no lane is read before the
  store initializes it — the buffer is now a single `MaybeUninit<T>` array
  hoisted out of the loop (sized `MAX_SIMD_LANES` with the `LANE_BOUND_CHECK`
  compile-time guard), removing 256–512 B of per-chunk zero-init stores from the
  hot compaction loop. The loop-invariant `mask.popcount()` is hoisted too.
  Behavior unchanged (verified by the existing compress tests). A focused
  `SimdView compress` Criterion group now records scalar and host-AVX2
  all/half/quarter-mask rows at 1K, 16K, and 256K elements in
  `benchmarks_baseline.json` / `benchmarks_results.md`.
- `hermes-simd-core` [patch]: consolidate `reduce_popcount_{and,or,xor}` — three
  byte-identical ~104-line 4-accumulator popcount reductions differing only in
  the bitwise combining op — into one generic `reduce_popcount_op<Op:
  ElementOp<T>>` plus three thin wrappers passing the existing `BitAnd`/`BitOr`/
  `BitXor` ZST markers. The op monomorphizes away (zero-cost), so codegen is
  unchanged; `view/reduce.rs` drops 153 lines and the canonical-implementation
  rule is satisfied. Verified by the existing popcount tests.
- `hermes-simd-benches` [patch]: extend the `Tiled GEMM f32` bench to 256³–768³
  and report throughput as FLOPs (`2·size³`) rather than output elements. Closes
  the "GEMM benched only to n=64" coverage gap and delivers a **measured negative
  result** that retires the audit's KC-cache-blocking item: throughput is
  flat-to-rising with `k` (256³ = 78.4, 512³ = 69.8, 768³ = 79.9, 1024³ = 85.6
  GFLOP/s ≈ 67% of AVX2 f32 peak), so the packed `k × block_n` B panel spilling
  L1d does **not** degrade large-`k` GEMM — the current full-panel-pack +
  L2-residency design is correct for this microarchitecture, and the 512³ dip is
  a power-of-two cache-conflict artifact (768³ recovers). BLIS KC-blocking is
  rejected as fixing a non-problem; the 256/512/768 rows stay as the
  scaling-regression gate. See gap_audit round 8.
- `hermes-simd-core`/`hermes-simd-benches` [patch]: **masked-vector GEMV column
  tail** + the workspace's first GEMV benchmark (`gemv_bench.rs`, tail-isolating:
  each size pairs a tail-free `ncols` with a tail-having one at matched scale).
  The `ncols % LANE_COUNT` trailing columns of `gemv_strided_impl` (both the
  `TILE_M`-blocked and row-remainder paths) ran a per-row scalar loop; they now
  fold into the same vector accumulator via one masked fmadd — `x`'s tail lanes
  loaded once and reused across every row — replacing ~`lane_count−1` scalar
  ops/row with a single masked op and giving the tail the main loop's
  fused-multiply reduction. Inactive lanes load zero (`a·0` contributes
  nothing). Bench-gated per its Definition of Ready: at cache-resident 256×256
  (compute-visible) the 7-lane-tail row measured **3.58 µs → 2.83 µs (+27%
  throughput**, 18.2 → 23.1 Gelem/s), the tail-free row unchanged within
  run-to-run noise (2.4–2.6 µs); the DRAM 3000×1504 rows are bandwidth-bound and
  tail-neutral (retained as regression rows). Reduction order shifts (tail folded
  into the lane reduction vs a trailing scalar add), within the documented
  backend reduction-order envelope; verified by a new f32 facade differential
  (`n=21` = two full groups + 5-lane tail, `nrows=11` covering blocked + remainder
  rows, dyadic-exact ⇒ bitwise-equal) plus the existing dyadic f64 tail-shape
  suite.
- `hermes-simd-core` [patch]: **masked-vector GEMM column tails.** The
  `n % (TILE_N·LANE_COUNT)` trailing columns of `tiled_gemm` ran a strided
  scalar triple loop — up to `block_n − 1` columns (≈ half the FLOPs for `n`
  just under a block multiple, e.g. 31 of 63 on AVX2 f32). They now run the
  same fmadd contraction as the register tiles through `leading_k_mask`-guarded
  lane groups (`masked_load` → fmadd over k, B row-loads reused across a
  `TILE_M` row block → `masked_store`); inactive lanes load zero, accumulate
  `a·0`, and are excluded from the store, so Theorem 1's exactly-once cell
  coverage is preserved (proof text updated). Tail columns thereby also gain
  the tiles' fused-multiply rounding instead of separate mul+add. Verified by a
  new bitwise differential at `m=7, n=45, k=13` (one full block + one full
  masked group + a 5-lane partial group, dyadic-exact operands) plus the
  existing suite. Measured (criterion, AVX2 f32, new `n=63` bench row): 25.17 µs
  → 7.33 µs (**3.43×**, 9.9 → 34.1 Gelem/s, change −72%, p=0.00); full-block
  sizes are structurally unaffected.
- `hermes-simd-core` [patch]: `SimdCow::scale` now delegates to the fused
  `mul_scalar_cow` broadcast kernel. The previous body copied the buffer
  (`from_slice`) and then rescaled in place — two full read+write passes (4n
  element traffic) for a result bitwise-identical to the single fused pass
  (2n). Consolidates two parallel implementations of the same operation onto
  the `broadcast_op` SSOT.
- `hermes-simd` [patch]: **measured negative result** — chunk-width-aware
  SELL-p/BCOO dispatch (routing to the widest ISA whose `LANE_COUNT` matches
  `C`/`BN`) was implemented, A/B-benchmarked, and **reverted**: on an AVX2 host
  the `sellp4` 100k-row/10%-density case ran 2.4× slower via the lane-matched
  4-lane kernel (17.6 ms) than via the existing widest-first path (7.5 ms),
  because the "scalar fallback" loop auto-vectorizes at full width inside the
  AVX2 `#[target_feature]` dispatch helper. The widest-first ladder stands; the
  AVX-512-host variant of the original finding stays open in gap_audit.md,
  gated on AVX-512 hardware for its A/B. A dispatcher-independent SELL-8
  multislice differential test (non-uniform values, per-slice padding, dense
  reference) is kept from the experiment.
- `hermes-simd-intrinsics` [patch]: **F16C hardware-conversion arithmetic core
  for the AVX2 `f16` kernel.** The 16-lane `f16` kernel's `add`/`sub`/`mul`/
  `fmadd` performed per-element software f16↔f32 conversion (measured: `dot`
  ~220 Melem/s, ~90× below f32-class throughput). `half::f16` arithmetic is
  definitionally convert→f32-op→round-back per operation, and F16C
  (`vcvtph2ps`/`vcvtps2ph`, round-to-nearest-even) performs the identical IEEE
  conversions in hardware — so the upgraded methods (two 8-lane converts → AVX
  op → convert back) are **bitwise-equal** to the software path on all numeric
  values; NaN payloads follow the hardware quieting convention like every
  native backend. Each method gates on a cached
  `is_x86_feature_detected!("f16c") && ("fma")` probe (compile-time `cfg!`
  under `no_std`) and keeps the per-lane software loop as the documented
  fallback, so an AVX2-without-F16C host stays sound. Verified by new
  differential tests over an adversarial lane corpus (subnormals,
  overflow→inf, round-to-even ties, ±0, mixed signs — exact bit equality) plus
  NaN propagation, executed on F16C hardware. Loads/stores/masks/gather stay
  conversion-free array form. Measured (criterion, f16c host): `dot::<f16>`
  221 Melem/s → 7.22 Gelem/s at 16 Ki (**31.7×**, criterion change +3074%,
  p=0.00; 9.2× at n=256 where the software `sum_reduce` tail weighs more).
  bf16 measured separately at ~2 Gelem/s (shift-conversion partially
  auto-vectorizes); a bf16 hardware core is not justified until a consumer
  needs more.
- `hermes-simd-benches` [patch]: dense sum/dot benches gain `i32` groups and the
  four per-type scalar baselines consolidate into two generic `scalar_sum`/
  `scalar_dot`. The integer rows are the evidence gate for the "emulated
  integer kernels rely on auto-vectorization, unverified" audit finding — and
  the verdict is a **verified negative result**: inside the
  `#[target_feature(enable="avx2,fma")]` dispatch wrappers LLVM fully
  auto-vectorizes the `[i32; 8]` emulated kernels (sum 50–62 Gelem/s vs 4.5
  scalar ≈ 12×, L1-resident; dot 20.3 Gelem/s at 16 Ki ≈ 7.4×,
  memory-bandwidth-bound). Hand-written AVX2 integer `SimdKernel` impls for the
  dense op families are therefore rejected as duplication with no measurable
  win; the bench rows stand as the regression gate for that conclusion.
  (Residual: emulated `gather`/`compress`/mask ops compile to scalar loops
  auto-vectorization cannot rescue — revisit only when a sparse-integer
  consumer exists.)
- `hermes-simd` [patch]: conservatively disable AMX auto-dispatch by reporting
  no AMX support until the crate has a stable, permission-aware probe for
  hardware bits, XCR0 OS state, and Linux XTILEDATA process permission. This
  removes unstable Rust AMX feature-detection macro usage and avoids CPUID-only
  dispatch risk. AVX-512 tile probes remain exact stable
  `is_x86_feature_detected!` checks. Evidence: `cargo check -p hermes-simd` and
  `cargo clippy -p hermes-simd --all-targets -- -D warnings` pass.
- `hermes-simd-core` [minor]: give the six masked-merge `SimdKernel` methods
  (`masked_load_unaligned`, `masked_store_unaligned`, `masked_add`, `masked_mul`,
  `masked_fmadd`, `masked_sum_reduce`) scalar-emulated trait defaults — the
  arithmetic via `blend(mask_to_vector(mask), …)` and the load/store via new
  `kernel_helpers::generic_masked_{load,store}` — so a new backend or scalar type
  inherits the tail-masked family for free instead of hand-implementing it. These
  were the last capability family still `required` on every impl (rsqrt, popcount,
  horizontal-bitwise, reductions, and scans were already defaulted), the one
  paying an N-impl tax that gated cheap backend/type expansion. The six redundant
  hand-written impls are removed from `impl_emulated_kernel!` (~66 lines, inherited
  free by all ~24 emulated backends); native AVX2/AVX-512/NEON overrides are
  unchanged. Behavior is bit-identical to the removed per-element loops, verified
  by a new cross-backend differential property test (Scalar/SveArch defaults vs
  AVX2/AVX-512 native overrides). `gather`/`compress`/`expand` stay `required`
  (no generic `IndexVector`/lane-introspection primitive to default them).
- `hermes-simd` [patch]: extract `axpy_rows_batch`'s type-independent extent
  validation into a non-generic `#[inline(never)]` `check_axpy_rows_batch_extents`
  so it is emitted once instead of re-monomorphized into every `(T, Arch)`
  instantiation of the kernel (the validation runs once per call, not in the hot
  loop, so the dedup has no hot-path cost).
- `hermes-numeric` [patch]: consolidate the signed-integer `NumericElement`
  impls (`i8`/`i16`/`i32`/`i64` were four hand-copied blocks) into one
  `impl_numeric_element_signed!` macro mirroring the unsigned one, and drop the
  `min_scalar`/`max_scalar` overrides from every integer impl — they were
  byte-for-byte identical to the `PartialOrd`-based trait defaults (the float/half
  overrides stay for NaN semantics). Net ~275 fewer lines, behavior unchanged
  (verified by the existing integer-contract tests).
- `hermes-simd-core` [patch]: finish the `MAX_SIMD_LANES` SSOT migration in
  `view/vector_reg.rs` — the `Vector` scalar-fallback buffers (`Debug`,
  `PartialEq`, `to_bitmask`, `cast`, `extract`, `insert`, masked slice load/store)
  were still hardcoded `[_; 128]` with dead `assert!(lane_count <= 128)` runtime
  checks. They now use the named `MAX_SIMD_LANES` (64) const with the compile-time
  `LANE_BOUND_CHECK`, halving those stack frames and converting the dead runtime
  asserts into per-backend compile errors; the masked-slice OOB guard's magic `64`
  is now `u64::BITS`.
- `hermes-simd-core` [patch]: split the 601-line `tensor/view.rs` into a vertical
  `tensor/view/` hierarchy by concern — `mod.rs` (core N-D struct, constructors,
  rank-agnostic accessors), `rank_ops.rs` (rank-2/3 specialized views + transpose),
  and `simd_bridge.rs` (the rank-1 → `SimdView` seam). Pure relocation; behavior
  unchanged.
- `hermes-simd-core` [patch]: drop `adjust_layout_for_mnemosyne`, the small-alloc
  padding that inflated every `<=8KB` NUMA allocation to `8192+align` bytes to
  "bypass the thread-local cache". That routed small allocations into Mnemosyne's
  ~2 MiB-per-allocation huge path; the small thread-cache path is correct,
  NUMA-partitioned, and bounded. Combined with the Mnemosyne alignment-aware
  small-path fix (`Mnemosyne perf/aligned-small-alloc-tcache`), 512 live
  256-byte/64-aligned `AlignedVec` allocations drop from ~1056 MiB to ~4 MiB
  mapped (measured). Also removed the no-op NUMA thread bind in `dealloc_on_node`
  (a free routes by the pointer's owning segment, not the caller's node).
- `hermes-simd-core` [patch]: encode the scalar-fallback stack-buffer lane bound
  at compile time and tighten it to the true maximum. The default `SimdKernel`
  methods (`scan_vector`, `swap_adjacent`, `dup_even`/`dup_odd`) and the
  `kernel_helpers` scalar emulations store a full vector into a fixed
  `[MaybeUninit<T>; N]` stack buffer, so a backend whose `LANE_COUNT > N` would
  silently overflow it (UB). `N` is the named SSOT constant `MAX_SIMD_LANES`,
  now `64` (the workspace maximum, AVX-512 `i8`) rather than the previous
  over-provisioned `128` — halving every fallback frame. A defaulted associated
  const `SimdKernel::LANE_BOUND_CHECK` (referenced via inline `const {}` in each
  buffer method) asserts `LANE_COUNT <= MAX_SIMD_LANES` per backend at
  monomorphization, turning a would-be silent overflow into a compile error.
  `reduction.rs::finalize` (formerly a divergent `MAX_LANE_COUNT = 64` + debug
  assert) and `generic_mask_from_bitmask`'s bitmask buffer now both fold onto
  this SSOT under the compile-time check.
- `hermes-simd` [patch]: `dispatch_axpy` and `dispatch_scale` use a 4-accumulator
  unrolled SIMD body to break the store-to-load dependency chain, matching the
  throughput model used by `dot`.
- `hermes-simd` [patch]: the three target-gated `impl SimdOps for T` blocks
  (byte-identical 206-line bodies differing only in their `where` kernel bound)
  collapse into one `impl_simd_ops_methods!` macro (`dispatch/mod.rs` 1217 → 845
  lines); the per-call element-width flush limit in `view/reduce.rs` dedupes to a
  single `const fn flush_limit_for::<T>()`.
- `hermes-simd-macros` [patch]: `#![forbid(unsafe_code)]` (the crate executes no
  unsafe; the unsafe it emits lives in generated token streams).
- `hermes-simd-intrinsics` [patch]: magic-table init CAS success ordering relaxed
  from `Acquire` to `Relaxed` (the 0→1 winner acquires no shared data).

### Fixed
- `hermes-simd` [patch]: **detection soundness** — AMX/AVX-512 tile-kernel
  dispatch used hand-rolled `__cpuid_count(7, _)` probes that (a) never checked
  XCR0/OSXSAVE, so a host advertising the CPUID bit without the OS enabling the
  XSAVE state (ZMM/opmask, TILECFG/TILEDATA) would `#UD`/`#NM` on the first wide
  instruction; (b) had no leaf-7 max-leaf guard, so on a pre-leaf-7 x86_64 CPU
  leaf-1 bits (FXSR/ACPI) aliased as AMX support; and (c) on Linux, AMX tile data
  is gated per-thread by XFD until `arch_prctl(ARCH_REQ_XCOMP_PERM,
  XFEATURE_XTILEDATA)`, which CPUID/XCR0 do not reflect. **AVX-512** bf16/vnni tile
  probes are now `is_x86_feature_detected!` for the exact set each kernel's
  `#[target_feature]` enables (`avx512f,avx512bw,avx512vl` and
  `avx512f,avx512vnni,avx512vl`) — the macro handles XCR0 and the max-leaf, and
  this also corrects the bf16 probe, which previously required the unrelated
  `avx512bf16` dot-product bit (a `#UD` window *and* a false skip on capable
  non-bf16 parts, since the kernel widens to f32 and never uses `dpbf16`).
  `widen_i8_to_i16`'s AVX-512 branch is now gated on `avx512bw` (the
  `_mm512_cvtepi8_epi16`/`vpmovsxbw` requirement) rather than `TargetId::Avx512`
  (`avx512f`-only), which would `#UD` on Knights Landing. **AMX** dispatch is
  disabled (probes return `false`) until a stable, permission-aware probe exists
  that verifies hardware + XCR0 + the Linux XTILEDATA `arch_prctl` — the stable
  toolchain does not accept the AMX feature strings in `is_x86_feature_detected!`
  (`x86_amx_intrinsics` is unstable), so returning `false` preserves the
  safe-dispatch contract instead of risking the fault. Restoring AMX behind a
  correct probe is filed (needs an AMX host to verify).
- `hermes-simd-core` [patch]: **memory safety** — SELL-p vectorized SpMV and
  `elementwise_mul_dense` read out of bounds from safe code. On the
  `Arch::LANE_COUNT == C` fast path, `sellp_spmv_vectorized` gathered
  `x[col_idx]` and loaded `values[offset..]` (and the elementwise path stored
  `out_values[offset..]`) at full vector width with no bounds check — unlike the
  sibling CSR/BlockedCoo paths, which scan their indices up front, and unlike the
  SELL-p scalar fallback, which guards per lane. Because `SellPMatrix` has `pub`
  fields, a no-op `new`, and an opt-in `validate()` the SpMV path never called, a
  caller could drive a safe `SparseView::<SellP<C>>::spmv` to read past `x`,
  `values`, or `out_values`. Fixed by validating the structure through the SSOT
  `SparseValidate::validate()` (via `spmv::assert_sellp_validated`) before the
  unsafe kernel — proving `col < ncols`, `col_indices.len() == values.len()`, and
  `slice_ptr[s] + slice_col_count[s]·C <= values.len()` — plus an
  `out_values.len() >= values.len()` guard on the elementwise store. Two
  `#[should_panic]` regressions exercise the Scalar-backed vectorized path
  (`Scalar::LANE_COUNT 4 == C`, host-independent) with an out-of-range column and
  with over-long slice geometry.
- `hermes-simd-intrinsics` [patch]: **numeric precision** — `recip_sqrt` (`1/√x`)
  gave reduced, backend-dependent accuracy on the SIMD f64 paths and NEON f32. All
  copied the f32 "hardware `rsqrt` seed + one Newton step" pattern, but one step
  only refines a ~8–14-bit seed to ~16–28 bits — fine for f32 from a ≥12-bit seed
  (x86), but far below f64's 52-bit mantissa and below f32's 23 bits from NEON's
  8-bit seed. The same `recip_sqrt::<f64>` thus ranged from ~1e-16 (scalar) to
  ~1.5e-5 (NEON), a native-precision violation, hidden by tests using perfect-square
  inputs (where Newton converges exactly by luck) and magic `1e-4`/`1e-6`
  tolerances. Now full native precision (~1 ulp) on every backend: f32 keeps the
  fast `rsqrt`+Newton (x86 one step; NEON two steps, 8→16→32 bits); f64 uses the
  correctly-rounded hardware `sqrt`+divide (x86 via codegen, NEON via `vdivq`/
  `vsqrtq`). Trait doc states the precision contract. Verified by a cross-backend
  differential test (Scalar/SveArch/AVX2/AVX-512, NEON on aarch64 CI) over
  non-perfect-square inputs with analytically-derived relative bounds (`8·ε_f32`,
  `4·ε_f64`); the old per-backend tests were de-gamed (non-trivial inputs, derived
  tolerances).
- `hermes-numeric` [patch]: integer `NumericElement::sqrt` computed
  `(self as f64).sqrt() as Self`, rounding operands above 2⁵³ to `f64` *before*
  taking the root — lossy for large `i64`/`u64` (e.g. `u64::MAX.sqrt()` returned
  4_294_967_296, whose square overflows `u64`; the correct floor root is
  4_294_967_295). Now uses exact integer `isqrt`; signed negatives keep the
  defined degenerate contract (return 0 — integers have no `NaN`). Trait doc states
  the integer/float/negative contract. Covered by new value-semantic tests: exact
  small roots for all eight integer types, the large-operand regression cases
  (`u64::MAX`, `i64::MAX`), the `r² ≤ n < (r+1)²` invariant above 2⁵³, and the
  negative-input contract.
- `hermes-simd-core` [patch]: **memory-safety** — the tiling GEMV/GEMM dimension
  checks computed the required operand span with unchecked `usize` arithmetic
  (`(nrows−1)·lda + ncols`, `m·k`, `k·n`, `m·n`) as the *sole* guard before
  `unsafe` SIMD loads/stores. An adversarial dimension reachable from the public
  dispatch API (e.g. `dispatch_gemv_strided(.., nrows=2, lda=usize::MAX)`)
  overflowed the product: under release `overflow-checks = false` it wrapped to a
  small value, the `a_len < a_needed` guard passed, and the kernel read out of
  bounds (and panicked undocumented in dev, where checks default on). Span math is
  now one SSOT module `tiling::dims` (`checked_strided_span`/`checked_area`,
  shared by the forward and transpose GEMV checkers — previously duplicated) that
  returns `SimdError::LengthMismatch` on overflow, closing the OOB path in every
  profile independently of `overflow-checks`; the checked bound also proves the
  kernels' own `row_idx·lda` index arithmetic cannot overflow. Added
  `[profile.dev] overflow-checks = true` (explicit per the numerical-discipline
  mandate; release keeps the default for hot-loop speed). Verified by exact-variant
  overflow regression tests on all three dispatchers, passing in **both** dev and
  release (the release pass is the proof the OOB load is unreachable), plus
  `tiling::dims` unit tests.
- `hermes-simd` [patch]: `spmv_bcoo` was hardcoded to `ScalarArch`, so the
  runtime-dispatched SIMD BlockedCoo kernels (and their bounds guards) were dead
  — every blocked-COO SpMV ran scalar regardless of host SIMD. It now routes
  through a `#[runtime_dispatch]` `dispatch_spmv_bcoo` like the CSR/SELL-P/
  dense-masked paths, selecting AVX-512/AVX2/NEON/scalar at runtime. Covered by a
  differential test exercising the SIMD branch against a scalar reference.
- `hermes-simd-core` [patch]: harden the NUMA alloc-generation cross-thread
  invalidation signal. The counter now publishes with `Release` and is read with
  `Acquire` (was `Relaxed`, which gave no happens-before, so a reader could trust
  a stale locality flag for a recycled address), and `verify_numa_locality`
  captures the generation once before the OS residency probe instead of
  re-reading it at store time — closing a TOCTOU window where a concurrent bump
  stamped pre-bump probe data with the post-bump generation.

### Safety
- `hermes-simd` [patch]: documented the `# Panics` contract (`square >= 64`) and
  added the `// SAFETY:` justification on the public `rook_attacks`/`bishop_attacks`/
  `queen_attacks` wrappers over the `Magic` `unsafe` kernel — verified the kernel
  uses bounds-checked table indexing (panics, never OOB), closing a round-1
  finding; backed by a `#[should_panic]` test.
- `hermes-simd-intrinsics` [patch]: the raw AMX tile wrappers
  (`tilezero`/`tileloadd`/`tilestored`/`tdpbf16ps`/`tdpbssd`) replaced their
  silent `_ => {}` fallthrough with `unreachable!` so an out-of-range tile index
  is a loud panic rather than a silently-dropped compute step; documented the
  AMX-availability precondition (CPU feature + OS tile-state enable) on
  `AmxGemm::amx_gemm`'s `# Safety` (it is reached only via the `has_amx()`-gated
  dispatch path).
- `hermes-simd-core` [patch]: CSR `spmv` now validates every column index is
  `< ncols` (linear pre-loop scan) before the unchecked SIMD gather `x[cols[j]]`,
  making the safe `spmv_csr` sound on malformed input (negative/oversized indices
  panic instead of reading out of bounds). The scan is cheap relative to the
  gather-bound kernel; covered by a `#[should_panic]` test.
- `hermes-simd-core` [patch]: the BlockedCoo `spmv` and `elementwise_mul_dense`
  kernels issued unchecked `load_unaligned` reads of `BN` lanes at each block's
  column base with no guarantee the span stayed within `x`/`dense`. Added an
  O(nblocks) pre-loop guard (every block's column span fits the input, row span
  the output) plus dense/output buffer-size checks, so a malformed block
  coordinate panics rather than reading out of bounds.
- `hermes-simd-core` [patch]: `build_index_vector` binds its `IndexVector` layout
  assumption with a `const` assert (`size_of::<IndexVector>() == LANE_COUNT *
  size_of::<i32>()`), so a layout-mismatched backend is a build error rather than
  an out-of-bounds unaligned read.

## [0.3.0] — 2026-06-21

### Added
- Public runtime-dispatched `gemv` (`y += A·x`, register-blocked level-2 BLAS
  matrix–vector product) plumbing the existing `TilingStrategy::gemv` /
  `gemv_impl` core through the `SimdOps` dispatch trait, in its own
  `dispatch/gemv.rs` leaf module. `TILE_M` row-blocking scales with the register
  file (8/4/1 by lane count); the operand-reuse theorem is documented inline.
  Value-semantic differential tests vs a scalar reference across shapes
  (incl. `TILE_M` remainder + column tail), accumulate semantics, and the
  length-mismatch error path; a `gemv_f32` Criterion benchmark vs a scalar
  row-by-row reference (measured ≈9× at 256² on the local AVX2 host).
- Public runtime-dispatched `gemv_transpose` (`y += Aᵀ·x`), the complement of
  `gemv`: a new register-blocked `gemv_transpose` core kernel
  (`tiling/gemv_transpose.rs` + `TilingStrategy::gemv_transpose`) plus the
  `dispatch/gemv_transpose.rs` leaf. Computes `Σᵢ xᵢ·A[i,:]` (sum of scaled rows),
  vectorizing across the `ncols` output lanes with **no horizontal reduction**;
  `TILE_N` blocks output lane-chunks for accumulator reuse across rows. Inline
  output-reuse theorem, value-semantic differential tests across shapes (incl.
  `TILE_N` remainder + column tail), accumulate semantics, error path; and a
  `gemv_transpose_f32` Criterion benchmark vs a scalar reference.
- Public runtime-dispatched `gemv_strided` (`y += A·x` over a row-major
  **sub-matrix** with leading dimension `lda ≥ ncols`). The core `gemv` kernel is
  generalized in place to `gemv_strided_impl(.., lda)` (DRY — packed `gemv` now
  delegates with `lda = ncols`, bit-for-bit unchanged, verified by a test), with
  `TilingStrategy::gemv_strided` and `dispatch/gemv_strided.rs`. Admits matvec
  over a trailing/leading block of a larger buffer (e.g. a reflector apply's
  column block) without copying it out. Differential test over a true sub-matrix
  (`lda > ncols`), a packed-equals-`gemv` equivalence test, an `lda < ncols` /
  short-span rejection test, and a `gemv_strided_f32` Criterion benchmark over a
  padded buffer (the gapped-row access path).
- Public runtime-dispatched `gemv_transpose_strided` (`y += Aᵀ·x` over a
  row-major sub-matrix, row stride `lda ≥ ncols`) — the transpose analogue of
  `gemv_strided`. The `gemv_transpose` core kernel is generalized in place to
  `gemv_transpose_strided_impl(.., lda)` (DRY — packed `gemv_transpose` delegates
  with `lda = ncols`, verified bit-for-bit equal by a test), with
  `TilingStrategy::gemv_transpose_strided` and a `dispatch/` leaf. Admits the
  `Aᵀ·x` reduction over a strided block (e.g. forming `Aw = Σⱼ wⱼ·colⱼ` in a
  reflector apply) without copying. Differential test over a sub-matrix, a
  packed-equals-`gemv_transpose` equivalence test, and an invalid-`lda` rejection.
- Generic absolute reductions: `AbsSum` / `AbsMax` reduction strategies plus
  runtime-dispatched `abs_sum` / `abs_max` APIs for Leto/Apollo norm paths,
  using transformed SIMD seeds and transform-free partial merges to avoid
  temporary absolute-value buffers.
- Criterion benchmark suite for the interleaved complex kernels (`benches/complex.rs`), runtime-dispatch vs `Scalar` series as a built-in regression reference.
- CI: `cargo miri test` job over hermes-simd-core (pointer/view/cow logic) and a `--no-default-features` build check.
- `#![deny(missing_docs)]` on all six public crates; remaining undocumented items (bitboard backends, AMX submodules, emulated-kernel macro, magic `OnceLock`) documented.
- Internal x86 VNNI asm instruction macro for `vpdpbssd`, keeping assembly behind the monomorphized tile-matmul backend contract without a hot-loop wrapper call.
- Benchmark reports now record benchmark-relevant host ISA features and the runtime dense-dispatch backend selected on the runner.
- Benchmark regression enforcement now uses `benchmarks_baseline.json` plus
  `run-benches --check-regressions`; the runner is split into CLI, Criterion
  parsing, host reporting, Markdown rendering, and threshold-check modules.
- Sparse SpMV benchmarks now sweep CSR, SELL-p, and Blocked-COO across 1K,
  10K, and 100K rows at 0.1%, 1%, and 10% non-zero density; Dense-with-mask is
  capped at 10K rows to bound local memory use.
- Packed4 COW unpacking now has a focused Criterion benchmark target over
  1K, 16K, and 256K logical elements for both public packed formats.
- Runnable doctests now cover the public complex kernels, sparse CSR
  `SparseCow` SpMV, and const-generic `TensorView` construction/access paths;
  `hermes-simd-core` doctests are enabled.
- Host-capability integration tests validate runtime dispatch, local AVX2 execution when available, and irregular-shape GEMM fallback coverage.
- Miri coverage now extends to the `hermes-simd-intrinsics` boundary: AMX session state is tested under Miri while hardware execution paths remain native-only.
- `parallel` and `mnemosyne-memory` are default features on every Hermes package; `mnemosyne-memory` routes `AlignedVec::with_capacity_numa` allocation and deallocation through Mnemosyne by default.
- `hermes-simd` re-exports `SveArch` with the other architecture markers, and
  the kernel property suite now exercises its mask, compress/expand, gather,
  and leading-tail invariants on every host.
- Core kernel, compute, and tiling Rustdoc examples now run as value-semantic
  doctests instead of compile-only `no_run` examples.
- `BitMask` native-mask conversion and active-lane iteration examples now run
  as value-semantic doctests.
- `TargetId`, `dispatch_view_to`, and `dispatch_view_mut_to` provide an
  explicit target-token surface for tests and benchmarks, rejecting unsupported
  targets before constructing architecture-specific views.
- `Vector<T, Arch>` now has safe one-vector slice wrappers for aligned and
  unaligned load/store, returning value-semantic `SimdError` variants for short
  or misaligned slices while preserving raw pointer kernels for hot loops.
- Host-capability tests now force every supported `TargetId` and compare dense
  facade sum, dot, elementwise arithmetic, gather, and select results against
  the scalar target.
- `axpy_rows_batch` adds one runtime-dispatched fused kernel for
  `out[row, i] += sum_k alphas[k, row] * x_panel[k, i]`, avoiding repeated
  facade dispatch when a consumer accumulates a dense row panel; the kernel
  accumulates each output lane across depth in registers and stores it once.
- Dense Criterion benchmarks now include `axpy_rows_batch_f32`, comparing the
  fused row-panel kernel against repeated public `axpy_rows` calls.
- Dense and AXPY length-mismatch tests now assert the exact
  `SimdError::LengthMismatch` contract instead of existence-only failures.
- Select, unary-map, and COW FMA error-path tests now assert exact
  `SimdError` variants for length and output-capacity failures.
- New operation, strategy, complex, gather, scan, and COW math error-path
  tests now assert exact `SimdError` variants for invalid shape, short output,
  and invalid index contracts.
- `SimdCow::map_unary` now asserts its internally constructed output-length
  invariant instead of silently discarding the impossible `map_unary` error.
- GEMM tiling module docs now avoid private intra-doc links, keeping workspace
  rustdoc warning-clean after the vertical tiling split.
- README/backlog now include an operation-family coverage map that distinguishes
  delivered SIMD families from consumer-demand pending families.
- Runtime FMA support probing now uses Rust's platform-aware feature detector
  behind a cached `has_fma3` helper and `FmaSupport` trait impls.
- GEMV dispatch docs now disambiguate function links from same-named modules,
  keeping rustdoc warning-clean.

### Changed
- `SveArch` is now a callable 512-bit-shape emulated backend for f32/f64
  (`16xf32`, `8xf64`) instead of a public marker with `unimplemented!()`
  kernel methods. Native SVE intrinsics remain a separate pending backend.
- Blocked-COO SpMV dispatch now uses one const-generic `spmv_bcoo::<T, BM, BN>`
  entry point, so tile shape monomorphizes from the call site instead of
  cloning fixed 4x4 and 8x8 public functions.
- SELL-p SpMV dispatch now uses one const-generic `spmv_sellp::<T, C>` entry
  point, preserving runtime architecture dispatch while removing fixed slice
  height functions.
- Interleaved complex `mul_assign` now processes two SIMD registers per loop
  iteration on SIMD backends and uses a direct four-pair scalar loop for large
  scalar inputs, reducing loop overhead in the measured complex benchmark
  range.
- `benchmarks_baseline.json` and `benchmarks_results.md` now include the
  packed4 COW unpack benchmark rows and refreshed complex `mul_assign`
  measurements from the local AVX2 host.

### Performance
- `SimdView::dot` middle SIMD loop now accumulates into the vector register via
  `fmadd` and reduces to scalar once at the end, instead of a horizontal
  `sum_reduce` per lane group. The per-group reduction serialized the loop on the
  ~5–7-cycle horizontal-reduction latency and dominated small/odd-length dot
  products (e.g. the length-`m−k` bidiagonal-SVD reflector applies in Leto). The
  unrolled head's vector accumulator now carries through the residual loop; only
  the final scalar tail reduces. Value-semantic (within the existing dot
  tolerance; 322 workspace tests green).

### Fixed
- `#[runtime_dispatch]` emitted `std::is_x86_feature_detected!` unconditionally, breaking `--no-default-features` builds; runtime-detection arms are now gated on the consuming crate's `std` feature (no_std keeps compile-time arms + scalar fallback).
- rkyv-exercising unit tests are ignored under Miri (rkyv 0.7 archived access violates Stacked Borrows inside the dependency); hermes's own unsafe passes Miri clean.
- INT4 unpack regression coverage now asserts the complete signed nibble domain.
- AMX context-pressure benchmarks no longer publish scalar fallback timings under an AMX-specific label on non-AMX hosts.
- Dense scalar benchmark baselines now black-box operands and accumulation so Criterion measures real iteration work instead of an optimized-away constant.
- Inline asm compute forms panic under Miri instead of returning fake values; Miri-valid AMX lifecycle operations are no-ops only for session-state verification.
- README now documents the Atlas SIMD/MIMD/GPU ownership boundary so consumers
  compose Hermes with Moirai and Hephaestus without duplicating responsibility.
- Packed4 COW unpacking delegates to the `Packable4` dispatcher, so the
  facade uses the existing AVX-512/AVX2/scalar runtime selection instead of an
  AVX-512-only x86 branch.
- README current-version metadata now reflects the released `0.2.0` workspace
  state.
- Added a Highway reference gap audit (`gap_audit.md`) and README baseline
  section, tracking Hermes-native follow-ups for target-token forced dispatch,
  safe slice wrappers, SSE2 feasibility, cross-target conformance, and
  operation-family coverage.
- `Cargo.lock` now matches the patched local Themis package version (`0.9.11`),
  keeping `cargo check --locked` coherent with the current Atlas checkout.

### Breaking
- Removed fixed Blocked-COO public dispatch functions `spmv_bcoo4x4` and
  `spmv_bcoo8x8`; use `spmv_bcoo::<T, BM, BN>`. Removed fixed
  `SparseView::from_blocked_coo_4x4` and `SparseView::from_blocked_coo_8x8`;
  use `SparseView::<T, BlockedCoo<BM, BN>, Arch>::from_blocked_coo`.
- Removed fixed SELL-p public dispatch functions `spmv_sellp4` and
  `spmv_sellp8`; use `spmv_sellp::<T, C>`. Removed fixed
  `SparseView::from_sellp4` and `SparseView::from_sellp8`; use
  `SparseView::<T, SellP<C>, Arch>::from_sellp`.

## [0.2.0] — 2026-06-10

### Added
- Interleaved complex kernels (`interleaved_complex_dot`, `interleaved_complex_mul_assign`, runtime variants) over `[re, im, ...]` primitive slices, generic over `T: Scalar` and architecture with `const CONJ_B` conjugation (ADR-004).
- `SimdKernel` adjacent-pair primitives: `swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd` — default scalar emulation plus AVX2, AVX-512, and NEON intrinsic overrides.
- `CowFormat` trait and generic `SparseCow<'a, T, F, Arch>` clone-on-write sparse container.
- `SimdView::prefix_scan_in_place` (vectorized, single authoritative scan implementation).
- `SimdOps::interleaved_complex_mul_assign` / `interleaved_complex_dot` trait methods.
- Property-test suites: complex kernels with analytically derived rounding tolerances; differential AVX2/AVX-512-vs-Scalar tests on dyadic-exact inputs; `f16`/`bf16` complex differential tests (bitwise for elementwise multiply, reordering-bound for dot); kernel-level mask/compress/expand/gather/`leading_k_mask` invariants per backend.
- `complex_dot` example with throughput comparison.
- Exact NTT butterfly stage kernel (`dispatch/modular.rs`) with integration tests.
- CI pipeline (GitHub Actions): fmt, clippy `-D warnings`, tests on x86_64 and aarch64 (runtime NEON validation), warning-clean docs, aarch64 cross-compile check, `cargo-deny` supply-chain gate.
- `rust-toolchain.toml` pin (1.95.0) and workspace MSRV declaration (`rust-version = "1.95"`, verified by full build + test on 1.95.0).
- `deny.toml`: permissive-license allowlist, yanked-crate denial, source restrictions.
- PM artifacts: `backlog.md`, `checklist.md`, this changelog; README refreshed to the current architecture.

### Changed
- Complex kernel runtime dispatch unified onto `#[runtime_dispatch]` (replaces per-type `OnceLock` feature caching).
- Complex dot uses two independent accumulators (measured 47.3 → 31.7 ms on the example workload).
- Workspace-wide `cargo fmt` normalization; rustdoc builds warning-clean.

### Fixed
- `SimdCow::histogram_cow` computed bin indices through `f32` for every lane type, misbinning `f64` values near bin boundaries; indices now derive in `f64`.
- `SimdCow::prefix_scan_in_place` used a scalar loop; now delegates to the vectorized view-level scan.

### Breaking
- Removed `InterleavedComplexLane`; the runtime complex entry points now bound on `SimdOps` (same call syntax for `f32`/`f64`).
- Removed `CsrCow`, `SellPCow`, `BlockedCooCow`, `DenseWithMaskCow`; use `SparseCow<T, Csr | SellP<C> | BlockedCoo<BM, BN> | DenseWithMask, Arch>`. Constructors are unchanged per format (`borrowed`, `owned`, `from_vecs` via turbofish).

## [0.1.0]

Initial workspace: `SimdView` typestate views, `SimdKernel` trait with Scalar/AVX2/AVX-512/NEON backends, `#[runtime_dispatch]` macro, dense/masked/sparse (CSR, SELL-p, BCOO, Dense-with-Mask) kernels, `SimdCow`, precision ladder (`hermes-numeric`), Intel AMX + AVX-512 VNNI tile GEMM, SWAR chess bitboards, tiling, tensor views, criterion/divan benches.
