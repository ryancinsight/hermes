# Backlog — hermes-simd

Strategic roadmap. Triage order: correctness → architecture → tests → docs → PM.
Tags: `[patch]` / `[minor]` / `[major]` / `[arch]` per SemVer change class.
Tactical breakdown of the active items lives in [checklist.md](checklist.md).
External gap findings live in [gap_audit.md](gap_audit.md).

## Delivered (2026-06-11)

- [x] [patch] (2026-06-26) Audit round 3 — SSOT, hierarchy, allocator retention.
  Finished the `MAX_SIMD_LANES` SSOT migration in `view/vector_reg.rs` (dead
  runtime asserts → compile-time `LANE_BOUND_CHECK`, 128→64 buffers); split
  `tensor/view.rs` into a vertical `tensor/view/{mod,rank_ops,simd_bridge}`
  hierarchy (SoC). Upstream Mnemosyne (`perf/huge-pool-byte-cap`): byte-bounded
  huge-pool retention (~16 GiB→~256 MiB/bucket) + removed a redundant per-pop
  atomic reload. 367 hermes tests + 210 Mnemosyne tests green; clippy/fmt/doc
  clean. See [gap_audit](gap_audit.md#audit-2026-06-26-r3).
- [x] [patch] (2026-06-26) Memory-efficiency cross-repo fix. Root-caused
  `AlignedVec<_, Aligned<64>>` small allocations costing ~2 MiB each (Mnemosyne
  routed `align > 16` to its huge path). Fixed upstream in Mnemosyne
  (`perf/aligned-small-alloc-tcache`: alignment-aware size-class selection) and
  removed the counterproductive hermes `adjust_layout_for_mnemosyne` 8 KiB
  padding + no-op dealloc NUMA bind. Measured 512 × 256B/64-aligned `AlignedVec`:
  **~1056 MiB → ~4 MiB** mapped. Plus O(nblocks) bounds guards on the BlockedCoo
  `spmv`/`elementwise_mul_dense` SIMD column loads (safety). See
  [gap_audit](gap_audit.md#alloc-audit-2026-06-26). 367 tests green; Mnemosyne
  98+23 tests green.
- [x] [minor] (2026-06-26) Audit sprint — safety, contention-free perf, memory.
  `NumericElement` extended to `i64`/`u8`/`u16`/`u32`/`u64` (+ first
  `hermes-numeric` tests). `MAX_SIMD_LANES` 128→64 (true max) halving fallback
  buffers, with `reduction.rs`/bitmask buffers folded onto the SSOT under the
  compile-time `LANE_BOUND_CHECK`. NUMA alloc-generation hardened
  (Relaxed→Release/Acquire + single-capture, closing a stale-cache/TOCTOU
  window). `build_index_vector` layout invariant made a `const` assert;
  `#![forbid(unsafe_code)]` on `hermes-simd-macros`; magic-table CAS ordering
  relaxed. Triplicated `SimdOps` impls collapsed to one macro (mod.rs
  1217→845); `flush_limit` deduped to a `const fn`; `axpy`/`scale` 4×-unrolled.
  367 tests + clippy `-D warnings` + fmt green.
- [x] [patch] (2026-06-24) Compile-time `LANE_COUNT <= MAX_SIMD_LANES` guard on
  the scalar-fallback `[MaybeUninit<T>; 128]` stack buffers (kernel + kernel_helpers):
  named SSOT constant + `SimdKernel::LANE_BOUND_CHECK` asserted per backend,
  replacing the unasserted/misleadingly-half-guarded magic 128. Prevents a silent
  stack overflow if a future wide backend (e.g. native SVE) uses the defaults.
  Validated by a lower-the-bound build failing AVX-512 compilation. Plus a
  rust-1.95 workspace clippy cleanup. 357 tests + clippy `-D warnings` green.
- [x] [minor] AXPY provider: `SimdOps::axpy` / dispatched `axpy` free fn —
  fused row update `out[i] += alpha * x[i]` via the `fmadd` primitive with
  scalar tail, no temporaries, length-mismatch error. Driver: leto matmul SIMD
  dispatch (its Stage C2 gate). Value tests across all tail sizes, f32/f64,
  zero-alpha identity, mismatch rejection.
- [x] [minor] Batched AXPY rows: `SimdOps::axpy_rows_batch` / dispatched
  `axpy_rows_batch` free fn — fused depth-major dense row-panel accumulation
  via one runtime-dispatched kernel, no temporaries, length-mismatch error.
  Driver: leto/coeus dense-panel accumulation. Delivered 2026-06-15 with
  repeated-`axpy_rows` differential coverage, invalid-extent tests, and
  register accumulation that stores each output lane once per call. Criterion
  coverage: `axpy_rows_batch_f32` compares the fused kernel against repeated
  public `axpy_rows` calls on depth-major row panels.
- [x] [patch] Dense/AXPY error-contract hardening: selected public dense
  facade and AXPY length-mismatch tests assert exact
  `SimdError::LengthMismatch` values instead of existence-only failures.
- [x] [patch] Select/unary error-contract hardening: select, unary-map, and
  COW FMA tests assert exact `SimdError` variants for length mismatch and
  insufficient output capacity.
- [x] [patch] Operation-family error-contract hardening: new operation,
  strategy, complex, and COW math tests assert exact `SimdError` variants for
  short outputs, length mismatch, and invalid gather indices.
- [x] [patch] COW unary invariant cleanup: `SimdCow::map_unary` now asserts
  its internally constructed output-length invariant instead of discarding the
  `SimdView::map_unary` result.
- [x] [patch] GEMM tiling rustdoc cleanup: module theorem prose now references
  private implementation details as code text instead of public intra-doc
  links.
- [x] [patch] Runtime FMA capability probe: `has_fma3` / `FmaSupport` now
  route through Rust's platform-aware runtime detector and are covered by
  host-capability tests.
- [x] [patch] GEMV rustdoc link cleanup: same-named dispatch modules and
  functions are disambiguated in public docs.
- [x] [minor] Const-generic Blocked-COO dispatch: replaced fixed public
  `spmv_bcoo4x4`/`spmv_bcoo8x8` dispatch and fixed
  `SparseView::from_blocked_coo_4x4`/`from_blocked_coo_8x8` constructors with
  one `spmv_bcoo::<T, BM, BN>` public API and the existing generic
  `from_blocked_coo` constructor. Driver: structural duplication audit.
- [x] [minor] Const-generic SELL-p dispatch: replaced fixed public
  `spmv_sellp4`/`spmv_sellp8` dispatch and fixed
  `SparseView::from_sellp4`/`from_sellp8` constructors with one
  `spmv_sellp::<T, C>` public API and the existing generic `from_sellp`
  constructor. Driver: structural duplication audit.

## Atlas in-house replacement roadmap — hermes slice [arch]

hermes is the Atlas **SIMD SSOT** (data-parallel lanes), replacing std::simd / packed_simd
and hand-rolled intrinsics. Scope boundary: hermes owns SIMD only; thread-level **MIMD**
is moirai's domain, GPU is the `hephaestus` substrate (wgpu + CUDA) via coeus/apollo. Work to make hermes the
complete SIMD substrate for leto-ops/coeus hot kernels:
- [ ] [minor] Stage C1: dedicated AVX-512 / AMX CI runners (currently self-skip on
  unsupported hosts), `no_std` feature matrix, committed criterion baselines.
  Partial delivered (2026-06-12): local AVX2 Criterion baseline refreshed with
  packed4 COW unpack and unrolled complex `mul_assign` rows; runner self-check
  covered 48 rows. Dedicated AVX-512/AMX runners remain open.
- [x] [patch] Stage C1: `SveArch` callable fallback (stub removal) — delivered
  2026-06-13 as a value-preserving 512-bit-shape emulated backend for f32/f64.
- [x] [minor] Stage C1: `SveArch` public marker + property coverage —
  delivered 2026-06-13 by re-exporting it from `hermes-simd` and adding it to
  the host-independent kernel property suite.
- [ ] [minor] Stage C1: native SVE intrinsic backend for AArch64 server targets.
- [ ] [minor] Stage C2: expand op/dtype coverage on demand from leto-ops/coeus
  (gather/scatter variants, additional reductions/scans, complex precisions) so every
  leto/coeus CPU hot kernel has a hermes path rather than a scalar fallback.
  Delivered (2026-06-12): abs-sum (`Σ|x|`) and abs-max (`max|x|`) slice
  reductions via `AbsSum`/`AbsMax` ReductionOp ZSTs + `SimdOps::{abs_sum,
  abs_max}` dispatch. The reduce loop's unrolled head previously seeded
  accumulators with raw loads and merged partials with `accumulate` — correct
  only for transform-free ops; it now seeds through `transform_vector` and
  merges through `combine_vectors`, fixing the latent defect for every
  transform-bearing reduction (the documented SquaredSum hook included).
- [x] [patch] Document the SIMD(hermes) vs MIMD(moirai) vs GPU(hephaestus: wgpu + CUDA)
  boundary in README so consumers compose the three deliberately. Delivered
  2026-06-12: README defines Hermes as the synchronous, slice-oriented SIMD
  substrate; Moirai owns thread-level partitioning; Hephaestus owns GPU
  resource lifetimes and device-resident kernels.

## External reference audits <a id="external-reference-audits"></a>

- [x] **[patch] Highway comparison audit** (2026-06-14): audited
      `https://github.com/NikoMalik/highway.git` at
      `0984271e74db124cf5e200de542e745348eb0b9e` and recorded Hermes-native
      gaps in [gap_audit.md](gap_audit.md#highway-2026-06-14).
- [x] **[patch] NumKong comparison audit** (2026-06-17): audited
      `https://github.com/ashvardanian/NumKong` and recorded Hermes-native
      gaps in [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
- [x] **[minor] Target-token forced dispatch**: add a Hermes `TargetId` and
      `dispatch_to`-style test/benchmark surface that checks CPU support before
      entering target-feature trampolines. Driver:
      [gap_audit.md#highway-2026-06-14](gap_audit.md#highway-2026-06-14).
      Delivered 2026-06-14 as `TargetId`, `dispatch_view_to`, and
      `dispatch_view_mut_to` with value-semantic host capability tests.
- [x] **[minor] Safe one-vector slice wrappers**: add bounds-checked and
      alignment-checked wrappers over `load_aligned`, `load_unaligned`,
      `store_aligned`, and `store_unaligned` for one-vector use cases,
      preserving raw-pointer kernels for hoisted hot loops. Driver:
      [gap_audit.md#highway-2026-06-14](gap_audit.md#highway-2026-06-14).
      Delivered 2026-06-14 on `Vector<T, Arch>` with exact failure tests.
- [x] **[arch] SSE2 backend feasibility ADR** (delivered 2026-06-21): evaluated a 128-bit
      x86_64 backend between Scalar and AVX2, resulting in ADR 006 recommending
      relying on compiler auto-vectorization or evaluating SSE4.1/SSSE3 as a modern baseline.
- [x] **[minor] Public dense facade cross-target matrix**: force every
      supported target available on the host and compare public dense facade
      results against Scalar for representative arithmetic, mask, reduction,
      gather, and shuffle paths. Driver:
      [gap_audit.md#highway-2026-06-14](gap_audit.md#highway-2026-06-14).
      Delivered 2026-06-15 with host-supported `TargetId` checks over sum,
      dot, elementwise arithmetic, gather, and select.
- [x] **[patch] Operation-family coverage map**: expanded the coarse Stage C2
      row into per-family entries in README and this backlog. Evidence tier:
      source audit over the current public surface and Highway reference audit;
      no performance or correctness claim is made for unimplemented families.

### Operation-family coverage map <a id="operation-family-coverage-map"></a>

Consumer admission rule: a family becomes implementation work only when an
Atlas consumer names a hot path or contract that requires it. Public APIs remain
Hermes-native, monomorphized, and backed by value-semantic tests before a row is
marked delivered.

- [x] [minor] Arithmetic: dense `sum`, `dot`, elementwise add/sub/mul/div,
  `axpy`, `axpy_rows`, `axpy_rows_batch`, sparse SpMV, and tiled GEMM/GEMV are
  present with scalar fallback and runtime dispatch.
- [x] [minor] Reductions: `sum`, `min`, `max`, `argmin`, `argmax`, `abs_sum`,
  `abs_max`, dot, masked reductions, and COW reductions are present.
- [x] [minor] Masks/select: `BitMask`, masked dense operations, `select`,
  `masked_negate`, mask round-trip property coverage, and safe target-forced
  dense conformance are present.
- [x] [minor] Memory: typestate `SimdView`, `AlignedVec`, COW promotion,
  packed4 COW unpack, safe one-vector load/store wrappers, and gather are
  present.
- [x] [minor] Shuffle/rearrange: complex adjacent-pair primitives
  (`swap_adjacent`, `dup_even`, `dup_odd`, `fmaddsub`, `fmsubadd`) and packed
  unpacking are present where consumer kernels require them.
- [x] [minor] Float-specialized: interleaved complex multiply/dot, norm,
  normalize, absolute reductions, and sqrt/abs/clamp unary strategies are
  present.
- [ ] [minor] Scatter/compress-store family: add only when an Atlas consumer
  needs indirect writes or compaction output; current delivered scope covers
  gather and mask/select, not scatter.
- [ ] [minor] Comparison predicate family: add lane-wise compare APIs only when
  a consumer needs reusable predicate outputs beyond existing min/max/select
  contracts.
- [ ] [minor] Conversion family: add vectorized widening/narrowing conversion
  APIs only when a consumer needs conversion as a public SIMD operation;
  current packed4 unpack is format-specific and owned by packed storage.
- [ ] [minor] Bitwise public facade family: add public bitwise dense APIs only
  when a consumer requires them; strategy ZSTs exist, but a broad public facade
  is not admitted without demand.
- [ ] [minor] Crypto/hash family: out of current Hermes scope unless an Atlas
  consumer requires lane-parallel primitive support; no implementation is
  claimed.

## Stage assessment (2026-06-10)

Phase: **Execution → Closure transition for 0.2.0.** Canonical trait surfaces
(`Scalar`, `SimdKernel`, `SparseFormat`/`CowFormat`, op-strategy ZSTs,
`#[runtime_dispatch]`) are defined with one-or-more concrete implementations
each; 278 workspace tests green; clippy/doc/fmt gates clean; aarch64
cross-compile verified. The dominant remaining risks are *infrastructure*
(no CI, no toolchain pin, no changelog) and *unverified hardware paths*
(AVX-512, NEON, AMX run compile-checked but not runtime-validated locally).

## P0 — Release engineering for 0.2.0 <a id="p0"></a>

- [x] **[patch] CI pipeline** (delivered 0.2.0; AVX-512 runner still open) (highest risk reducer): GitHub Actions running
      fmt → clippy `-D warnings` → `cargo test --workspace` → doc build →
      `cargo check --target aarch64-unknown-linux-gnu`. Add an ARM runner
      (or QEMU) job to runtime-validate the NEON complex/dense paths, and an
      AVX-512-capable runner if available for the `Avx512` differential tests.
- [x] **[patch] Toolchain pin + supply chain** (delivered 0.2.0; cargo-audit covered by cargo-deny in CI): `rust-toolchain.toml`, declared
      MSRV, `cargo audit` + `cargo deny check` in CI.
- [x] **[minor] 0.2.0 release** (semver-checks scoped per crate; see checklist): CHANGELOG sections (Added/Changed/Breaking —
      includes `InterleavedComplexLane` removal and per-format sparse-Cow type
      removal), `cargo-semver-checks` run, version bump committed atomically.

## P1 — Correctness hardening <a id="p1"></a>

- [x] **[patch] Reduced-precision complex coverage** (delivered 0.2.0): property/differential
      tests for `f16`/`bf16` interleaved complex kernels (currently exercised
      only via emulated defaults, asserted only for f32/f64).
- [x] **[patch] Mask/gather/compress property suite** (delivered 0.2.0): proptest invariants —
      `compress`∘`expand` identity under fixed mask, `mask_to_bitmask` ∘
      `mask_from_bitmask` round-trip, gather with permuted indices vs scalar
      reference, `leading_k_mask` boundary cases (k=0, k=LANE_COUNT, k>LANE_COUNT).
- [x] **[patch] `cargo miri` pass** (delivered post-0.2.0: core unit tests green under Miri; rkyv 0.7 tests excluded as upstream Stacked Borrows violations; CI job added) over crates containing `unsafe`
      (intrinsics excluded where Miri lacks ISA support; cover the
      view/cow/sparse pointer logic in hermes-simd-core).
- [x] **[patch] no_std + feature matrix** (delivered post-0.2.0: runtime_dispatch std-gating fixed, --no-default-features green + CI step; broader feature-combination sweep remains open): verify `--no-default-features` and
      key feature combinations build and pass.
- [x] **[minor] Fast reciprocal square root** (delivered 2026-06-21): implement `ops::RecipSqrt` (or `rsqrt`)
      with a Newton-Raphson refinement step for floating-point scalars, enabling Leto
      to avoid standard `sqrt` latency in normalized vector operations. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
- [x] **[arch] Masked tail-load/store API infrastructure** (delivered 2026-06-21): expose active-lane masked
      load and store helpers in `SimdKernel` and `Vector<T, Arch>`/`Mask<T, Arch>`
      for `Avx512` and `SveArch` so Leto can run tail-free kernels. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).

## P2 — Performance & memory <a id="p2"></a>

- [x] **[minor] Criterion regression thresholds** (delivered 2026-06-12):
      `benchmarks_baseline.json` records structured Criterion point estimates;
      `run-benches --check-regressions` fails on missing baseline rows or rows
      exceeding the configured ratio threshold. The runner is split into
      cohesive modules for CLI parsing, result discovery, host metadata,
      threshold comparison, and Markdown report rendering.
- [x] **[minor] SpMV scalability sweep** (delivered 2026-06-12): bench row
      counts ∈ {1K, 10K, 100K} at structural non-zero density {0.1%, 1%, 10%}
      across CSR/SELL-p/BCOO with 1024 columns; Dense-with-mask is capped at
      10K rows because it stores full dense values and masks. Sparse module
      docs now record format-selection guidance.
- [x] **[minor] Packed4 unpack generalization** (delivered 2026-06-12):
      `Packed4CowExt` delegates to `Packable4::unpack_slice_packed`, reusing
      the hermes-numeric AVX-512/AVX2/scalar dispatcher and removing the
      facade-local Bf4/F4 hardware-unpack impl pair. Criterion now includes a
      focused packed COW unpack benchmark.
- [x] **[minor] Complex mul_assign unroll** (delivered 2026-06-12):
      `interleaved_complex_mul_assign` processes two SIMD registers per loop
      iteration before the single-register and scalar tails. Criterion
      validation on this host showed runtime improvement across 256, 1K, 4K,
      and 16K complex-pair inputs.
- [x] **[minor] Expose popcount and horizontal reductions** (delivered 2026-06-21): add SIMD population
      count (`popcnt`) and bitwise horizontal fold/reduction primitives to the facade,
      enabling Leto/Hephaestus to implement Jaccard and Hamming distance metrics. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
- [x] **[minor] Sub-byte sign-extension and unpacking/widening** (delivered 2026-06-21): implement vector
      sign-extension and unpacking primitives for `Bf4`/`F4`/`I8` types to support
      quantized dot product optimizations in Leto. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).

## P3 — Architecture & maintenance <a id="p3"></a>

- [x] **[patch] x86 VNNI asm form** (delivered post-0.2.0): factor repeated
      `vpdpbssd` inline assembly into one internal instruction macro with
      explicit target-feature contract. The portable surface remains
      `TileMatrixMultiply`/runtime dispatch; asm is not promoted to a separate
      public abstraction.
- [x] **[arch] Per-type x86 kernel dedup** (delivered 2026-06-21): evaluated build-time
      code generation vs macros for AVX2/AVX-512 duplication, resulting in
      ADR 005 recommending build-time code generation via a custom `build.rs` script.
- [x] **[patch] SVE callable fallback**: removed `unimplemented!()` SVE
      `SimdKernel` methods and routed `SveArch` f32/f64 through the existing
      lane-emulated kernel macro with value-semantic tests.
- [x] **[minor] SVE property coverage**: `hermes-simd` re-exports `SveArch`,
      and `kernel_property_tests` now exercises its mask round-trip,
      compress/expand, gather, and leading-tail invariants on every host.
- [ ] **[minor] Native SVE backend**: hardware intrinsic implementation remains
      blocked on stable `core::arch::aarch64` SVE vector types; revisit on
      toolchain updates.
- [x] **[minor] Arm SME target feasibility study**: evaluate outer-product based
      tiled matrix multiplication kernels for Apple M4/M5 platforms. Driver:
      [gap_audit.md](file:///d:/atlas/repos/hermes/gap_audit.md#numkong-2026-06-17).
      Delivered 2026-06-21 as ADR 007 feasibility study.
- [x] **[minor] NUMA module status** (audited 2026-06-11): `numa.rs` IS
      integrated — `hermes-simd::dispatcher` uses `NumaTopologyService`/
      `verify_numa_locality`, `vec` uses `NumaAllocator`, and types_tests
      cover node count/distance. Finding: it reimplements platform NUMA
      detection (`GetNumaHighestNodeNumber` on Windows, sysfs on Linux) that
      **themis `CpuTopology` owns**, and its `MnemosyneNumaAllocator` names
      mnemosyne's allocation responsibility — a structural duplication across
      the stack SSOT map (themis=topology law, mnemosyne=allocation).
- [x] **[patch] Default provider feature policy**: every Hermes package
      defaults `parallel` and `mnemosyne-memory`; the default
      `MnemosyneNumaAllocator` path now uses Mnemosyne allocation instead of a
      name-only std/platform allocator branch. The broader Themis topology
      replacement below remains open.
- [x] **[arch] NUMA consolidation onto themis/mnemosyne** (delivered
      2026-06-12): `numa.rs` detection now delegates to themis —
      `current_numa_node` → `themis::try_current_numa_node` (Option-honest,
      added in themis 0.7.0), `NumaTopologyService::{current_cpu,total_nodes,
      node_distance}` → `themis::current_processor` / process-cached
      `CpuTopology::detect()` distance tables. The duplicated libnuma /
      GetNumaHighestNodeNumber / sched_getcpu platform blocks are deleted.
      Allocation already routes through mnemosyne (`MnemosyneNumaAllocator`
      with `NumaBinding`). Kept in hermes by design: `NumaAllocator` trait,
      `NumaBinding` thread-affinity RAII, and `verify_numa_locality` —
      SIMD-specific concerns the topology SSOT should not own. Public query
      surface unchanged; dispatcher/vec/tests untouched.

## P4 — Documentation <a id="p4"></a>

- [x] **[patch] Doctest coverage**: `cargo doc` is warning-clean; extended
      runnable doctests to the complex, sparse-Cow, and tensor public surfaces.
- [x] **[patch] Runnable core examples**: converted kernel, compute, and tiling
      public Rustdoc examples from compile-only `no_run` to executable
      value-semantic doctests.
- [x] **[patch] Runnable `BitMask` examples**: converted native-mask conversion
      and active-lane iteration examples from ignored snippets to executable
      value-semantic doctests.
- [x] **[patch] `#![deny(missing_docs)]`** (delivered post-0.2.0: all six public crates) on all public crates (currently
      `warn` in hermes-simd-core).

## Completed (recent)

- [x] [minor] Generic vectorized interleaved complex kernels + runtime dispatch
      (ADR-004; commits 33ce1b8, 3aa963e).
- [x] [minor] NEON adjacent-pair primitive overrides, aarch64 compile-verified (3aa963e).
- [x] [arch] Sparse Cow consolidation → generic `SparseCow<T, F, Arch>` + `CowFormat` (3aa963e).
- [x] [patch] Native-precision histogram binning fix + regression test (8b4a796).
- [x] [patch] Vectorized in-place prefix scan, single authoritative impl (8b4a796).
- [x] [patch] Complex-kernel property tests with analytical tolerances (8b4a796).
- [x] [patch] Workspace fmt normalization; rustdoc warning cleanup (fc34e6a, 3aa963e).
