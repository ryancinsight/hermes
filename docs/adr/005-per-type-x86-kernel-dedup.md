# ADR 005: Code Generation vs Macros for Per-Type x86 Kernel Deduplication

## Status
Accepted

## Context

The x86 specialized hardware kernels (`avx2_f32.rs`, `avx2_f64.rs`, `avx512_f32.rs`, `avx512_f64.rs`, etc.) share highly redundant method-body structures. Standard arithmetic (addition, subtraction, multiplication, division), fused multiply-accumulate operations, masked load/store, conversions, and simple comparison predicates contain identical logic flow, differing only in:
1. **Target feature gating**: e.g., `#[target_feature(enable = "avx2")]` vs `#[target_feature(enable = "avx512f")]`.
2. **Intrinsic names**: e.g., `_mm256_add_ps` vs `_mm256_add_pd`, or `_mm512_mul_ps` vs `_mm512_mul_pd`.
3. **Register/vector type wrappers**: e.g., `Avx2F32Vec` vs `Avx2F64Vec`.
4. **Lane count and unroll factors**.

This redundancy increases maintenance surface area, complicates bug fixes (which must be ported to all four files individually), and heightens the risk of silent divergence.

## Proposed Options

### Option 1: Declarative Macros (`macro_rules!`)
We could wrap the kernel implementation inside a large `macro_rules!` block and invoke it with type mappings, intrinsic names, and target feature strings.

* **Pros**:
  - Zero extra compile-time dependencies.
  - Resolved directly by `rustc` without invoking procedural code.
* **Cons**:
  - Severe degradation of IDE support, code navigation, and autocompletion within macro bodies.
  - Obscure compiler diagnostic spans, making debugging of compile errors extremely difficult.
  - Poor readability due to syntax escaping and macro repetitions (`$($...)`).

### Option 2: Procedural Macros
We could define an attribute macro (e.g. `#[generate_simd_kernel]`) in a helper crate to dynamically generate the trait implementations at compile time.

* **Pros**:
  - Keeps source files small and readable.
* **Cons**:
  - Slower build times due to compilation of `proc-macro` crates (`syn`, `quote`, `proc-macro2`).
  - IDEs cannot easily resolve code definitions generated dynamically in memory, breaking autocomplete.
  - Opaque code expansions make build/debugging loops hard to trace.

### Option 3: Build-Time Code Generation (`build.rs`)
Write a custom code generator in a `build.rs` script that outputs the target implementation source files at build time into standard files. During development, the generator can write directly to the `src/` directory (or be invoked via a custom CLI command/script) so that the generated code is checked into version control.

* **Pros**:
  - **IDE First-Class Support**: The generated source files are physical files in the filesystem, allowing cargo, IDEs, and cargo-expand to index them, complete types, and jump to definitions.
  - **Clean Compiler Spans**: Compiler errors point directly to the exact line in the generated file.
  - **No Runtime Dispatch or Proc-Macro Overhead**: Compile times are kept fast.
  - **Explicit Auditability**: Changes to the generator are visible via standard git diffs on the generated source files.
* **Cons**:
  - The generated source files must be regenerated if the schema changes.
  - Requires maintaining the generator logic.

## Decision

We recommend **Option 3 (Build-Time Code Generation via `build.rs` or a generation script)**. It provides a Single Source of Truth (SSOT) for kernel structures while preserving IDE ergonomics, clear compiler diagnostics, and version control auditability.

Under this model:
1. The common kernel logic (e.g. FMA, absolute sum/max, masked load/store loops) is defined in a JSON or Rust-based layout schema.
2. The generator parses the schema, injects target-specific intrinsic names and features, and outputs standard formatted Rust source files.
3. The generated files are checked into git to ensure transparency and prevent build-system compilation bottlenecks.

## Consequences

- x86 specialized float/integer kernel structures become highly maintainable, with new features or refinements implemented once in the generator and populated to all targets.
- Standard git diffs serve as the validation boundary for checking that code generation produces exact, warning-free Rust intrinsics.
- Autocomplete, documentation hover, and cargo-deny checks remain fully functional for all target files.
