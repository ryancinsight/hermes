# hermes-simd-macros

Procedural macros for [Hermes](https://github.com/ryancinsight/hermes), the
Atlas CPU SIMD substrate. Consumers reach these through the `hermes-simd`
facade rather than depending on this crate directly.

## Macros

- `#[runtime_dispatch(avx512f, avx2, neon, scalar)]` — turns one generic kernel
  function into a CPU-feature-dispatched wrapper that calls the monomorphized
  per-ISA specializations in priority order, emitting the `#[target_feature]`
  wrappers and the detection ladder. This is what keeps a single
  `<T: Scalar, A: SimdKernel<T>>` kernel per operation instead of per-type
  clones.
- `#[derive(SparseData)]` — generates the `SparseFormat` boilerplate for sparse
  data structs.

The crate executes no `unsafe` itself and forbids it; the `unsafe` it emits
lives in the generated token streams and is compiled in the consumer crate.

## Documentation

- API reference: [docs.rs/hermes-simd-macros](https://docs.rs/hermes-simd-macros)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hermes#readme)

## License

MIT OR Apache-2.0
