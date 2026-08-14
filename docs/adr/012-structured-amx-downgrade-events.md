# ADR 012: Structured AMX downgrade events

## Status

Accepted

## Context

`hermes-simd::AdaptiveDispatcher` can reject AMX for a large operation when
the input buffers are remote to the executing NUMA node. The dispatcher then
selects AVX-512 when available, or scalar execution. The existing diagnostic
was a `debug_assertions`-only `eprintln!`, so release users received no signal
that the selected backend changed. The branch became reachable after the AMX
probe gained hardware, XCR0, and process-permission checks.

The facade supports `no_std`. The event mechanism must therefore compile
without the standard library, add no allocation or callback state to the hot
dispatch path, and remain optional at the subscriber boundary. The existing
once-latched atomic keeps the warning from becoming a per-dispatch side
effect; this decision changes only how the single event is emitted and when
it is compiled.

## Decision

Depend on `tracing` with default features disabled and emit one warning event
from the standard-library dispatcher path. The event target is
`hermes_simd::dispatcher` and carries `numa_node`, `from_backend`,
`to_backend`, and `reason` fields. Release builds emit the same event as debug
builds. A consumer chooses whether and where to record it by installing a
`tracing` subscriber; no stderr output or process-global logger is owned by
the library.

The `no_std` build keeps the dependency available without enabling its
standard-library feature, while the dispatcher branch remains gated on the
facade's existing `std` feature because NUMA locality and the once-latched
diagnostic use `std` there. AMX sessions also reject safely without `std`:
portable no-std Rust has no thread-local storage primitive, so the old global
`Cell` substitute and its unsound `Sync` implementation are removed rather
than pretending to provide per-thread state. A subscriber-backed unit test
asserts the event message and all routing fields.

## Alternatives rejected

- Keep `eprintln!`: rejected because it is silent in release builds and
  bypasses the application's logging and redaction policy.
- Add a callback or logger trait to the dispatcher: rejected because it adds
  state and an indirect call to a hot operation boundary for a diagnostic
  concern.
- Make `tracing` standard-library-only: rejected because it would make the
  public facade's dependency graph configuration-dependent and would not
  preserve its `no_std` build.

## Consequences

The workspace gains one small, no-std-compatible observability dependency and
one test-only subscriber dependency. Applications that configure `tracing`
now receive the downgrade in release builds; applications that do not install
a subscriber retain the normal disabled-event cost. The event is warning
level because the dispatcher remains correct, but the requested backend was
not usable for the current memory placement.

## Verification

- `cargo fmt --all -- --check`
- `cargo clippy -p hermes-simd --all-targets --all-features -- -D warnings`
- `cargo nextest run -p hermes-simd --all-features`
- `cargo check -p hermes-simd --no-default-features`
- the subscriber-backed value-semantic event test above
