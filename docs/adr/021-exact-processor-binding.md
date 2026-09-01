# ADR 021: Exact Processor Binding

## Status

Accepted

## Context

Hermes already owns the stack's explicit thread-affinity guards. `NumaBinding`
temporarily narrows a thread to a NUMA node while Themis owns topology discovery
and Mnemosyne owns allocation. That node-level contract cannot make a
microbenchmark reproducible on a hybrid processor: the operating system may
move one process among performance and efficiency cores inside the same node.

Apollo exposes the failure. After its benchmark profile was aligned with
release code generation, 1,024- and 2,048-point FFT comparisons became stable,
but consecutive 32,768-point runs retained separate latency bands in both
Apollo and RustFFT. More samples cannot remove a change in the processor class
executing the samples. Apollo currently has a test-only Win32
`SetThreadAffinityMask` shim; it neither restores the previous affinity nor
addresses Windows processor groups, and copying it into a benchmark would fork
the affinity contract again.

## Options

1. **Keep benchmark-local operating-system shims.** This duplicates unsafe FFI
   in every consumer, omits restoration, and makes each consumer independently
   define processor identity and processor-group behavior.
2. **Put binding in Themis.** Themis reports topology and locality; a mutable
   thread-lifetime guard is an execution mechanism, not topology data. This
   would reverse the established ownership in ADR 010.
3. **Adopt a third-party affinity crate.** Hermes already owns the narrower
   affinity role, and the required Windows processor-group and typed-error
   contract is small. A second provider would leave two stack authorities for
   the same operation.
4. **Extend Hermes' affinity module with one exact-processor guard.** Keep
   processor identity, validation, platform mutation, lifetime, restoration,
   and errors behind one public allocation-free API; consumers keep only their
   measurement policy.

## Decision

Adopt option 4. `ProcessorIndex` is the operating system's logical-processor
index. On Windows it uses the same stable flattening as the stack's topology
queries: `group * 64 + processor_number`. Themis's
`ProcessorGroupAffinity::from_processor` converts that index to a native group
and one-bit mask. `ProcessorBinding::bind` validates the provider value against
the active processor-group inventory before changing affinity.

The Windows backend uses `SetThreadGroupAffinity`, not the single-group
`SetThreadAffinityMask`. It records the complete prior `GROUP_AFFINITY`, binds
one bit in the requested group, and restores the prior value on scope exit. The
guard is neither `Send` nor `Sync`: the saved affinity belongs to the calling
thread, and restoring it from another thread would mutate the wrong thread.
`ProcessorIndex::current` exposes the actual current processor so measurement
instruments can verify placement after yielding.

Binding returns a typed `Result`. An invalid or inactive processor is rejected
before mutation. Unsupported targets return an explicit unsupported-platform
variant; no target silently succeeds without binding. Platform query, bind, or
explicit restore failures retain the operation and operating-system error code.
An explicit `restore` method permits callers to observe restoration failure;
`Drop` remains the panic-free unwind fallback and retries restoration when the
guard is still active.

The API remains available in `no_std`: its state is fixed-size, errors borrow
only static operation names, and the Windows backend calls the platform ABI
directly. The current increment implements Windows because it is the measured
consumer requirement. Other targets keep the same public type and return the
unsupported variant until a native backend lands with equivalent validation
and restoration semantics.

## Consequences

- Apollo can bind its comparison process to a named processor and delete its
  duplicated test-only Win32 shim. Processor selection remains an Apollo
  measurement-policy input; Hermes does not choose a core class.
- Exact binding allocates nothing, performs no scheduler dispatch, and adds no
  work inside a timed kernel. Construction and destruction are cold operation
  boundaries.
- Windows tests must prove exact current-processor identity, restoration of the
  complete prior group affinity, invalid-request no-mutation, and explicit
  restore behavior. Cross-target checks prove the unsupported implementation
  remains buildable; they do not prove runtime affinity on those targets.
- The public surface is additive [minor]. No compatibility alias, alternate
  guard family, or consumer-owned platform wrapper remains after adoption.

## Revision history

- 2026-09-01: Delegate flattened-index decomposition and native one-bit mask
  construction to Themis for `HS-THEMIS-AFFINITY-CONSUMER-2026-09-01`.
  Hermes still owns active-host validation and the thread-bound bind/restore
  mechanism, preserving the decision's topology-versus-execution boundary.
- 2026-08-31: Accepted for `HS-EXACT-PROCESSOR-BINDING-2026-08-31`, driven by
  Apollo's 32,768-point comparison variance after release-profile correction.
