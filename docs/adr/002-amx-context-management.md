# ADR 002: AMX Context Management and State Tracking

## Status
Accepted

## Context

Intel Advanced Matrix Extensions (AMX) provide tile matrix multiplication accelerators. However, AMX registers (8 tiles, 8KB state) introduce system overhead:
1. **Context-Switch Penalties**: If tile registers are active, the operating system must save and restore the 8KB tile state on every thread context switch.
2. **Configuration Cost**: The `ldtilecfg` instruction loads configuration parameters from memory, which takes significant CPU cycles.

## Design

### 1. Thread-Local State Tracking

We implement `AmxSession` and `AmxBatchSession` to manage the lifecycle of tile register configuration:
- `SESSION_DEPTH`: A thread-local `Cell<usize>` tracking nested session entries.
- `ACTIVE_CONFIG`: A thread-local `Cell<Option<AmxConfig>>` caching the current active configuration.
- `AmxSession::new` and `AmxBatchSession::begin` return `AmxSessionError::UnsupportedTarget`
  before issuing `ldtilecfg` unless the runtime AMX support probe confirms the
  host and process may execute tile instructions.

### 2. Context-Switch Mitigation

- When `SESSION_DEPTH` returns to 0 (or when `AmxSession::release` is invoked), the system executes `tilerelease()` only for a supported active session.
- Releasing the tile registers resets their state to "initialized". The OS kernel detects this state and skips saving/restoring the 8KB register file on subsequent context switches.

### 3. Dynamic Configuration Caching

- When entering a new session, the system queries the thread-local `ACTIVE_CONFIG`.
- If the requested configuration matches `ACTIVE_CONFIG`, the redundant `ldtilecfg` instruction is skipped.
- Caching allows multiple independent matrix multiplication kernels to run sequentially without reloading the configuration structure.

### 4. Adaptive Heuristics and Fallback

- The hardware dispatcher selects the compute backend dynamically:
  - If an AMX session is already active (overheads are already paid), AMX execution is permitted for smaller workloads (down to 2048 operations).
  - If no session is active, small workloads bypass AMX to avoid `ldtilecfg` latency, falling back to AVX-512 or Scalar paths.
  - Cross-node NUMA memory configurations fall back to AVX-512 to prevent cache-coherency bottlenecks.

## Consequences

- Bypasses unnecessary register re-loading inside nested loops.
- Mitigates OS scheduler context-switch overhead when AMX is idle.
- Scales compute path selection automatically based on thread state.
