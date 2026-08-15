# 1. ISA Detection

A single release binary must run correctly and quickly on CPUs that differ in
instruction-set capability: a Haswell-era x86-64 host has FMA3 but no AVX-512,
a Sapphire Rapids host has both, and an AArch64 host has neither. hermes-simd
therefore separates two questions that are easy to conflate:

1. *Is this instruction available on the host?* — a runtime probe.
2. *Does this code path execute it?* — a dispatch decision that must never run
   an unsupported instruction.

This chapter covers the probes; the next chapter covers the dispatch.

## The probe layer

The runtime probes live in `hermes_simd::cpu`. The primary entry points:

- `has_fma3()` — processor-wide FMA3 availability, cached in a `OnceLock` for
  the process lifetime.
- `f32::has_fma()` / `f64::has_fma()` — the per-element-type views from the
  `FmaSupport` trait.
- `has_avx512_bf16()` — native `DPBF16PS` capability, the *exact* gate for the
  native BF16 tile path (the broader `Avx512Support` trait also admits the
  AVX-512F/BW/VL conversion+FMA fallback).
- `has_avx_vnni()` — 256-bit AVX-VNNI (`vpdpbusd`/`vpdpwssd`), present on
  Intel Alder Lake+ and AMD Zen 5 parts that have no AVX-512.

The probes are cheap: each is computed once under `OnceLock::get_or_init` and
every later call reads a cached `bool`. The underlying detector is Rust's
platform-aware `std::is_x86_feature_detected!`, which handles the CPUID leaf
work, including XCR0/OSXSAVE state — a CPUID-only probe can report a feature
bit that the OS has not enabled, and dispatching on that is a fault.

Architecture is part of the contract. `has_fma3()` is x86_64-specific and
returns `false` on every other target without branching; AArch64/NEON has its
own fused multiply-add instruction family, dispatched through a separate path,
not through this probe.

## Conservative support reporting

AMX is the instructive case for why probes must be *capability-exact*.
`AmxSupport::has_amx()` deliberately reports `false` on all current hosts.
Raw CPUID cannot distinguish hardware feature bits from OS enablement and
Linux `XTILEDATA` process permission; entering `ldtilecfg`/`tdpbf16ps` when the
OS has not enabled the tile state causes a `#NM` fault. Until a stable,
permission-aware probe exists, reporting `false` preserves the safe-dispatch
contract — a dispatch into an instruction the host cannot execute is undefined
behavior, not a missed optimization.

The same discipline applies to the AVX-512 BF16 tile. `has_avx512_bf16()` is
deliberately distinct from `Avx512Support::has_avx512()`: the latter admits the
conversion/FMA fallback on AVX-512F/BW/VL hosts, while native callers must use
the exact capability gate before entering `DPBF16PS` code. The two probes
answer different questions and are never substituted for each other.

## Executing vs. covered

A probe that returns `false` and therefore *skips* a test is indistinguishable
from a test that passed — it is a silent gap in coverage. hermes-simd's CI
addresses this directly. `TargetId` enumerates the dispatch targets (an
`#[non_exhaustive]` set, so additions are non-breaking), and a coverage step
prints the per-runner matrix and asserts against
`HERMES_EXPECTED_TARGETS` declared per runner as configuration. The report
distinguishes three outcomes:

- **executes** — the target ran on this host,
- **not covered** — the architecture applies but the CPU lacks the feature,
- **n/a** — different architecture.

Collapsing the last two would make an ARM log read as missing AVX-512. Where
no selectable silicon exists (AVX-512 and AMX on GitHub-hosted runners), the
suite runs under Intel SDE emulating Sapphire Rapids — under the emulator the
identification step must still pass, so a green run is a hard assertion that
the emulator satisfies the same runtime probes, not merely that the test did
not break.

## Reading the probes

```text
FMA3 (fused multiply-add) : true
f32 FmaSupport            : true
```

The worked example in [Example: ISA Dispatch](examples/isa_dispatch.md) prints
these probes and then drives `sum` and `dot` through the runtime-dispatch path,
so the numbers in the sample output are computed by whichever ISA the host
supports — on an FMA3-capable host, the FMA-accelerated kernel.

## What to notice

- **One probe, many calls.** The `OnceLock` cache means the probe cost is paid
  once per process; the dispatch decision itself is a single cached read.
- **Probes are exact gates, not hints.** A capability whose probe is
  conservative (AMX) stays *off*, never *maybe*. The safe-dispatch contract is
  that code reaches an instruction only when the host provably supports it.
- **Coverage is asserted, not assumed.** A probe-guarded test that skips is a
  coverage defect. The distinction between "executes" and "not covered" is
  recorded explicitly per runner.
