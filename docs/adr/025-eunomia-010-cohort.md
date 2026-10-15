# ADR 025: Adopt the Eunomia 0.10 provider cohort

## Status

Accepted (2026-10-15)

## Contract and decision

Hermes reexports Eunomia scalar types and exposes their traits through public
generic kernels. The provider's CastFrom/CastTo retirement is versioned 0.10.0;
its nominal type identities require a pre-1.0 minor boundary in Hermes. Advance
the aligned Rust workspace line and internal package edges from 0.8.2 to 0.9.0.
Require Eunomia 0.10.0 and the coordinated Mnemosyne-memory 0.10.0 line. Keep
canonical Git URLs and Cargo-managed precise locks, without revision aliases,
path patches or compatibility shims. Rust implementations, fixtures, assertions,
precision choices and native deadlines are unchanged.

## Exact-head acceptance

[Reviewed head](https://github.com/ryancinsight/hermes/commit/cf8013a375501b746b41988e2d5beed90639baba)
and [published merge](https://github.com/ryancinsight/hermes/commit/7a584238c3da61b7d38f09b1da1b277ff20816aa)
share an [identical tree](https://github.com/ryancinsight/hermes/tree/83a3ba0e519a54fd65d4b4705ccabe99dc09c7e9).
All twelve executed [final CI jobs](https://github.com/ryancinsight/hermes/actions/runs/38754537267)
pass, including actual required Miri, SDE, native ARM/macOS/x86 and unchanged
runtime budgets. All ten branch-required statuses pass; no requested review
changes or review override exists. The unchanged tracked gate over the complete
provider-lock range passes with zero conformance regressions and no baseline change.

The final [blocking release validation](https://github.com/ryancinsight/hermes/actions/runs/38754801509)
builds/classifies four ordinary-library published baselines from 0.8.2 to 0.9,
with 260 rules skipped per library by the automatic major-compatible class.
The existing caller excludes the unchanged proc-macro API; its Rust remains
byte-identical and compiler/native plus standard package checks cover it.
This is no backward-compatibility claim or release-type override; no unbuildable
baseline fallback supplies the verdict. The core standard package dry-run
passes, and all five clean exact-head all-feature workspace packages verify.

The final runtime/lock passes all 507 native cases, zero skips, strict
all-target/all-feature Clippy, minimal builds, warning-denied docs and 27
doctests (ten existing illustrative ignores). The documentation successor
changes no Rust/Cargo/workflow/deadline bytes; its full required CI and package
results are collected at the exact reviewed head. The lock selects published
Eunomia a231e54d, Mnemosyne b7e0e1e, Themis 63c54410 and Melinoe 681c550e.
Required Miri establishes its unchanged declared core/intrinsics boundary only;
held safety work and performance experiments remain untouched.

## Immutable registry receipts

The existing [publisher](https://github.com/ryancinsight/hermes/actions/runs/38756242324)
succeeds after the normal guarded merge. All five archives are present and not
yanked, independently checksum-matched, and their `.cargo_vcs_info.json` source
identifies the published merge above.

| Crate | Version | Registry SHA-256 |
| --- | --- | --- |
| `hermes-simd-core` | 0.9.0 | `22bc61312331fdc74923f331a885297962c03a73292f893a5a268e3d4b38a093` |
| `hermes-simd-intrinsics` | 0.9.0 | `fd84257a8d36b3dbc4253363f4a34dd925369b948328247e4a61dbc093870fd7` |
| `hermes-simd-macros` | 0.9.0 | `67fb179583fca545a1d2c52dc8dc51e30ec675c9604905dcc5bc1aa0ac7b830d` |
| `hermes-simd-types` | 0.9.0 | `fd7218e102a51f8a6b109f41275b37aea01808fb0e2186465bc5cb84e3c4e86f` |
| `hermes-simd` | 0.9.0 | `03fd29cbc8055ae34b314607b63731b06b956436140c3fba498eca44fab2ac57` |

## Unexpected Pages side effect and remaining boundary

The authorized package merge also triggers GitHub's built-in legacy Pages
[deployment](https://github.com/ryancinsight/hermes/actions/runs/38756241166),
which is already successful when inspected. The earlier review checked tracked
workflow path filters but missed repository Pages configured for main at root.
Only Aequitas documentation deployment had explicit approval. No settings were
changed, and no additional deployment is triggered to undo or conceal it.
The user subsequently approves automatic documentation deployment on merge for
these repository repairs. That prospective approval clears the further-merge
hold; retain all exact-head checks/reviews and verify each resulting deployment.
The earlier unanticipated deployment remains recorded as such. This five-Markdown-
file change adds no package version, Rust source, lock, workflow or native deadline.
