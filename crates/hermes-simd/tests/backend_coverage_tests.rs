//! Host capability identification and coverage accounting.
//!
//! Every backend-specific test in this workspace is guarded by a runtime
//! feature probe (`is_x86_feature_detected!`, `TargetId::is_supported`). That
//! is the correct design — backend selection is automated from what the host
//! actually provides — but it has one failure mode: a guarded test that does
//! not run does not fail either. It skips, and a skip is indistinguishable
//! from a pass in the log. A suite can therefore report full green while never
//! executing an entire ISA, which is exactly what happened on this repository's
//! x86 CI runner, where no AVX-512 path had ever executed.
//!
//! This module closes that hole without duplicating any detection logic: it
//! reads the same probes the dispatcher uses, reports the resulting matrix, and
//! asserts against an expectation supplied as configuration.

use hermes_simd::TargetId;

/// Environment variable naming the targets a given runner must cover.
///
/// Comma-separated [`TargetId::name`] values, e.g. `scalar,avx2`. Unset means
/// "report only", which is the right default for a developer machine whose
/// capabilities are not known ahead of time; CI sets it per runner so a host
/// that silently loses a capability fails loudly instead of quietly skipping.
const EXPECTED_TARGETS: &str = "HERMES_EXPECTED_TARGETS";

fn coverage_report() -> String {
    let mut report = String::from("host backend coverage:\n");
    for target in TargetId::ALL {
        let mark = if target.is_supported() {
            "executes"
        } else {
            "SKIPPED (host cannot execute)"
        };
        report.push_str(&format!("  {:<8} {}\n", target.name(), mark));
    }
    report
}

/// Identifies the host's executable backends and asserts the runner covers the
/// set it declares. The report is printed unconditionally so every CI log
/// states which backends that job exercised.
#[test]
fn host_backend_coverage_is_reported_and_meets_expectation() {
    let report = coverage_report();
    // Printed rather than returned so the matrix lands in the job log even when
    // the assertion below passes.
    println!("{report}");

    let Ok(raw) = std::env::var(EXPECTED_TARGETS) else {
        println!(
            "{EXPECTED_TARGETS} is unset: reporting only, no coverage assertion. \
             CI sets it per runner."
        );
        return;
    };

    let mut unknown = Vec::new();
    let mut missing = Vec::new();
    for name in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match TargetId::from_name(name) {
            // A misspelled expectation must not silently weaken the gate into
            // vacuous success, so an unparsable name is itself a failure.
            None => unknown.push(name.to_string()),
            Some(target) if !target.is_supported() => missing.push(target.name()),
            Some(_) => {}
        }
    }

    assert!(
        unknown.is_empty(),
        "{EXPECTED_TARGETS} names unknown targets {unknown:?}; valid names are {:?}\n{report}",
        TargetId::ALL.map(TargetId::name)
    );
    assert!(
        missing.is_empty(),
        "this runner was declared to cover {missing:?}, but the host cannot \
         execute them, so every test guarded on those targets skipped silently\n{report}"
    );
}

/// `from_name` must invert `name` across the whole closed set, since the
/// coverage expectation above is configuration and a broken round-trip would
/// turn a real expectation into an unknown-target error or, worse, drop it.
#[test]
fn target_name_round_trips() {
    for target in TargetId::ALL {
        assert_eq!(
            TargetId::from_name(target.name()),
            Some(target),
            "name/from_name round-trip failed for {}",
            target.name()
        );
    }
    assert_eq!(TargetId::from_name("not-a-target"), None);
}

/// The scalar target is unconditional, so `supported_on_host` is never empty
/// and the coverage report always has at least one executing row.
#[test]
fn scalar_is_always_supported() {
    let supported = TargetId::supported_on_host();
    assert!(
        supported.contains(&TargetId::Scalar),
        "scalar must execute on every host; got {supported:?}"
    );
}
