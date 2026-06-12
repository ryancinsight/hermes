use crate::criterion_results::{BenchKey, BenchResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
struct BaselineFile {
    version: u32,
    generated_at: String,
    cpu: String,
    runtime_backend: String,
    isa_features: String,
    results: Vec<BenchResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionFailure {
    key: BenchKey,
    baseline_ns: f64,
    current_ns: f64,
    threshold: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionReport {
    checked: usize,
    failures: Vec<RegressionFailure>,
    missing_current: Vec<BenchKey>,
}

impl RegressionReport {
    pub(crate) fn has_failures(&self) -> bool {
        !self.failures.is_empty() || !self.missing_current.is_empty()
    }
}

impl Display for RegressionReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Benchmark regression check: {} matching baseline row(s) checked",
            self.checked
        )?;
        for failure in &self.failures {
            writeln!(
                f,
                "- regression: {}/{}/{} current {:.2} ns > allowed {:.2} ns (baseline {:.2} ns, threshold {:.2}x)",
                failure.key.group_id,
                failure.key.function_id,
                failure.key.value_str.as_deref().unwrap_or("-"),
                failure.current_ns,
                failure.baseline_ns * failure.threshold,
                failure.baseline_ns,
                failure.threshold
            )?;
        }
        for missing in &self.missing_current {
            writeln!(
                f,
                "- missing current row: {}/{}/{}",
                missing.group_id,
                missing.function_id,
                missing.value_str.as_deref().unwrap_or("-")
            )?;
        }
        Ok(())
    }
}

pub(crate) fn write_baseline(
    path: &Path,
    results: &[BenchResult],
    cpu: &str,
    runtime_backend: &str,
    isa_features: &str,
    generated_at: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let baseline = BaselineFile {
        version: 1,
        generated_at: generated_at.to_string(),
        cpu: cpu.to_string(),
        runtime_backend: runtime_backend.to_string(),
        isa_features: isa_features.to_string(),
        results: sorted_results(results),
    };

    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, &baseline)?;
    Ok(())
}

pub(crate) fn check_regressions(
    baseline_path: &Path,
    current: &[BenchResult],
    threshold: f64,
) -> Result<RegressionReport, Box<dyn std::error::Error>> {
    if threshold < 1.0 {
        return Err("regression threshold must be >= 1.0".into());
    }

    let file = std::fs::File::open(baseline_path)?;
    let baseline: BaselineFile = serde_json::from_reader(file)?;
    Ok(compare_results(&baseline.results, current, threshold))
}

fn compare_results(
    baseline: &[BenchResult],
    current: &[BenchResult],
    threshold: f64,
) -> RegressionReport {
    let current_by_key: BTreeMap<BenchKey, &BenchResult> = current
        .iter()
        .map(|result| (result.key(), result))
        .collect();

    let mut checked = 0;
    let mut failures = Vec::new();
    let mut missing_current = Vec::new();

    for base in baseline {
        let key = base.key();
        match current_by_key.get(&key) {
            Some(cur) => {
                checked += 1;
                if cur.time_ns > base.time_ns * threshold {
                    failures.push(RegressionFailure {
                        key,
                        baseline_ns: base.time_ns,
                        current_ns: cur.time_ns,
                        threshold,
                    });
                }
            }
            None => missing_current.push(key),
        }
    }

    RegressionReport {
        checked,
        failures,
        missing_current,
    }
}

fn sorted_results(results: &[BenchResult]) -> Vec<BenchResult> {
    let mut sorted = results.to_vec();
    sorted.sort_by_key(BenchResult::key);
    sorted
}

#[cfg(test)]
mod tests {
    use super::compare_results;
    use crate::criterion_results::BenchResult;

    fn result(group: &str, function: &str, value: Option<&str>, time_ns: f64) -> BenchResult {
        BenchResult {
            group_id: group.to_string(),
            function_id: function.to_string(),
            value_str: value.map(str::to_string),
            time_ns,
            throughput_kind: None,
            throughput_val: None,
        }
    }

    #[test]
    fn flags_only_rows_over_the_threshold() {
        let baseline = [result("Dense Sum", "dispatch", Some("1024"), 100.0)];
        let current = [result("Dense Sum", "dispatch", Some("1024"), 111.0)];

        let report = compare_results(&baseline, &current, 1.10);

        assert_eq!(report.checked, 1);
        assert_eq!(report.failures.len(), 1);
        assert!(report.has_failures());
    }

    #[test]
    fn accepts_rows_within_threshold() {
        let baseline = [result("Dense Sum", "dispatch", Some("1024"), 100.0)];
        let current = [result("Dense Sum", "dispatch", Some("1024"), 109.0)];

        let report = compare_results(&baseline, &current, 1.10);

        assert_eq!(report.checked, 1);
        assert!(report.failures.is_empty());
        assert!(!report.has_failures());
    }

    #[test]
    fn missing_baseline_rows_are_failures() {
        let baseline = [result("Dense Sum", "dispatch", Some("1024"), 100.0)];
        let current = [];

        let report = compare_results(&baseline, &current, 1.10);

        assert_eq!(report.checked, 0);
        assert_eq!(report.missing_current.len(), 1);
        assert!(report.has_failures());
    }
}
