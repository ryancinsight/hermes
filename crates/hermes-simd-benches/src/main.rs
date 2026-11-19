mod cli;
mod criterion_results;
mod host;
mod regression;
mod report;

use std::env;
use std::path::PathBuf;
use std::process::Command;

use cli::BenchRunnerArgs;
use criterion_results::find_criterion_results;
use host::{get_cpu_info, HostCapabilities};
use regression::{check_regressions, write_baseline};
use report::render_markdown;

/// Render a UTC timestamp for the benchmark baseline, without a fallible
/// formatter: the calendar fields of an `OffsetDateTime` cannot fail to
/// format, and UTC keeps the baseline unambiguous across hosts.
fn format_utc_timestamp(now: time::OffsetDateTime) -> String {
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = BenchRunnerArgs::parse(env::args().skip(1))?;
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .expect("invariant: bench crate is under crates/")
        .parent()
        .expect("invariant: crates/ is under the workspace root")
        .to_path_buf();
    let target_dir = match env::var("CARGO_TARGET_DIR") {
        Ok(val) => PathBuf::from(val),
        Err(_) => workspace_root.join("target"),
    };
    let target_criterion = target_dir.join("criterion");
    let baseline_path = args
        .baseline_path
        .clone()
        .unwrap_or_else(|| workspace_root.join("benchmarks/benchmarks_baseline.json"));

    if args.parse_only {
        println!(
            "Skipping benchmark run. Parsing existing results from: {}",
            target_criterion.display()
        );
    } else {
        println!("Running workspace benchmarks via cargo bench...");
        let status = Command::new("cargo")
            .args(["bench", "--workspace"])
            .current_dir(&workspace_root)
            .status()?;
        if !status.success() {
            return Err("cargo bench execution failed".into());
        }
    }

    if !target_criterion.exists() {
        return Err(format!(
            "Criterion results directory not found at {}",
            target_criterion.display()
        )
        .into());
    }

    let mut results = Vec::new();
    find_criterion_results(&target_criterion, &mut results)?;
    if results.is_empty() {
        eprintln!("Warning: no benchmark results found in target/criterion");
    }

    let cpu = get_cpu_info();
    let capabilities = HostCapabilities::detect();
    let generated_at = format_utc_timestamp(time::OffsetDateTime::now_utc());

    if args.write_baseline {
        write_baseline(
            &baseline_path,
            &results,
            &cpu,
            capabilities.runtime_backend(),
            &capabilities.format_markdown(),
            &generated_at,
        )?;
        println!(
            "Successfully wrote benchmark baseline to {}",
            baseline_path.display()
        );
    }

    if args.check_regressions {
        let report = check_regressions(&baseline_path, &results, args.threshold)?;
        if report.has_failures() {
            eprintln!("{report}");
            return Err("benchmark regression threshold exceeded".into());
        }
        println!("{report}");
    }

    let markdown = render_markdown(&results, &cpu, capabilities, &generated_at);
    let output_path = workspace_root.join("benchmarks/benchmarks_results.md");
    std::fs::write(&output_path, markdown)?;
    println!(
        "Successfully wrote benchmark results to {}",
        output_path.display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::format_utc_timestamp;

    #[test]
    fn timestamp_renders_utc_calendar_fields() {
        let instant = time::OffsetDateTime::from_unix_timestamp(1_700_000_000)
            .expect("invariant: the epoch instant is representable");
        assert_eq!(format_utc_timestamp(instant), "2023-11-14 22:13:20 UTC");
    }
}
