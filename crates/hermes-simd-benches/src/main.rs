#![allow(
    clippy::derive_ord_xor_partial_ord,
    clippy::needless_borrows_for_generic_args,
    clippy::single_char_add_str
)]
use serde::Deserialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Deserialize, Debug)]
struct Estimate {
    point_estimate: f64,
}

#[derive(Deserialize, Debug)]
struct Estimates {
    mean: Option<Estimate>,
    slope: Option<Estimate>,
}

#[derive(Deserialize, Debug)]
struct Benchmark {
    group_id: String,
    function_id: Option<String>,
    value_str: Option<String>,
    throughput: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct BenchResult {
    group_id: String,
    function_id: String,
    value_str: Option<String>,
    time_ns: f64,
    throughput_kind: Option<String>,
    throughput_val: Option<u64>,
}

fn get_throughput_val(val: &serde_json::Value) -> Option<(String, u64)> {
    if let Some(obj) = val.as_object() {
        for (k, v) in obj {
            if let Some(n) = v.as_u64() {
                return Some((k.clone(), n));
            }
        }
    }
    None
}

fn format_time(ns: f64) -> String {
    if ns < 1.0 {
        format!("{:.2} ps", ns * 1000.0)
    } else if ns < 1000.0 {
        format!("{:.2} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.2} \u{03bc}s", ns / 1000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

fn format_throughput(kind: &str, count: u64, ns: f64) -> String {
    let secs = ns / 1_000_000_000.0;
    if secs <= 0.0 {
        return "N/A".to_string();
    }
    let speed = count as f64 / secs;
    if kind == "Elements" {
        if speed >= 1_000_000_000.0 {
            format!("{:.3} Gelem/s", speed / 1_000_000_000.0)
        } else if speed >= 1_000_000.0 {
            format!("{:.3} Melem/s", speed / 1_000_000.0)
        } else {
            format!("{:.3} Elem/s", speed)
        }
    } else if kind == "Bytes" {
        if speed >= 1_000_000_000.0 {
            format!("{:.3} GB/s", speed / 1_000_000_000.0)
        } else if speed >= 1_000_000.0 {
            format!("{:.3} MB/s", speed / 1_000_000.0)
        } else {
            format!("{:.3} B/s", speed)
        }
    } else {
        format!("{:.3} {}/s", speed, kind)
    }
}

#[derive(Debug, Clone, Copy)]
struct HostCapabilities {
    avx2: bool,
    fma: bool,
    avx512f: bool,
    avx512bw: bool,
    avx512vl: bool,
    avx512vnni: bool,
    amx_tile: bool,
    amx_bf16: bool,
    amx_int8: bool,
}

impl HostCapabilities {
    #[cfg(target_arch = "x86")]
    fn cpuid_leaf7() -> core::arch::x86::CpuidResult {
        core::arch::x86::__cpuid_count(7, 0)
    }

    #[cfg(target_arch = "x86_64")]
    fn cpuid_leaf7() -> core::arch::x86_64::CpuidResult {
        core::arch::x86_64::__cpuid_count(7, 0)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn detect() -> Self {
        let leaf7 = Self::cpuid_leaf7();
        Self {
            avx2: std::is_x86_feature_detected!("avx2"),
            fma: std::is_x86_feature_detected!("fma"),
            avx512f: std::is_x86_feature_detected!("avx512f"),
            avx512bw: std::is_x86_feature_detected!("avx512bw"),
            avx512vl: std::is_x86_feature_detected!("avx512vl"),
            avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
            amx_tile: (leaf7.edx & (1 << 24)) != 0,
            amx_bf16: (leaf7.edx & (1 << 22)) != 0,
            amx_int8: (leaf7.edx & (1 << 25)) != 0,
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    fn detect() -> Self {
        Self {
            avx2: false,
            fma: false,
            avx512f: false,
            avx512bw: false,
            avx512vl: false,
            avx512vnni: false,
            amx_tile: false,
            amx_bf16: false,
            amx_int8: false,
        }
    }

    fn runtime_backend(self) -> &'static str {
        if self.avx512f {
            "avx512"
        } else if self.avx2 && self.fma {
            "avx2"
        } else {
            "scalar"
        }
    }

    fn format_markdown(self) -> String {
        let mut enabled = Vec::new();
        for (name, enabled_flag) in [
            ("avx2", self.avx2),
            ("fma", self.fma),
            ("avx512f", self.avx512f),
            ("avx512bw", self.avx512bw),
            ("avx512vl", self.avx512vl),
            ("avx512vnni", self.avx512vnni),
            ("amx-tile", self.amx_tile),
            ("amx-bf16", self.amx_bf16),
            ("amx-int8", self.amx_int8),
        ] {
            if enabled_flag {
                enabled.push(name);
            }
        }
        if enabled.is_empty() {
            "none from tracked x86 feature set".to_string()
        } else {
            enabled.join(", ")
        }
    }
}

#[derive(PartialEq, PartialOrd)]
enum ParamValue {
    Numeric(f64),
    String(String),
    None,
}

impl Eq for ParamValue {}
impl Ord for ParamValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn parse_param(value_str: Option<&str>) -> ParamValue {
    let s = match value_str {
        Some(val) => val,
        None => return ParamValue::None,
    };
    if let Some(pct) = s.strip_prefix("density_") {
        if let Some(num) = pct.strip_suffix("pct") {
            if let Ok(n) = num.parse::<f64>() {
                return ParamValue::Numeric(n);
            }
        }
    }
    if let Some(pct) = s.strip_prefix("sparsity_") {
        if let Some(num) = pct.strip_suffix("pct") {
            if let Ok(n) = num.parse::<f64>() {
                return ParamValue::Numeric(n);
            }
        }
    }
    if let Ok(n) = s.parse::<f64>() {
        ParamValue::Numeric(n)
    } else {
        ParamValue::String(s.to_string())
    }
}

fn parse_result(
    est_path: &Path,
    bench_path: &Path,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let est_file = fs::File::open(est_path)?;
    let est: Estimates = serde_json::from_reader(est_file)?;

    let bench_file = fs::File::open(bench_path)?;
    let bench: Benchmark = serde_json::from_reader(bench_file)?;

    let time_ns = if let Some(slope) = est.slope {
        slope.point_estimate
    } else if let Some(mean) = est.mean {
        mean.point_estimate
    } else {
        return Err("No point estimate found for slope or mean".into());
    };

    let mut throughput_kind = None;
    let mut throughput_val = None;

    if let Some(tp_val) = bench.throughput {
        if let Some((kind, val)) = get_throughput_val(&tp_val) {
            throughput_kind = Some(kind);
            throughput_val = Some(val);
        }
    }

    Ok(BenchResult {
        group_id: bench.group_id,
        function_id: bench.function_id.unwrap_or_else(|| "default".to_string()),
        value_str: bench.value_str,
        time_ns,
        throughput_kind,
        throughput_val,
    })
}

fn find_criterion_results(dir: &Path, results: &mut Vec<BenchResult>) -> std::io::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    let is_new_dir = dir.file_name().and_then(|s| s.to_str()) == Some("new");
    let est_path = dir.join("estimates.json");
    let bench_path = dir.join("benchmark.json");

    if is_new_dir && est_path.is_file() && bench_path.is_file() {
        if let Ok(bench_res) = parse_result(&est_path, &bench_path) {
            results.push(bench_res);
        }
    } else {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                find_criterion_results(&path, results)?;
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn get_cpu_info() -> String {
    if let Ok(output) = Command::new("wmic").args(&["cpu", "get", "name"]).output() {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            let lines: Vec<&str> = stdout
                .lines()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty() && *s != "Name")
                .collect();
            if !lines.is_empty() {
                return lines[0].to_string();
            }
        }
    }
    "Unknown Windows CPU".to_string()
}

#[cfg(not(target_os = "windows"))]
fn get_cpu_info() -> String {
    if let Ok(output) = Command::new("lscpu").output() {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            for line in stdout.lines() {
                if line.starts_with("Model name:") {
                    return line.replace("Model name:", "").trim().to_string();
                }
            }
        }
    }
    "Unknown CPU".to_string()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let parse_only =
        args.contains(&"--parse-only".to_string()) || args.contains(&"--no-run".to_string());

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let target_dir = match env::var("CARGO_TARGET_DIR") {
        Ok(val) => PathBuf::from(val),
        Err(_) => workspace_root.join("target"),
    };
    let target_criterion = target_dir.join("criterion");

    if !parse_only {
        println!("Running workspace benchmarks via cargo bench...");
        let status = Command::new("cargo")
            .args(&["bench", "--workspace"])
            .current_dir(&workspace_root)
            .status()?;
        if !status.success() {
            eprintln!("Error: cargo bench execution failed");
            std::process::exit(1);
        }
    } else {
        println!(
            "Skipping benchmark run. Parsing existing results from: {:?}",
            target_criterion
        );
    }

    if !target_criterion.exists() {
        eprintln!(
            "Error: Criterion results directory not found at {:?}",
            target_criterion
        );
        std::process::exit(1);
    }

    let mut results = Vec::new();
    find_criterion_results(&target_criterion, &mut results)?;

    if results.is_empty() {
        eprintln!("Warning: No benchmark results found in target/criterion");
    }

    // Group by group_id
    let mut groups: BTreeMap<String, Vec<BenchResult>> = BTreeMap::new();
    for res in results {
        groups.entry(res.group_id.clone()).or_default().push(res);
    }

    let cpu = get_cpu_info();
    let capabilities = HostCapabilities::detect();
    let date_str = chrono::Local::now()
        .format("%Y-%m-%d %H:%M:%S %Z")
        .to_string();

    let mut markdown = String::new();
    markdown.push_str("# Hermes SIMD Benchmark Results\n\n");
    markdown.push_str(&format!("- **CPU**: {}\n", cpu));
    markdown.push_str(&format!(
        "- **Runtime backend selected by dense dispatch**: `{}`\n",
        capabilities.runtime_backend()
    ));
    markdown.push_str(&format!(
        "- **Detected benchmark-relevant ISA features**: {}\n",
        capabilities.format_markdown()
    ));
    markdown.push_str(&format!("- **Date**: {}\n\n", date_str));
    markdown.push_str("This file is dynamically generated by the benchmark runner. Performance values represent point estimates computed by Criterion. ISA-specific rows only validate features detected on this host; unsupported AVX-512/AMX paths require a matching runner.\n\n");

    for (group_name, mut items) in groups {
        markdown.push_str(&format!("## {}\n\n", group_name));

        // Sort items: parameter value first, then function name
        items.sort_by(|a, b| {
            let a_param = parse_param(a.value_str.as_deref());
            let b_param = parse_param(b.value_str.as_deref());
            match a_param.cmp(&b_param) {
                std::cmp::Ordering::Equal => a.function_id.cmp(&b.function_id),
                other => other,
            }
        });

        // Determine if there are parameters in this group
        let has_params = items.iter().any(|item| item.value_str.is_some());

        // Find baseline times for speedup calculations.
        // 1. Look for a group-wide baseline (no parameter)
        let mut group_wide_baseline: Option<BenchResult> = None;
        for item in &items {
            let is_baseline_candidate = item.function_id == "scalar_iter"
                || item.function_id == "dense_sum"
                || item.function_id == "dense_dot"
                || item.function_id == "dense_masked"
                || item.function_id == "dense"
                || item.function_id == "dispatch";

            if is_baseline_candidate && item.value_str.is_none() {
                if let Some(ref existing) = group_wide_baseline {
                    if existing.function_id != "scalar_iter" && item.function_id == "scalar_iter" {
                        group_wide_baseline = Some(item.clone());
                    }
                } else {
                    group_wide_baseline = Some(item.clone());
                }
            }
        }

        // 2. Look for parameter-specific baselines if no group-wide baseline exists
        let mut baselines = BTreeMap::new();
        if group_wide_baseline.is_none() {
            for item in &items {
                let param_key = item.value_str.clone().unwrap_or_else(|| "none".to_string());
                let is_baseline_candidate = item.function_id == "scalar_iter"
                    || item.function_id == "dense_sum"
                    || item.function_id == "dense_dot"
                    || item.function_id == "dense_masked"
                    || item.function_id == "dense"
                    || item.function_id == "dispatch";

                if is_baseline_candidate {
                    if let Some(existing) = baselines.get(&param_key) {
                        let existing_item: &BenchResult = existing;
                        if existing_item.function_id != "scalar_iter"
                            && item.function_id == "scalar_iter"
                        {
                            baselines.insert(param_key, item.clone());
                        }
                    } else {
                        baselines.insert(param_key, item.clone());
                    }
                }
            }
        }

        let has_speedup = group_wide_baseline.is_some() || !baselines.is_empty();

        if has_params {
            if has_speedup {
                markdown.push_str(
                    "| Function | Parameter | Time | Throughput | Speedup vs Baseline |\n",
                );
                markdown.push_str("|---|---|---|---|---|\n");
            } else {
                markdown.push_str("| Function | Parameter | Time | Throughput |\n");
                markdown.push_str("|---|---|---|---|\n");
            }
        } else {
            if has_speedup {
                markdown.push_str("| Function | Time | Throughput | Speedup vs Baseline |\n");
                markdown.push_str("|---|---|---|---|\n");
            } else {
                markdown.push_str("| Function | Time | Throughput |\n");
                markdown.push_str("|---|---|---|\n");
            }
        }

        for item in &items {
            let time_formatted = format_time(item.time_ns);
            let throughput_formatted = match (&item.throughput_kind, item.throughput_val) {
                (Some(kind), Some(val)) => format_throughput(kind, val, item.time_ns),
                _ => "N/A".to_string(),
            };

            let param_key = item.value_str.clone().unwrap_or_else(|| "none".to_string());
            let speedup_str = if let Some(ref base_item) = group_wide_baseline {
                if item.function_id == base_item.function_id {
                    format!("1.00x (Baseline: {})", base_item.function_id)
                } else {
                    let speedup = base_item.time_ns / item.time_ns;
                    format!("{:.2}x", speedup)
                }
            } else if let Some(base_item) = baselines.get(&param_key) {
                if item.function_id == base_item.function_id {
                    format!("1.00x (Baseline: {})", base_item.function_id)
                } else {
                    let speedup = base_item.time_ns / item.time_ns;
                    format!("{:.2}x", speedup)
                }
            } else {
                "-".to_string()
            };

            if has_params {
                let p = item.value_str.as_deref().unwrap_or("-");
                if has_speedup {
                    markdown.push_str(&format!(
                        "| {} | {} | {} | {} | {} |\n",
                        item.function_id, p, time_formatted, throughput_formatted, speedup_str
                    ));
                } else {
                    markdown.push_str(&format!(
                        "| {} | {} | {} | {} |\n",
                        item.function_id, p, time_formatted, throughput_formatted
                    ));
                }
            } else {
                if has_speedup {
                    markdown.push_str(&format!(
                        "| {} | {} | {} | {} |\n",
                        item.function_id, time_formatted, throughput_formatted, speedup_str
                    ));
                } else {
                    markdown.push_str(&format!(
                        "| {} | {} | {} |\n",
                        item.function_id, time_formatted, throughput_formatted
                    ));
                }
            }
        }
        markdown.push_str("\n");
    }

    let output_path = workspace_root.join("benchmarks_results.md");
    fs::write(&output_path, markdown)?;
    println!("Successfully wrote benchmark results to {:?}", output_path);

    Ok(())
}
