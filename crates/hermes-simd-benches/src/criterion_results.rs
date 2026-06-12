use serde::{Deserialize, Serialize};
use std::path::Path;

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct BenchResult {
    pub(crate) group_id: String,
    pub(crate) function_id: String,
    pub(crate) value_str: Option<String>,
    pub(crate) time_ns: f64,
    pub(crate) throughput_kind: Option<String>,
    pub(crate) throughput_val: Option<u64>,
}

impl BenchResult {
    pub(crate) fn key(&self) -> BenchKey {
        BenchKey {
            group_id: self.group_id.clone(),
            function_id: self.function_id.clone(),
            value_str: self.value_str.clone(),
        }
    }
}

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct BenchKey {
    pub(crate) group_id: String,
    pub(crate) function_id: String,
    pub(crate) value_str: Option<String>,
}

fn get_throughput_val(val: &serde_json::Value) -> Option<(String, u64)> {
    val.as_object().and_then(|obj| {
        obj.iter()
            .find_map(|(key, value)| value.as_u64().map(|n| (key.clone(), n)))
    })
}

fn parse_result(
    est_path: &Path,
    bench_path: &Path,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    let est_file = std::fs::File::open(est_path)?;
    let est: Estimates = serde_json::from_reader(est_file)?;

    let bench_file = std::fs::File::open(bench_path)?;
    let bench: Benchmark = serde_json::from_reader(bench_file)?;

    let time_ns = if let Some(slope) = est.slope {
        slope.point_estimate
    } else if let Some(mean) = est.mean {
        mean.point_estimate
    } else {
        return Err("No point estimate found for slope or mean".into());
    };

    let (throughput_kind, throughput_val) = bench
        .throughput
        .as_ref()
        .and_then(get_throughput_val)
        .map_or((None, None), |(kind, val)| (Some(kind), Some(val)));

    Ok(BenchResult {
        group_id: bench.group_id,
        function_id: bench.function_id.unwrap_or_else(|| "default".to_string()),
        value_str: bench.value_str,
        time_ns,
        throughput_kind,
        throughput_val,
    })
}

pub(crate) fn find_criterion_results(
    dir: &Path,
    results: &mut Vec<BenchResult>,
) -> std::io::Result<()> {
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
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                find_criterion_results(&path, results)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::get_throughput_val;

    #[test]
    fn extracts_single_throughput_counter() {
        let val = serde_json::json!({ "Elements": 1024_u64 });

        assert_eq!(
            get_throughput_val(&val),
            Some(("Elements".to_string(), 1024))
        );
    }
}
