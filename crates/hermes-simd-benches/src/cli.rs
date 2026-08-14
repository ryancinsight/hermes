use std::path::PathBuf;

const DEFAULT_REGRESSION_THRESHOLD: f64 = 1.10;

#[derive(Debug, Clone)]
pub(crate) struct BenchRunnerArgs {
    pub(crate) parse_only: bool,
    pub(crate) write_baseline: bool,
    pub(crate) check_regressions: bool,
    pub(crate) threshold: f64,
    pub(crate) baseline_path: Option<PathBuf>,
}

impl BenchRunnerArgs {
    pub(crate) fn parse(
        args: impl IntoIterator<Item = String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut parsed = Self {
            parse_only: false,
            write_baseline: false,
            check_regressions: false,
            threshold: DEFAULT_REGRESSION_THRESHOLD,
            baseline_path: None,
        };

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--parse-only" | "--no-run" => parsed.parse_only = true,
                "--write-baseline" => parsed.write_baseline = true,
                "--check-regressions" => parsed.check_regressions = true,
                "--threshold" => {
                    let value = args
                        .next()
                        .ok_or("--threshold requires a floating-point value")?;
                    parsed.threshold = value.parse()?;
                    if parsed.threshold < 1.0 {
                        return Err("--threshold must be >= 1.0".into());
                    }
                }
                "--baseline" => {
                    let value = args.next().ok_or("--baseline requires a path")?;
                    parsed.baseline_path = Some(PathBuf::from(value));
                }
                "--help" | "-h" => {
                    return Err(Self::usage().into());
                }
                other => return Err(format!("unknown argument: {other}\n{}", Self::usage()).into()),
            }
        }

        Ok(parsed)
    }

    fn usage() -> &'static str {
        "usage: run-benches [--parse-only|--no-run] [--write-baseline] \
         [--check-regressions] [--threshold <ratio>] [--baseline <path>]"
    }
}

#[cfg(test)]
mod tests {
    use super::BenchRunnerArgs;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "The CLI parser test compares an exact threshold literal"
    )]
    fn parses_regression_threshold_and_baseline_path() {
        let args = BenchRunnerArgs::parse([
            "--parse-only".to_string(),
            "--check-regressions".to_string(),
            "--threshold".to_string(),
            "1.25".to_string(),
            "--baseline".to_string(),
            "baseline.json".to_string(),
        ])
        .unwrap();

        assert!(args.parse_only);
        assert!(args.check_regressions);
        assert_eq!(args.threshold, 1.25);
        assert_eq!(
            args.baseline_path.unwrap(),
            std::path::PathBuf::from("baseline.json")
        );
    }

    #[test]
    fn rejects_thresholds_below_one() {
        let err = BenchRunnerArgs::parse(["--threshold".to_string(), "0.99".to_string()])
            .unwrap_err()
            .to_string();

        assert!(err.contains("--threshold must be >= 1.0"));
    }
}
