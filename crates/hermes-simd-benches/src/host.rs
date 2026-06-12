use std::process::Command;

#[derive(Debug, Clone, Copy)]
pub(crate) struct HostCapabilities {
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
    pub(crate) fn detect() -> Self {
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
    pub(crate) fn detect() -> Self {
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

    pub(crate) fn runtime_backend(self) -> &'static str {
        if self.avx512f {
            "avx512"
        } else if self.avx2 && self.fma {
            "avx2"
        } else {
            "scalar"
        }
    }

    pub(crate) fn format_markdown(self) -> String {
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

#[cfg(target_os = "windows")]
pub(crate) fn get_cpu_info() -> String {
    if let Ok(output) = Command::new("wmic").args(["cpu", "get", "name"]).output() {
        if let Ok(stdout) = String::from_utf8(output.stdout) {
            let lines: Vec<&str> = stdout
                .lines()
                .map(str::trim)
                .filter(|s| !s.is_empty() && *s != "Name")
                .collect();
            if let Some(first) = lines.first() {
                return (*first).to_string();
            }
        }
    }
    "Unknown Windows CPU".to_string()
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn get_cpu_info() -> String {
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
