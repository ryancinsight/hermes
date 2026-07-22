use criterion::{measurement::WallTime, BenchmarkGroup, Criterion, SamplingMode};

pub fn configured<'criterion>(
    criterion: &'criterion mut Criterion,
    name: &str,
) -> BenchmarkGroup<'criterion, WallTime> {
    let mut group = criterion.benchmark_group(name);
    group.sampling_mode(SamplingMode::Flat);
    group
}
