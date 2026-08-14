use criterion::{black_box, BenchmarkId, Criterion, Throughput};

pub fn bench<T, F>(c: &mut Criterion, group_name: &str, a_value: T, b_value: T, dispatch: F)
where
    T: Copy + 'static,
    F: Copy + Fn(&[T], &[T]) -> T,
{
    let mut group = super::group::configured(c, group_name);
    for &size in &[256usize, 16384, 65536] {
        let a = vec![a_value; size];
        let b = vec![b_value; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dispatch(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}
