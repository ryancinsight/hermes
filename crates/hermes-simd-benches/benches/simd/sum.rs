use criterion::{black_box, BenchmarkId, Criterion, Throughput};

#[inline(never)]
fn scalar_sum<T: Copy + core::ops::Add<Output = T>>(data: &[T], zero: T) -> T {
    data.iter()
        .copied()
        .fold(zero, |acc, x| black_box(acc + black_box(x)))
}

pub fn bench<T, F>(c: &mut Criterion, group_name: &str, one: T, zero: T, dispatch: F)
where
    T: Copy + core::ops::Add<Output = T> + 'static,
    F: Copy + Fn(&[T]) -> T,
{
    let mut group = super::group::configured(c, group_name);
    for &size in &[256usize, 1024, 16384, 65536, 1 << 20] {
        let data = vec![one; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("scalar_iter", size), &size, |b, _| {
            b.iter(|| scalar_sum(black_box(&data), zero));
        });
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| dispatch(black_box(&data)));
        });
    }
    group.finish();
}
