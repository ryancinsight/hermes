use criterion::{black_box, BenchmarkId, Criterion, Throughput};

#[inline(never)]
fn scalar_dot<T>(a: &[T], b: &[T], zero: T) -> T
where
    T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    a.iter().zip(b.iter()).fold(zero, |acc, (&x, &y)| {
        black_box(acc + black_box(x) * black_box(y))
    })
}

pub fn bench<T, F>(
    c: &mut Criterion,
    group_name: &str,
    a_value: T,
    b_value: T,
    zero: T,
    dispatch: F,
) where
    T: Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + 'static,
    F: Copy + Fn(&[T], &[T]) -> T,
{
    let mut group = c.benchmark_group(group_name);
    for &size in &[256usize, 1024, 16384, 65536] {
        let a = vec![a_value; size];
        let b = vec![b_value; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("scalar_iter", size),
            &size,
            |bencher, _| bencher.iter(|| scalar_dot(black_box(&a), black_box(&b), zero)),
        );
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| dispatch(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}
