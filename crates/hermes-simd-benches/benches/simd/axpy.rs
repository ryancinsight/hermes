use criterion::{black_box, BatchSize, BenchmarkId, Criterion, Throughput};

pub fn bench<T, F>(
    criterion: &mut Criterion,
    group_name: &str,
    alpha: T,
    input_value: T,
    output_value: T,
    dispatch: F,
) where
    T: Copy + 'static,
    F: Copy + Fn(T, &[T], &mut [T]),
{
    let mut group = super::group::configured(criterion, group_name);
    for &size in &[3_usize, 7, 15, 31] {
        let input = vec![input_value; size];
        let initial_output = vec![output_value; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter_batched_ref(
                || initial_output.clone(),
                |output| {
                    dispatch(alpha, black_box(&input), black_box(output.as_mut_slice()));
                    black_box(output.as_slice());
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}
