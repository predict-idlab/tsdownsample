use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dev_utils::{config, utils};
use downsample_rs::fpcs as fpcs_mod;

fn fpcs_f32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();

    // Non parallel
    // --- without x
    c.bench_function("fpcs_vanilla_f32", |b| {
        b.iter(|| fpcs_mod::vanilla_fpcs_without_x(black_box(y.as_slice()), black_box(2_000)))
    });
    c.bench_function("fpcs_mm_f32", |b| {
        b.iter(|| fpcs_mod::fpcs_without_x(black_box(y.as_slice()), black_box(2_000)))
    });
    // --- with x
    c.bench_function("fpcs_mm_f32_x", |b| {
        b.iter(|| {
            fpcs_mod::fpcs_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
            )
        })
    });

    // Parallel
    // --- without x
    c.bench_function("fpcs_mm_f32_par", |b| {
        b.iter(|| fpcs_mod::fpcs_without_x_parallel(black_box(y.as_slice()), black_box(2_000)))
    });
    // --- with x
    c.bench_function("fpcs_mm_f32_x_par", |b| {
        b.iter(|| {
            fpcs_mod::fpcs_with_x_parallel(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
            )
        })
    });
}

fn fpcs_f32_random_array_50m(c: &mut Criterion) {
    let n = 50_000_000;
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();

    // Non parallel
    // --- without x
    c.bench_function("fpcs_vanilla_50M_f32", |b| {
        b.iter(|| fpcs_mod::vanilla_fpcs_without_x(black_box(y.as_slice()), black_box(2_000)))
    });
    c.bench_function("fpcs_mm_50M_f32", |b| {
        b.iter(|| fpcs_mod::fpcs_without_x(black_box(y.as_slice()), black_box(2_000)))
    });
    // --- with x
    c.bench_function("fpcs_mm_50M_f32_x", |b| {
        b.iter(|| {
            fpcs_mod::fpcs_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
            )
        })
    });
    // Parallel
    // --- without x
    c.bench_function("fpcs_mm_50M_f32_par", |b| {
        b.iter(|| fpcs_mod::fpcs_without_x_parallel(black_box(y.as_slice()), black_box(2_000)))
    });
    // --- with x
    c.bench_function("fpcs_mm_50M_f32_x_par", |b| {
        b.iter(|| {
            fpcs_mod::fpcs_with_x_parallel(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
            )
        })
    });
}

criterion_group!(
    benches,
    //
    fpcs_f32_random_array_long,
    fpcs_f32_random_array_50m,
);
criterion_main!(benches);
