#[macro_use]
extern crate criterion;
extern crate dev_utils;

use downsample_rs::m4 as m4_mod;

use criterion::{black_box, Criterion};
use dev_utils::{config, utils};

use ndarray::Array1;

fn m4_f32_random_array_long_single_core(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("m4_scal_f32", |b| {
        b.iter(|| m4_mod::m4_scalar_without_x(black_box(data.view()), black_box(2_000)))
    });
    c.bench_function("m4_simd_f32", |b| {
        b.iter(|| m4_mod::m4_simd_without_x(black_box(data.view()), black_box(2_000)))
    });
}

fn m4_f32_random_array_long_multi_core(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("m4_scal_p_f32", |b| {
        b.iter(|| m4_mod::m4_scalar_without_x_parallel(black_box(data.view()), black_box(2_000)))
    });
    c.bench_function("m4_simd_p_f32", |b| {
        b.iter(|| m4_mod::m4_simd_without_x_parallel(black_box(data.view()), black_box(2_000)))
    });
}

fn m4_f32_random_array_50M_single_core(c: &mut Criterion) {
    let n = 50_000_000;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let x = Array1::from((0..n).map(|i| i as i32).collect::<Vec<i32>>());
    c.bench_function("m4_scal_50M_f32", |b| {
        b.iter(|| m4_mod::m4_scalar_without_x(black_box(data.view()), black_box(2_000)))
    });
    c.bench_function("m4_simd_50M_f32", |b| {
        b.iter(|| m4_mod::m4_simd_without_x(black_box(data.view()), black_box(2_000)))
    });
    c.bench_function("m4_scalx_50M_f32", |b| {
        b.iter(|| {
            m4_mod::m4_scalar_with_x(
                black_box(x.view()),
                black_box(data.view()),
                black_box(2_000),
            )
        })
    });
    c.bench_function("m4_simdx_50M_f32", |b| {
        b.iter(|| {
            m4_mod::m4_simd_with_x(
                black_box(x.view()),
                black_box(data.view()),
                black_box(2_000),
            )
        })
    });
}

fn m4_f32_random_array_50M_multi_core(c: &mut Criterion) {
    let n = 50_000_000;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let x = Array1::from((0..n).map(|i| i as i32).collect::<Vec<i32>>());
    c.bench_function("m4_scal_p_50M_f32", |b| {
        b.iter(|| m4_mod::m4_scalar_without_x_parallel(black_box(data.view()), black_box(2_000)))
    });
    c.bench_function("m4_simd_p_50M_f32", |b| {
        b.iter(|| m4_mod::m4_simd_without_x_parallel(black_box(data.view()), black_box(2_000)))
    });
    c.bench_function("m4_scalx_p_50M_f32", |b| {
        b.iter(|| {
            m4_mod::m4_scalar_with_x_parallel(
                black_box(x.view()),
                black_box(data.view()),
                black_box(2_000),
            )
        })
    });
    c.bench_function("m4_simdx_p_50M_f32", |b| {
        b.iter(|| {
            m4_mod::m4_simd_with_x_parallel(
                black_box(x.view()),
                black_box(data.view()),
                black_box(2_000),
            )
        })
    });
}

// fn m4_f32_worst_case_array_long(c: &mut Criterion) {
//     let n = config::ARRAY_LENGTH_LONG;
//     let data = utils::get_worst_case_array::<f32>(n, 1.0);
//     c.bench_function("overlap_worst_long_f32", |b| {
//         b.iter(|| minmax_mod::min_max_overlap(black_box(data.view()), black_box(2_000)))
//     });
//     c.bench_function("simple_worst_long_f32", |b| {
//         b.iter(|| minmax_mod::min_max(black_box(data.view()), black_box(2_000)))
//     });
//     c.bench_function("simd_worst_long_f32", |b| {
//         b.iter(|| minmax_mod::min_max_simd_f32(black_box(data.view()), black_box(2_000)))
//     });
// }

criterion_group!(
    benches,
    // m4_f32_random_array_long_single_core,
    // m4_f32_random_array_long_multi_core,
    m4_f32_random_array_50M_single_core,
    m4_f32_random_array_50M_multi_core,
    // m4_f32_worst_case_array_long,
);
criterion_main!(benches);
