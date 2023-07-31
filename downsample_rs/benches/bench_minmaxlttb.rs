use downsample_rs::minmaxlttb as minmaxlttb_mod;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dev_utils::{config, utils};

const MINMAX_RATIO: usize = 30;

fn minmaxlttb_f32_random_array_long_single_core(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("mmlttb_scalx_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
            )
        })
    });
    c.bench_function("mlttb_simdx_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
            )
        })
    });
}

fn minmaxlttb_f32_random_array_long_multi_core(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let all_threads: usize = utils::get_all_threads();
    c.bench_function("mmlttb_scalx_p_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x_parallel(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
                black_box(all_threads),
            )
        })
    });
    c.bench_function("mlttb_simdx_p_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x_parallel(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
                black_box(all_threads),
            )
        })
    });
}

fn minmaxlttb_f32_random_array_50M_single_core(c: &mut Criterion) {
    let n = 50_000_000;
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("mlttb_scalx_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
            )
        })
    });
    c.bench_function("mlttb_simdx_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
            )
        })
    });
}

fn minmaxlttb_f32_random_array_50M_multi_core(c: &mut Criterion) {
    let n = 50_000_000;
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let all_threads: usize = utils::get_all_threads();
    c.bench_function("mlttb_scalx_p_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x_parallel(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
                black_box(all_threads),
            )
        })
    });
    c.bench_function("mlttb_simdx_p_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_with_x_parallel(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
                black_box(all_threads),
            )
        })
    });
}

fn minmaxlttb_without_x_f32_random_array_50M_single_core(c: &mut Criterion) {
    let n = 50_000_000;
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("mlttb_scal_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_without_x(
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
            )
        })
    });
    c.bench_function("mlttb_simd_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_without_x(
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
            )
        })
    });
}

fn minmaxlttb_without_x_f32_random_array_50M_multi_core(c: &mut Criterion) {
    let n = 50_000_000;
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    let all_threads: usize = utils::get_all_threads();
    c.bench_function("mlttb_scal_p_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_without_x_parallel(
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
                black_box(all_threads),
            )
        })
    });
    c.bench_function("mlttb_simd_p_50M_f32", |b| {
        b.iter(|| {
            minmaxlttb_mod::minmaxlttb_without_x_parallel(
                black_box(y.as_slice()),
                black_box(2_000),
                black_box(MINMAX_RATIO),
                black_box(all_threads),
            )
        })
    });
}

criterion_group!(
    benches,
    // minmaxlttb_f32_random_array_long_single_core,
    // minmaxlttb_f32_random_array_long_multi_core,
    minmaxlttb_f32_random_array_50M_single_core,
    minmaxlttb_f32_random_array_50M_multi_core,
    minmaxlttb_without_x_f32_random_array_50M_single_core,
    minmaxlttb_without_x_f32_random_array_50M_multi_core,
    // minmaxlttb_f32_random_array_100m
);
criterion_main!(benches);
