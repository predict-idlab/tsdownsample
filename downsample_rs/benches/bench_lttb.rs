use downsample_rs::lttb as lttb_mod;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dev_utils::{config, utils};

fn lttb_f32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("lttb_scalx_f32", |b| {
        b.iter(|| {
            lttb_mod::lttb_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
            )
        })
    });
}
fn lttb_f32_random_array_50m(c: &mut Criterion) {
    let n = 50_000_000;
    let x = (0..n).map(|i| i as i32).collect::<Vec<i32>>();
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("lttb_scalx_50M_f32", |b| {
        b.iter(|| {
            lttb_mod::lttb_with_x(
                black_box(x.as_slice()),
                black_box(y.as_slice()),
                black_box(2_000),
            )
        })
    });
}

fn lttb_without_x_f32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("lttb_scal_f32", |b| {
        b.iter(|| lttb_mod::lttb_without_x(black_box(y.as_slice()), black_box(2_000)))
    });
}
fn lttb_without_x_f32_random_array_50m(c: &mut Criterion) {
    let n = 50_000_000;
    let y = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("lttb_scal_50M_f32", |b| {
        b.iter(|| lttb_mod::lttb_without_x(black_box(y.as_slice()), black_box(2_000)))
    });
}

criterion_group!(
    benches,
    // lttb_f32_random_array_long,
    lttb_f32_random_array_50m,
    // lttb_without_x_f32_random_array_long,
    lttb_without_x_f32_random_array_50m,
);
criterion_main!(benches);
