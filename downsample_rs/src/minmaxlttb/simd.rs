use super::super::minmax;
use super::generic::{
    minmaxlttb_generic, minmaxlttb_generic_without_x, MinMaxFunctionWithX, MinMaxFunctionWithoutX,
};

use super::super::helpers::Average;
use super::super::types::Num;
use ndarray::{Array1, ArrayView1};
use num_traits::{AsPrimitive, FromPrimitive};

use argminmax::ArgMinMax;

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn minmaxlttb_simd_with_x<
    Tx: Num + FromPrimitive + AsPrimitive<f64>,
    Ty: Num + AsPrimitive<f64>,
>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    for<'a> ArrayView1<'a, Ty>: Average,
{
    minmaxlttb_generic(
        x,
        y,
        n_out,
        minmax_ratio,
        None,
        MinMaxFunctionWithX::Serial(minmax::min_max_simd_with_x),
    )
}

// ----------- WITHOUT X

pub fn minmaxlttb_simd_without_x<Ty: Num + AsPrimitive<f64>>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    for<'a> ArrayView1<'a, Ty>: Average,
{
    minmaxlttb_generic_without_x(
        y,
        n_out,
        minmax_ratio,
        None,
        MinMaxFunctionWithoutX::Serial(minmax::min_max_simd_without_x),
    )
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn minmaxlttb_simd_with_x_parallel<Tx, Ty>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    n_threads: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    for<'a> ArrayView1<'a, Ty>: Average,
    Tx: Num + FromPrimitive + AsPrimitive<f64> + Send + Sync,
    Ty: Num + AsPrimitive<f64> + Send + Sync,
{
    minmaxlttb_generic(
        x,
        y,
        n_out,
        minmax_ratio,
        Some(n_threads),
        MinMaxFunctionWithX::Parallel(minmax::min_max_simd_with_x_parallel),
    )
}

// ----------- WITHOUT X

pub fn minmaxlttb_simd_without_x_parallel<Ty: Num + AsPrimitive<f64> + Send + Sync>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    n_threads: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    for<'a> ArrayView1<'a, Ty>: Average,
{
    minmaxlttb_generic_without_x(
        y,
        n_out,
        minmax_ratio,
        Some(n_threads),
        MinMaxFunctionWithoutX::Parallel(minmax::min_max_simd_without_x_parallel),
    )
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use super::{minmaxlttb_simd_with_x, minmaxlttb_simd_without_x};
    use super::{minmaxlttb_simd_with_x_parallel, minmaxlttb_simd_without_x_parallel};
    use ndarray::{array, Array1};

    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    // Template for the n_threads matrix
    #[template]
    #[rstest]
    #[case(1)]
    #[case(utils::get_all_threads() / 2)]
    #[case(utils::get_all_threads())]
    #[case(utils::get_all_threads() * 2)]
    fn threads(#[case] n_threads: usize) {}

    #[test]
    fn test_minmaxlttb_with_x() {
        let x = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_simd_with_x(x.view(), y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_minmaxlttb_without_x() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_simd_without_x(y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[apply(threads)]
    fn test_minmaxlttb_with_x_parallel(n_threads: usize) {
        let x = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_simd_with_x_parallel(x.view(), y.view(), 4, 2, n_threads);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[apply(threads)]
    fn test_minmaxlttb_without_x_parallel(n_threads: usize) {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_simd_without_x_parallel(y.view(), 4, 2, n_threads);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[apply(threads)]
    fn test_many_random_runs_same_output(n_threads: usize) {
        let n: usize = 20_000;
        let n_out: usize = 100;
        let minmax_ratio: usize = 5;
        for _ in 0..100 {
            // TODO: test with x
            let arr = get_array_f32(n);
            let idxs1 = minmaxlttb_simd_without_x(arr.view(), n_out, minmax_ratio);
            let idxs2 =
                minmaxlttb_simd_without_x_parallel(arr.view(), n_out, minmax_ratio, n_threads);
            assert_eq!(idxs1, idxs2);
        }
    }
}
