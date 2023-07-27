use super::super::minmax;
use super::super::types::Num;
use super::generic::{
    minmaxlttb_generic, minmaxlttb_generic_without_x, MinMaxFunctionWithX, MinMaxFunctionWithoutX,
};
use argminmax::ArgMinMax;
use ndarray::{Array1, ArrayView1};
use num_traits::{AsPrimitive, FromPrimitive};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn minmaxlttb_scalar_with_x<
    Tx: Num + AsPrimitive<f64> + FromPrimitive,
    Ty: Num + AsPrimitive<f64>,
>(
    x: &[Tx],
    y: &[Ty],
    n_out: usize,
    minmax_ratio: usize,
) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
{
    minmaxlttb_generic(
        x,
        y,
        n_out,
        minmax_ratio,
        None,
        MinMaxFunctionWithX::Serial(minmax::min_max_scalar_with_x),
    )
}

// ----------- WITHOUT X

pub fn minmaxlttb_scalar_without_x<Ty: Num + AsPrimitive<f64>>(
    y: &[Ty],
    n_out: usize,
    minmax_ratio: usize,
) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
{
    minmaxlttb_generic_without_x(
        y,
        n_out,
        minmax_ratio,
        None,
        MinMaxFunctionWithoutX::Serial(minmax::min_max_scalar_without_x),
    )
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn minmaxlttb_scalar_with_x_parallel<
    Tx: Num + AsPrimitive<f64> + FromPrimitive + Send + Sync,
    Ty: Num + AsPrimitive<f64> + Send + Sync,
>(
    x: &[Tx],
    y: &[Ty],
    n_out: usize,
    minmax_ratio: usize,
    n_threads: usize,
) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
{
    minmaxlttb_generic(
        x,
        y,
        n_out,
        minmax_ratio,
        Some(n_threads),
        MinMaxFunctionWithX::Parallel(minmax::min_max_scalar_with_x_parallel),
    )
}

// ----------- WITHOUT X

pub fn minmaxlttb_scalar_without_x_parallel<Ty: Num + AsPrimitive<f64> + Send + Sync>(
    y: &[Ty],
    n_out: usize,
    minmax_ratio: usize,
    n_threads: usize,
) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
{
    minmaxlttb_generic_without_x(
        y,
        n_out,
        minmax_ratio,
        Some(n_threads),
        MinMaxFunctionWithoutX::Parallel(minmax::min_max_scalar_without_x_parallel),
    )
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use super::{minmaxlttb_scalar_with_x, minmaxlttb_scalar_without_x};
    use super::{minmaxlttb_scalar_with_x_parallel, minmaxlttb_scalar_without_x_parallel};
    use ndarray::{array, Array1};

    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Vec<f32> {
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
        let x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_with_x(&x, &y, 4, 2);
        assert_eq!(sampled_indices, vec![0, 1, 5, 9]);
    }

    #[test]
    fn test_minmaxlttb_without_x() {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_without_x(&y, 4, 2);
        assert_eq!(sampled_indices, vec![0, 1, 5, 9]);
    }

    #[apply(threads)]
    fn test_minmaxlttb_with_x_parallel(n_threads: usize) {
        let x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_with_x_parallel(&x, &y, 4, 2, n_threads);
        assert_eq!(sampled_indices, vec![0, 1, 5, 9]);
    }

    #[apply(threads)]
    fn test_minmaxlttb_without_x_parallel(n_threads: usize) {
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_without_x_parallel(&y, 4, 2, n_threads);
        assert_eq!(sampled_indices, vec![0, 1, 5, 9]);
    }

    #[apply(threads)]
    fn test_many_random_runs_same_output(n_threads: usize) {
        const N: usize = 20_000;
        const N_OUT: usize = 100;
        const MINMAX_RATIO: usize = 5;
        for _ in 0..100 {
            // TODO: test with x
            let arr = get_array_f32(N);
            let idxs1 = minmaxlttb_scalar_without_x(arr.as_slice(), N_OUT, MINMAX_RATIO);
            let idxs2 = minmaxlttb_scalar_without_x_parallel(
                arr.as_slice(),
                N_OUT,
                MINMAX_RATIO,
                n_threads,
            );
            assert_eq!(idxs1, idxs2);
        }
    }
}
