use super::super::helpers::Average;
use super::super::minmax;
use super::super::types::Num;
use super::generic::{minmaxlttb_generic, minmaxlttb_generic_without_x};
use ndarray::{Array1, ArrayView1};
use num_traits::{AsPrimitive, FromPrimitive};

extern crate argminmax;
use argminmax::{ScalarArgMinMax, SCALAR};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn minmaxlttb_scalar_with_x<
    Tx: Num + AsPrimitive<f64> + FromPrimitive,
    Ty: Num + AsPrimitive<f64>,
>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    for<'a> ArrayView1<'a, Ty>: Average,
{
    minmaxlttb_generic(x, y, n_out, minmax_ratio, minmax::min_max_scalar_with_x)
}

// ----------- WITHOUT X

pub fn minmaxlttb_scalar_without_x<Ty: Num + AsPrimitive<f64>>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    for<'a> ArrayView1<'a, Ty>: Average,
{
    minmaxlttb_generic_without_x(y, n_out, minmax_ratio, minmax::min_max_scalar_without_x)
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn minmaxlttb_scalar_with_x_parallel<
    Tx: Num + AsPrimitive<f64> + FromPrimitive + Send + Sync,
    Ty: Num + AsPrimitive<f64> + Send + Sync,
>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    for<'a> ArrayView1<'a, Ty>: Average,
{
    minmaxlttb_generic(
        x,
        y,
        n_out,
        minmax_ratio,
        minmax::min_max_scalar_with_x_parallel,
    )
}

// ----------- WITHOUT X

pub fn minmaxlttb_scalar_without_x_parallel<Ty: Num + AsPrimitive<f64> + Send + Sync>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    for<'a> ArrayView1<'a, Ty>: Average,
{
    minmaxlttb_generic_without_x(
        y,
        n_out,
        minmax_ratio,
        minmax::min_max_scalar_without_x_parallel,
    )
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use super::{minmaxlttb_scalar_with_x, minmaxlttb_scalar_without_x};
    use super::{minmaxlttb_scalar_with_x_parallel, minmaxlttb_scalar_without_x_parallel};
    use ndarray::{array, Array1};

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    #[test]
    fn test_minmaxlttb_with_x() {
        let x = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_with_x(x.view(), y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_minmaxlttb_without_x() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_without_x(y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_minmaxlttb_with_x_parallel() {
        let x = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_with_x_parallel(x.view(), y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_minmaxlttb_without_x_parallel() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_without_x_parallel(y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_many_random_runs_same_output() {
        let n = 20_000;
        for _ in 0..100 {
            let arr = get_array_f32(n);
            let idxs1 = minmaxlttb_scalar_without_x(arr.view(), 100, 5);
            let idxs2 = minmaxlttb_scalar_without_x_parallel(arr.view(), 100, 5);
            assert_eq!(idxs1, idxs2);
        }
    }
}
