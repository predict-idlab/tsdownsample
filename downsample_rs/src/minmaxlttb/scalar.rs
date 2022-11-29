use super::super::lttb::utils::Num;
use super::super::minmax;
use super::generic::{minmaxlttb_generic, minmaxlttb_generic_without_x};
use ndarray::{Array1, ArrayView1};

extern crate argminmax;
use argminmax::{ScalarArgMinMax, SCALAR};

// ----------------------------------- NON-PARALLEL ------------------------------------

pub fn minmaxlttb_scalar<Tx: Num, Ty: Num + PartialOrd>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic(x, y, n_out, minmax_ratio, minmax::min_max_scalar)
}

pub fn minmaxlttb_scalar_without_x<Ty: Num + PartialOrd>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic_without_x(y, n_out, minmax_ratio, minmax::min_max_scalar)
}

// ------------------------------------- PARALLEL --------------------------------------

pub fn minmaxlttb_scalar_parallel<Tx: Num + Send + Sync, Ty: Num + PartialOrd + Send + Sync>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic(x, y, n_out, minmax_ratio, minmax::min_max_scalar_parallel)
}

pub fn minmaxlttb_scalar_without_x_parallel<Ty: Num + PartialOrd + Send + Sync>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic_without_x(y, n_out, minmax_ratio, minmax::min_max_scalar_parallel)
}

// ---- TEST

#[cfg(test)]
mod tests {
    use super::{minmaxlttb_scalar, minmaxlttb_scalar_without_x};
    use super::{minmaxlttb_scalar_parallel, minmaxlttb_scalar_without_x_parallel};
    use ndarray::{array, s, Array1};

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    #[test]
    fn test_minmaxlttb() {
        let x = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar(x.view(), y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_minmaxlttb_without_x() {
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_without_x(y.view(), 4, 2);
        assert_eq!(sampled_indices, array![0, 1, 5, 9]);
    }

    #[test]
    fn test_minmaxlttb_parallel() {
        let x = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let sampled_indices = minmaxlttb_scalar_parallel(x.view(), y.view(), 4, 2);
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
