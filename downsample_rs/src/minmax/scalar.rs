extern crate argminmax;

use argminmax::{ScalarArgMinMax, SCALAR};

use ndarray::{s, Array1, ArrayView1};
use std::ops::{Add, Div, Mul, Sub};

use super::super::utils::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel, FromUsize,
};
use super::generic::{min_max_generic, min_max_generic_parallel};
use super::generic::{min_max_generic_with_x, min_max_generic_with_x_parallel};

// ------------------ WITHOUT X

pub fn min_max_scalar<T: Copy + PartialOrd>(arr: ArrayView1<T>, n_out: usize) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<T>,
{
    min_max_generic(arr, n_out, SCALAR::argminmax)
}

pub fn min_max_scalar_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<T>,
{
    min_max_generic_parallel(arr, n_out, SCALAR::argminmax)
}

// ------------------ WITH X

pub fn min_max_scalar_with_x<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    Tx: Copy + PartialOrd + FromUsize + Sub<Output = Tx> + Add<Output = Tx> + Div<Output = Tx>,
    Ty: Copy + PartialOrd,
{
    // Get the indices iterator of the equidistant (in x-data range) bins
    // -> slice x fron 1 to x.len() because we perform left searchsorted
    let bin_idx_iterator = get_equidistant_bin_idx_iterator(x.slice(s![1..x.len()]), n_out / 2 - 1);
    min_max_generic_with_x(arr, bin_idx_iterator, n_out, SCALAR::argminmax)
}

pub fn min_max_scalar_with_x_parallel<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    Tx: Copy
        + PartialOrd
        + FromUsize
        + Sub<Output = Tx>
        + Add<Output = Tx>
        + Div<Output = Tx>
        + Mul<Output = Tx>
        + Send
        + Sync,
    Ty: Copy + PartialOrd + Send + Sync,
{
    // Get the indices iterator of the equidistant (in x-data range) bins
    // -> slice x fron 1 to x.len() because we perform left searchsorted
    let bin_idx_iterator =
        get_equidistant_bin_idx_iterator_parallel(x.slice(s![1..x.len()]), n_out / 2 - 1);
    min_max_generic_with_x_parallel(arr, bin_idx_iterator, n_out, SCALAR::argminmax)
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        min_max_scalar, min_max_scalar_parallel, min_max_scalar_with_x,
        min_max_scalar_with_x_parallel,
    };
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    #[test]
    fn test_min_max_scalar_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar(arr.view(), 16);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 1, 14, 15, 28, 29, 42, 43, 56, 57, 70, 71, 84, 85, 98, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_min_max_scalar_parallel_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_parallel(arr.view(), 16);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 1, 14, 15, 28, 29, 42, 43, 56, 57, 70, 71, 84, 85, 98, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_min_max_scalar_with_x_correct() {
        let x = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let x = Array1::from(x);
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_with_x(x.view(), arr.view(), 16);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 1, 14, 15, 28, 29, 42, 43, 56, 57, 70, 71, 84, 85, 98, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_min_max_scalar_with_x_parallel_correct() {
        let x = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let x = Array1::from(x);
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_with_x_parallel(x.view(), arr.view(), 16);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 1, 14, 15, 28, 29, 42, 43, 56, 57, 70, 71, 84, 85, 98, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_many_random_runs_same_output() {
        let n = 20_000;
        let n_out = 198; // 198 bc (20_000 - 2) % 198 = 0 (-> no rounding errors)
        let x = (0..n).map(|x| x as i32).collect::<Vec<i32>>();
        let x = Array1::from(x);
        for _ in 0..100 {
            let arr = get_array_f32(n);
            let idxs1 = min_max_scalar(arr.view(), n_out);
            let idxs2 = min_max_scalar_parallel(arr.view(), n_out);
            let idxs3 = min_max_scalar_with_x(x.view(), arr.view(), n_out);
            let idxs4 = min_max_scalar_with_x_parallel(x.view(), arr.view(), n_out);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4);
        }
    }
}
