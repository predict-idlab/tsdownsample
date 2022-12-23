extern crate argminmax;

use argminmax::ArgMinMax;

use ndarray::{Array1, ArrayView1};
use std::ops::{Add, Div, Mul, Sub};

use super::super::types::{FromUsize, Num};
use super::super::utils::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::generic::{min_max_generic, min_max_generic_parallel};
use super::generic::{min_max_generic_with_x, min_max_generic_with_x_parallel};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITHOUT X

#[inline]
pub fn min_max_simd<T: Copy + PartialOrd>(arr: ArrayView1<T>, n_out: usize) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    assert_eq!(n_out % 2, 0);
    min_max_generic(arr, n_out, |arr| arr.argminmax())
}

// ----------- WITH X

pub fn min_max_simd_with_x<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    Tx: Num + FromUsize,
    Ty: Copy + PartialOrd,
{
    assert_eq!(n_out % 2, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator(x, n_out / 2);
    min_max_generic_with_x(arr, bin_idx_iterator, n_out, |arr| arr.argminmax())
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITHOUT X

#[inline]
pub fn min_max_simd_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    assert_eq!(n_out % 2, 0);
    min_max_generic_parallel(arr, n_out, |arr| arr.argminmax())
}

// ----------- WITH X

pub fn min_max_simd_with_x_parallel<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    Tx: Num + FromUsize + Send + Sync,
    Ty: Copy + PartialOrd + Send + Sync,
{
    assert_eq!(n_out % 2, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out / 2);
    min_max_generic_with_x_parallel(arr, bin_idx_iterator, n_out, |arr| arr.argminmax())
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        min_max_simd, min_max_simd_parallel, min_max_simd_with_x, min_max_simd_with_x_parallel,
    };
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    #[test]
    fn test_min_max_simd_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_simd(arr.view(), 10);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_min_max_simd_parallel_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_simd_parallel(arr.view(), 10);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_min_max_simd_with_x_correct() {
        // 101 bc arr ... TODO
        let x = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let x = Array1::from(x);
        let arr = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_simd_with_x(x.view(), arr.view(), 10);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_min_max_simd_with_x_parallel_correct() {
        let x = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let x = Array1::from(x);
        let arr = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_simd_with_x_parallel(x.view(), arr.view(), 10);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
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
            let idxs1 = min_max_simd(arr.view(), n_out);
            let idxs2 = min_max_simd_parallel(arr.view(), n_out);
            let idxs3 = min_max_simd_with_x(x.view(), arr.view(), n_out);
            let idxs4 = min_max_simd_with_x_parallel(x.view(), arr.view(), n_out);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4);
        }
    }
}
