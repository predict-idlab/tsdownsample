extern crate argminmax;

use argminmax::ArgMinMax;

use ndarray::{Array1, ArrayView1};
use std::ops::{Add, Div, Mul, Sub};

use super::super::types::{FromUsize, Num};
use super::super::utils::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::generic::{m4_generic, m4_generic_parallel};
use super::generic::{m4_generic_with_x, m4_generic_with_x_parallel};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITHOUT X

pub fn m4_simd<T: Copy + PartialOrd>(arr: ArrayView1<T>, n_out: usize) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    assert_eq!(n_out % 4, 0);
    m4_generic(arr, n_out, |arr| arr.argminmax())
}

// ----------- WITH X

pub fn m4_simd_with_x<Tx, Ty>(x: ArrayView1<Tx>, arr: ArrayView1<Ty>, n_out: usize) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    Tx: Num + FromUsize,
    Ty: Copy + PartialOrd,
{
    assert_eq!(n_out % 4, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator(x, n_out / 4);
    m4_generic_with_x(arr, bin_idx_iterator, n_out, |arr| arr.argminmax())
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITHOUT X

pub fn m4_simd_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    assert_eq!(n_out % 4, 0);
    m4_generic_parallel(arr, n_out, |arr| arr.argminmax())
}

// ----------- WITH X

pub fn m4_simd_with_x_parallel<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
    Tx: Num + FromUsize + Send + Sync,
    Ty: Copy + PartialOrd + Send + Sync,
{
    assert_eq!(n_out % 4, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out / 4);
    m4_generic_with_x_parallel(arr, bin_idx_iterator, n_out, |arr| arr.argminmax())
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use super::{m4_simd, m4_simd_parallel, m4_simd_with_x, m4_simd_with_x_parallel};
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    #[test]
    fn test_m4_simd_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_simd(arr.view(), 12);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 0, 32, 32, 33, 33, 65, 65, 66, 66, 98, 98];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_m4_simd_parallel_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_simd_parallel(arr.view(), 12);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 0, 32, 32, 33, 33, 65, 65, 66, 66, 98, 98];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_m4_simd_with_x_correct() {
        let x = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let x = Array1::from(x);
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_simd_with_x(x.view(), arr.view(), 12);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 0, 32, 32, 33, 33, 65, 65, 66, 66, 98, 98];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_m4_simd_with_x_parallel_correct() {
        let x = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let x = Array1::from(x);
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_simd_with_x_parallel(x.view(), arr.view(), 12);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 0, 32, 32, 33, 33, 65, 65, 66, 66, 98, 98];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_many_random_runs_correct() {
        let n = 20_001; // not 20_000 because then the last bin is not "full"
        let x = (0..n).map(|x| x as i32).collect::<Vec<i32>>();
        let x = Array1::from(x);
        for _ in 0..100 {
            let arr = get_array_f32(n);
            let idxs1 = m4_simd(arr.view(), 100);
            let idxs2 = m4_simd_parallel(arr.view(), 100);
            let idxs3 = m4_simd_with_x(x.view(), arr.view(), 100);
            let idxs4 = m4_simd_with_x_parallel(x.view(), arr.view(), 100);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4);
        }
    }
}
