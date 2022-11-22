extern crate argminmax;

use argminmax::ArgMinMax;

use ndarray::{Array1, ArrayView1};

use super::generic::{min_max_generic, min_max_generic_parallel};

#[inline]
pub fn min_max_simd<T: Copy + PartialOrd>(arr: ArrayView1<T>, n_out: usize) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    min_max_generic(arr, n_out, |arr| arr.argminmax())
}

#[inline]
pub fn min_max_simd_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    min_max_generic_parallel(arr, n_out, |arr| arr.argminmax())
}

#[cfg(test)]
mod tests {
    use super::{min_max_simd, min_max_simd_parallel};
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

        let expected_indices = vec![0, 1, 24, 25, 48, 49, 72, 73, 96, 99];
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

        let expected_indices = vec![0, 1, 24, 25, 48, 49, 72, 73, 96, 99];
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
        for _ in 0..100 {
            let arr = get_array_f32(n);
            let idxs1 = min_max_simd(arr.view(), 100);
            let idxs2 = min_max_simd_parallel(arr.view(), 100);
            assert_eq!(idxs1, idxs2);
        }
    }
}
