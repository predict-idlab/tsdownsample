extern crate argminmax;

use argminmax::ArgMinMax;

use ndarray::{Array1, ArrayView1};

use super::generic::{m4_generic, m4_generic_parallel};

// ----------------------------------- NON-PARALLEL ------------------------------------

pub fn m4_simd<T: Copy + PartialOrd>(arr: ArrayView1<T>, n_out: usize) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    m4_generic(arr, n_out, |arr| arr.argminmax())
}

// ------------------------------------- PARALLEL --------------------------------------

pub fn m4_simd_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, T>: ArgMinMax,
{
    m4_generic_parallel(arr, n_out, |arr| arr.argminmax())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    // extern crate dev_utils;
    // use dev_utils::utils;

    // fn get_array_f32(n: usize) -> Array1<f32> {
    //     utils::get_random_array(n, f32::MIN, f32::MAX)
    // }

    #[test]
    fn test_m4_simple_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_simd(arr.view(), 13);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 0, 32, 32, 33, 33, 65, 65, 66, 66, 98, 98, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }

    #[test]
    fn test_m4_sinple_parallel_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_simd_parallel(arr.view(), 13);
        let sampled_values = sampled_indices.mapv(|x| arr[x]);

        let expected_indices = vec![0, 0, 32, 32, 33, 33, 65, 65, 66, 66, 98, 98, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, Array1::from(expected_indices));
        assert_eq!(sampled_values, Array1::from(expected_values));
    }
}
