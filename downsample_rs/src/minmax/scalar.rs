extern crate argminmax;

use argminmax::{ScalarArgMinMax, SCALAR};
use num_traits::{AsPrimitive, FromPrimitive};

use ndarray::{Array1, ArrayView1};

use super::super::searchsorted::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::super::types::Num;
use super::generic::{min_max_generic, min_max_generic_parallel};
use super::generic::{min_max_generic_with_x, min_max_generic_with_x_parallel};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn min_max_scalar_with_x<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    Tx: Num + FromPrimitive + AsPrimitive<f64>,
    Ty: Copy + PartialOrd,
{
    assert_eq!(n_out % 2, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator(x, n_out / 2);
    min_max_generic_with_x(arr, bin_idx_iterator, n_out, SCALAR::argminmax)
}

// ----------- WITHOUT X

pub fn min_max_scalar_without_x<T: Copy + PartialOrd>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<T>,
{
    assert_eq!(n_out % 2, 0);
    min_max_generic(arr, n_out, SCALAR::argminmax)
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn min_max_scalar_with_x_parallel<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    Tx: Num + FromPrimitive + AsPrimitive<f64> + Send + Sync,
    Ty: Copy + PartialOrd + Send + Sync,
{
    assert_eq!(n_out % 2, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out / 2);
    min_max_generic_with_x_parallel(arr, bin_idx_iterator, n_out, SCALAR::argminmax)
}

// ----------- WITHOUT X

pub fn min_max_scalar_without_x_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<T>,
{
    assert_eq!(n_out % 2, 0);
    min_max_generic_parallel(arr, n_out, SCALAR::argminmax)
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        min_max_scalar_with_x, min_max_scalar_with_x_parallel, min_max_scalar_without_x,
        min_max_scalar_without_x_parallel,
    };
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    #[test]
    fn test_min_max_scalar_without_x_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_without_x(arr.view(), 10);
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
    fn test_min_max_scalar_without_x_parallel_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_without_x_parallel(arr.view(), 10);
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
    fn test_min_max_scalar_with_x_correct() {
        let x = (0..101).collect::<Vec<i32>>();
        let x = Array1::from(x);
        let arr = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_with_x(x.view(), arr.view(), 10);
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
    fn test_min_max_scalar_with_x_parallel_correct() {
        let x = (0..101).collect::<Vec<i32>>();
        let x = Array1::from(x);
        let arr = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_with_x_parallel(x.view(), arr.view(), 10);
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
    fn test_min_max_scalar_with_x_gap() {
        // We will create a gap in the middle of the array
        let x = (0..101).collect::<Vec<i32>>();

        // Increment the second half of the array by 50
        let x = x
            .iter()
            .map(|x| if *x > 50 { *x + 50 } else { *x })
            .collect::<Vec<i32>>();
        let x = Array1::from(x);
        let arr = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_with_x(x.view(), arr.view(), 10);
        assert_eq!(sampled_indices.len(), 8); // One full gap
        let expected_indices = vec![0, 29, 30, 50, 51, 69, 70, 99];
        assert_eq!(sampled_indices, Array1::from(expected_indices));

        // Increment the second half of the array by 50 again
        let x = x
            .iter()
            .map(|x| if *x > 101 { *x + 50 } else { *x })
            .collect::<Vec<i32>>();
        let x = Array1::from(x);

        let sampled_indices = min_max_scalar_with_x(x.view(), arr.view(), 10);
        assert_eq!(sampled_indices.len(), 9); // Gap with 1 value
        let expected_indices = vec![0, 39, 40, 50, 51, 52, 59, 60, 99];
        assert_eq!(sampled_indices, Array1::from(expected_indices));
    }

    #[test]
    fn test_min_max_scalar_with_x_parallel_gap() {
        // Create a gap in the middle of the array
        let x = (0..101).collect::<Vec<i32>>();

        // Increment the second half of the array by 50
        let x = x
            .iter()
            .map(|x| if *x > 50 { *x + 50 } else { *x })
            .collect::<Vec<i32>>();
        let x = Array1::from(x);
        let arr = (0..101).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = min_max_scalar_with_x_parallel(x.view(), arr.view(), 10);
        assert_eq!(sampled_indices.len(), 8); // One full gap
        let expected_indices = vec![0, 29, 30, 50, 51, 69, 70, 99];
        assert_eq!(sampled_indices, Array1::from(expected_indices));

        // Increment the second half of the array by 50 again
        let x = x
            .iter()
            .map(|x| if *x > 101 { *x + 50 } else { *x })
            .collect::<Vec<i32>>();
        println!("{:?}", x);
        let x = Array1::from(x);

        let sampled_indices = min_max_scalar_with_x_parallel(x.view(), arr.view(), 10);
        assert_eq!(sampled_indices.len(), 9); // Gap with 1 value
        let expected_indices = vec![0, 39, 40, 50, 51, 52, 59, 60, 99];
        assert_eq!(sampled_indices, Array1::from(expected_indices));
    }

    #[test]
    fn test_many_random_runs_same_output() {
        let n: usize = 20_001;
        let n_out = 200;
        let x = (0..n as i32).collect::<Vec<i32>>();
        let x = Array1::from(x);
        for _ in 0..100 {
            let arr = get_array_f32(n);
            let idxs1 = min_max_scalar_without_x(arr.view(), n_out);
            let idxs2 = min_max_scalar_without_x_parallel(arr.view(), n_out);
            let idxs3 = min_max_scalar_with_x(x.view(), arr.view(), n_out);
            let idxs4 = min_max_scalar_with_x_parallel(x.view(), arr.view(), n_out);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4);
        }
    }
}
