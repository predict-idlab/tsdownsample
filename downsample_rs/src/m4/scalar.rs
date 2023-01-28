extern crate argminmax;

use argminmax::{ScalarArgMinMax, SCALAR};

use ndarray::{Array1, ArrayView1};
use num_traits::{AsPrimitive, FromPrimitive};

use super::super::searchsorted::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::super::types::Num;
use super::generic::{m4_generic, m4_generic_parallel};
use super::generic::{m4_generic_with_x, m4_generic_with_x_parallel};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn m4_scalar_with_x<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    Tx: Num + FromPrimitive + AsPrimitive<f64>,
    Ty: Copy + PartialOrd,
{
    assert_eq!(n_out % 4, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator(x, n_out / 4);
    m4_generic_with_x(arr, bin_idx_iterator, n_out, SCALAR::argminmax)
}

// ----------- WITHOUT X

pub fn m4_scalar_without_x<T: Copy + PartialOrd>(arr: ArrayView1<T>, n_out: usize) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<T>,
{
    assert_eq!(n_out % 4, 0);
    m4_generic(arr, n_out, SCALAR::argminmax)
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn m4_scalar_with_x_parallel<Tx, Ty>(
    x: ArrayView1<Tx>,
    arr: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
    Tx: Num + FromPrimitive + AsPrimitive<f64> + Send + Sync,
    Ty: Copy + PartialOrd + Send + Sync,
{
    assert_eq!(n_out % 4, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out / 4);
    m4_generic_with_x_parallel(arr, bin_idx_iterator, n_out, SCALAR::argminmax)
}

// ----------- WITHOUT X

pub fn m4_scalar_without_x_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<T>,
{
    assert_eq!(n_out % 4, 0);
    m4_generic_parallel(arr, n_out, SCALAR::argminmax)
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        m4_scalar_with_x, m4_scalar_with_x_parallel, m4_scalar_without_x,
        m4_scalar_without_x_parallel,
    };
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Array1<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    #[test]
    fn test_m4_scalar_without_x_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_scalar_without_x(arr.view(), 12);
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
    fn test_m4_scalar_without_x_parallel_correct() {
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_scalar_without_x_parallel(arr.view(), 12);
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
    fn test_m4_scalar_with_x_correct() {
        let x = (0..100).collect::<Vec<i32>>();
        let x = Array1::from(x);
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_scalar_with_x(x.view(), arr.view(), 12);
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
    fn test_m4_scalar_with_x_parallel_correct() {
        let x = (0..100).collect::<Vec<i32>>();
        let x = Array1::from(x);
        let arr = (0..100).map(|x| x as f32).collect::<Vec<f32>>();
        let arr = Array1::from(arr);

        let sampled_indices = m4_scalar_with_x_parallel(x.view(), arr.view(), 12);
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
    fn test_m4_scalar_with_x_gap() {
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

        let sampled_indices = m4_scalar_with_x(x.view(), arr.view(), 20);
        assert_eq!(sampled_indices.len(), 16); // One full gap
        let expected_indices = vec![0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99];
        assert_eq!(sampled_indices, Array1::from(expected_indices));

        // Increment the second half of the array by 50 again
        let x = x
            .iter()
            .map(|x| if *x > 101 { *x + 50 } else { *x })
            .collect::<Vec<i32>>();
        let x = Array1::from(x);

        let sampled_indices = m4_scalar_with_x(x.view(), arr.view(), 20);
        assert_eq!(sampled_indices.len(), 17); // Gap with 1 value
        let expected_indices = vec![
            0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99, 100,
        ];
    }

    #[test]
    fn test_m4_scalar_with_x_gap_parallel() {
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

        let sampled_indices = m4_scalar_with_x_parallel(x.view(), arr.view(), 20);
        assert_eq!(sampled_indices.len(), 16); // One full gap
        let expected_indices = vec![0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99];
        assert_eq!(sampled_indices, Array1::from(expected_indices));

        // Increment the second half of the array by 50 again
        let x = x
            .iter()
            .map(|x| if *x > 101 { *x + 50 } else { *x })
            .collect::<Vec<i32>>();
        let x = Array1::from(x);

        let sampled_indices = m4_scalar_with_x_parallel(x.view(), arr.view(), 20);
        assert_eq!(sampled_indices.len(), 17); // Gap with 1 value
        let expected_indices = vec![
            0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99, 100,
        ];
    }

    #[test]
    fn test_many_random_runs_correct() {
        let n: usize = 20_001; // not 20_000 because then the last bin is not "full"
        let x = (0..n as i32).collect::<Vec<i32>>();
        let x = Array1::from(x);
        for _ in 0..100 {
            let arr = get_array_f32(n);
            let idxs1 = m4_scalar_without_x(arr.view(), 100);
            let idxs2 = m4_scalar_without_x_parallel(arr.view(), 100);
            let idxs3 = m4_scalar_with_x(x.view(), arr.view(), 100);
            let idxs4 = m4_scalar_with_x_parallel(x.view(), arr.view(), 100);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4);
        }
    }
}
