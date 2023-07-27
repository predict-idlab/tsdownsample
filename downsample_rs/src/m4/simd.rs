use argminmax::ArgMinMax;

use num_traits::{AsPrimitive, FromPrimitive};

use super::super::searchsorted::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::super::types::Num;
use super::generic::{m4_generic, m4_generic_parallel};
use super::generic::{m4_generic_with_x, m4_generic_with_x_parallel};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn m4_simd_with_x<Tx, Ty>(x: &[Tx], arr: &[Ty], n_out: usize) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
    Tx: Num + FromPrimitive + AsPrimitive<f64>,
    Ty: Copy + PartialOrd,
{
    assert_eq!(n_out % 4, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator(x, n_out / 4);
    m4_generic_with_x(arr, bin_idx_iterator, n_out, |arr| arr.argminmax())
}

// ----------- WITHOUT X

pub fn m4_simd_without_x<T: Copy + PartialOrd>(arr: &[T], n_out: usize) -> Vec<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    assert_eq!(n_out % 4, 0);
    m4_generic(arr, n_out, |arr| arr.argminmax())
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn m4_simd_with_x_parallel<Tx, Ty>(
    x: &[Tx],
    arr: &[Ty],
    n_out: usize,
    n_threads: usize,
) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
    Tx: Num + FromPrimitive + AsPrimitive<f64> + Send + Sync,
    Ty: Copy + PartialOrd + Send + Sync,
{
    assert_eq!(n_out % 4, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out / 4, n_threads);
    m4_generic_with_x_parallel(arr, bin_idx_iterator, n_out, n_threads, |arr| {
        arr.argminmax()
    })
}

// ----------- WITHOUT X

pub fn m4_simd_without_x_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: &[T],
    n_out: usize,
    n_threads: usize,
) -> Vec<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    assert_eq!(n_out % 4, 0);
    m4_generic_parallel(arr, n_out, n_threads, |arr| arr.argminmax())
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use super::{m4_simd_with_x, m4_simd_without_x};
    use super::{m4_simd_with_x_parallel, m4_simd_without_x_parallel};
    use ndarray::Array1;

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
    fn test_m4_simd_without_x_correct() {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_simd_without_x(&arr, 12);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 0, 33, 33, 34, 34, 66, 66, 67, 67, 99, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[apply(threads)]
    fn test_m4_simd_without_x_parallel_correct(n_threads: usize) {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_simd_without_x_parallel(&arr, 12, n_threads);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 0, 33, 33, 34, 34, 66, 66, 67, 67, 99, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[test]
    fn test_m4_simd_with_x_correct() {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_simd_with_x(&x, &arr, 12);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 0, 33, 33, 34, 34, 66, 66, 67, 67, 99, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[apply(threads)]
    fn test_m4_simd_with_x_parallel_correct(n_threads: usize) {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_simd_with_x_parallel(&x, &arr, 12, n_threads);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 0, 33, 33, 34, 34, 66, 66, 67, 67, 99, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[test]
    fn test_m4_simd_with_x_gap() {
        // We will create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_simd_with_x(&x, &arr, 20);
        assert_eq!(sampled_indices.len(), 16); // One full gap
        let expected_indices = vec![0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|x| if x > 101 { x + 50 } else { x });

        let sampled_indices = m4_simd_with_x(&x, &arr, 20);
        assert_eq!(sampled_indices.len(), 17); // Gap with 1 value
        let expected_indices = vec![
            0, 0, 39, 39, 40, 40, 50, 50, 51, 52, 52, 59, 59, 60, 60, 99, 99,
        ];
        assert_eq!(sampled_indices, expected_indices);
    }

    #[apply(threads)]
    fn test_m4_simd_with_x_gap_parallel(n_threads: usize) {
        // We will create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_simd_with_x_parallel(&x, &arr, 20, n_threads);
        assert_eq!(sampled_indices.len(), 16); // One full gap
        let expected_indices = vec![0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|x| if x > 101 { x + 50 } else { x });

        let sampled_indices = m4_simd_with_x_parallel(&x, &arr, 20, n_threads);
        assert_eq!(sampled_indices.len(), 17); // Gap with 1 value
        let expected_indices = vec![
            0, 0, 39, 39, 40, 40, 50, 50, 51, 52, 52, 59, 59, 60, 60, 99, 99,
        ];
        assert_eq!(sampled_indices, expected_indices);
    }

    #[apply(threads)]
    fn test_many_random_runs_correct(n_threads: usize) {
        const N: usize = 20_003;
        const N_OUT: usize = 204;
        let x: [i32; N] = core::array::from_fn(|i| i.as_());
        for _ in 0..100 {
            let arr = get_array_f32(N);
            let idxs1 = m4_simd_without_x(arr.as_slice(), N_OUT);
            let idxs2 = m4_simd_without_x_parallel(arr.as_slice(), N_OUT, n_threads);
            let idxs3 = m4_simd_with_x(&x, arr.as_slice(), N_OUT);
            let idxs4 = m4_simd_with_x_parallel(&x, arr.as_slice(), N_OUT, n_threads);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4); // TODO: this should not fail when n_threads = 16
        }
    }
}
