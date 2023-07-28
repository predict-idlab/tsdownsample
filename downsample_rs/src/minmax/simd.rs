use argminmax::ArgMinMax;

use num_traits::{AsPrimitive, FromPrimitive};

use super::super::searchsorted::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::super::types::Num;
use super::generic::{min_max_generic, min_max_generic_parallel};
use super::generic::{min_max_generic_with_x, min_max_generic_with_x_parallel};

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn min_max_simd_with_x<Tx, Ty>(x: &[Tx], arr: &[Ty], n_out: usize) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
    Tx: Num + FromPrimitive + AsPrimitive<f64>,
    Ty: Copy + PartialOrd,
{
    assert_eq!(n_out % 2, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator(x, n_out / 2);
    min_max_generic_with_x(arr, bin_idx_iterator, n_out, |arr| arr.argminmax())
}

// ----------- WITHOUT X

pub fn min_max_simd_without_x<T: Copy + PartialOrd>(arr: &[T], n_out: usize) -> Vec<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    assert_eq!(n_out % 2, 0);
    min_max_generic(arr, n_out, |arr| arr.argminmax())
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn min_max_simd_with_x_parallel<Tx, Ty>(
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
    assert_eq!(n_out % 2, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out / 2, n_threads);
    min_max_generic_with_x_parallel(arr, bin_idx_iterator, n_out, n_threads, |arr| {
        arr.argminmax()
    })
}

// ----------- WITHOUT X

pub fn min_max_simd_without_x_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: &[T],
    n_out: usize,
    n_threads: usize,
) -> Vec<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    assert_eq!(n_out % 2, 0);
    min_max_generic_parallel(arr, n_out, n_threads, |arr| arr.argminmax())
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use super::{min_max_simd_with_x, min_max_simd_without_x};
    use super::{min_max_simd_with_x_parallel, min_max_simd_without_x_parallel};

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
    fn test_min_max_simd_without_x_correct() {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_simd_without_x(&arr, 10);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[apply(threads)]
    fn test_min_max_simd_without_x_parallel_correct(n_threads: usize) {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_simd_without_x_parallel(&arr, 10, n_threads);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[test]
    fn test_min_max_simd_with_x_correct() {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_simd_with_x(&x, &arr, 10);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[apply(threads)]
    fn test_min_max_simd_with_x_parallel_correct(n_threads: usize) {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_simd_with_x_parallel(&x, &arr, 10, n_threads);
        let sampled_values = sampled_indices
            .iter()
            .map(|x| arr[*x])
            .collect::<Vec<f32>>();

        let expected_indices = vec![0, 19, 20, 39, 40, 59, 60, 79, 80, 99];
        let expected_values = expected_indices
            .iter()
            .map(|x| *x as f32)
            .collect::<Vec<f32>>();

        assert_eq!(sampled_indices, expected_indices);
        assert_eq!(sampled_values, expected_values);
    }

    #[test]
    fn test_min_max_simd_with_x_gap() {
        // We will create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_simd_with_x(&x, &arr, 10);
        assert_eq!(sampled_indices.len(), 8); // One full gap
        let expected_indices = vec![0, 29, 30, 50, 51, 69, 70, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|i| if i > 101 { i + 50 } else { i });

        let sampled_indices = min_max_simd_with_x(&x, &arr, 10);
        assert_eq!(sampled_indices.len(), 9); // Gap with 1 value
        let expected_indices = vec![0, 39, 40, 50, 51, 52, 59, 60, 99];
        assert_eq!(sampled_indices, expected_indices);
    }

    #[apply(threads)]
    fn test_min_max_simd_with_x_parallel_gap(n_threads: usize) {
        // Create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_simd_with_x_parallel(&x, &arr, 10, n_threads);
        assert_eq!(sampled_indices.len(), 8); // One full gap
        let expected_indices = vec![0, 29, 30, 50, 51, 69, 70, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|i| if i > 101 { i + 50 } else { i });

        let sampled_indices = min_max_simd_with_x_parallel(&x, &arr, 10, n_threads);
        assert_eq!(sampled_indices.len(), 9); // Gap with 1 value
        let expected_indices = vec![0, 39, 40, 50, 51, 52, 59, 60, 99];
        assert_eq!(sampled_indices, expected_indices);
    }

    #[apply(threads)]
    fn test_many_random_runs_same_output(n_threads: usize) {
        const N: usize = 20_003;
        const N_OUT: usize = 202;
        let x: [i32; N] = core::array::from_fn(|i| i.as_());
        for _ in 0..100 {
            let mut arr = get_array_f32(N);
            arr[N - 1] = f32::INFINITY; // Make sure the last value is always the max
            let idxs1 = min_max_simd_without_x(arr.as_slice(), N_OUT);
            let idxs2 = min_max_simd_without_x_parallel(arr.as_slice(), N_OUT, n_threads);
            let idxs3 = min_max_simd_with_x(&x, arr.as_slice(), N_OUT);
            let idxs4 = min_max_simd_with_x_parallel(&x, arr.as_slice(), N_OUT, n_threads);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4);
        }
    }
}
