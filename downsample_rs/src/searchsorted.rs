use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use super::types::Num;
use super::POOL;
use num_traits::{AsPrimitive, FromPrimitive};

// ---------------------- Binary search ----------------------

/// Binary search for the index position of the given value in the given array.
/// The array must be sorted in ascending order and contain no duplicates.
///
/// Complies with the Python bisect function
/// https://docs.python.org/3/library/bisect.html#bisect.bisect
///
// #[inline(always)]
fn binary_search<T: Copy + PartialOrd>(arr: &[T], value: T, left: usize, right: usize) -> usize {
    let mut size: usize = right - left;
    let mut left: usize = left;
    let mut right: usize = right;
    // Return the index where the value is >= arr[index] and arr[index-1] < value
    while left < right {
        let mid = left + size / 2;
        if arr[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
        size = right - left;
    }
    if arr[left] <= value {
        left + 1
    } else {
        left
    }
}

/// Binary search for the index position of the given value in the given array.
/// The array must be sorted in ascending order and contain no duplicates.
///
/// The mid index is pre-guessed to speed up the search.
///
/// Complies with the Python bisect function
/// https://docs.python.org/3/library/bisect.html#bisect.bisect
///
// #[inline(always)]
fn binary_search_with_mid<T: Copy + PartialOrd>(
    arr: &[T],
    value: T,
    left: usize,
    right: usize,
    mid: usize,
) -> usize {
    assert!(mid >= left || mid <= right);
    let mut left: usize = left;
    let mut right: usize = right;
    let mut mid: usize = mid;
    // Return the index where the value is <= arr[index] and arr[index+1] < value
    while left < right {
        if arr[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
        let size = right - left;
        mid = left + size / 2;
    }
    if arr[left] <= value {
        left + 1
    } else {
        left
    }
}

// ------------------- Equidistant binning --------------------

// --- Sequential version

pub(crate) fn get_equidistant_bin_idx_iterator<T>(
    arr: &[T],
    nb_bins: usize,
) -> impl Iterator<Item = Option<(usize, usize)>> + '_
where
    T: Num + FromPrimitive + AsPrimitive<f64>,
{
    assert!(nb_bins >= 2);
    // 1. Compute the step between each bin
    // Divide by nb_bins to avoid overflow!
    let val_step: f64 =
        (arr[arr.len() - 1].as_() / nb_bins as f64) - (arr[0].as_() / nb_bins as f64);
    // Estimate the step between each index (used to pre-guess the mid index)
    let idx_step: usize = arr.len() / nb_bins;

    // 2. The moving index & value
    let arr0: f64 = arr[0].as_(); // The first value of the array
    let mut idx: usize = 0; // Index of the search value

    // 3. Iterate over the bins
    (0..nb_bins).map(move |i| {
        let start_idx: usize = idx; // Start index of the bin (previous end index)

        // Update the search value
        let search_value: T = T::from_f64(arr0 + val_step * (i + 1) as f64).unwrap();
        if arr[start_idx] >= search_value {
            // If the first value of the bin is already >= the search value,
            // then the bin is empty.
            return None;
        }
        // Update the pre-guess index
        let mid: usize = std::cmp::min(idx + idx_step, arr.len() - 2);
        // TODO: Implementation WITHOUT pre-guessing mid is slower!!
        idx = binary_search_with_mid(arr, search_value, idx, arr.len() - 1, mid); // End index of the bin
        Some((start_idx, idx))
    })
}

// --- Parallel version

#[inline(always)]
fn sequential_add_mul(start_val: f64, add_val: f64, mul: usize) -> f64 {
    // start_val + add_val * mul will sometimes overflow when add_val * mul is
    // larger than the largest positive f64 number.
    // This code should not fail when: (f64::MAX - start_val) < (add_val * mul).
    //   -> Note that f64::MAX - start_val can be up to 2 * f64::MAX.
    let mul_2: usize = mul / 2;
    start_val + add_val * mul_2 as f64 + add_val * (mul - mul_2) as f64
}

pub(crate) fn get_equidistant_bin_idx_iterator_parallel<T>(
    arr: &[T],
    nb_bins: usize,
) -> impl IndexedParallelIterator<Item = impl Iterator<Item = Option<(usize, usize)>> + '_> + '_
where
    T: Num + FromPrimitive + AsPrimitive<f64> + Sync + Send,
{
    assert!(nb_bins >= 2);
    // 1. Compute the step between each bin
    // Divide by nb_bins to avoid overflow!
    let val_step: f64 =
        (arr[arr.len() - 1].as_() / nb_bins as f64) - (arr[0].as_() / nb_bins as f64);
    let arr0: f64 = arr[0].as_(); // The first value of the array
                                  // 2. Compute the number of threads & bins per thread
    let n_threads = std::cmp::min(POOL.current_num_threads(), nb_bins);
    let nb_bins_per_thread = nb_bins / n_threads;
    let nb_bins_last_thread = nb_bins - nb_bins_per_thread * (n_threads - 1);
    // 3. Iterate over the number of threads
    // -> for each thread perform the binary search sorted with moving left and
    // yield the indices (using the same idea as for the sequential version)
    (0..n_threads).into_par_iter().map(move |i| {
        // The moving index & value (for the thread)
        let arr0_thr: f64 = sequential_add_mul(arr0, val_step, i * nb_bins_per_thread); // Search value
        let start_value: T = T::from_f64(arr0_thr).unwrap();
        // Search the start of the fist bin (of the thread)
        let mut idx: usize = 0; // Index of the search value
        if i > 0 {
            idx = binary_search(arr, start_value, 0, arr.len() - 1);
        }

        // The number of bins for the thread
        let nb_bins_thread = if i == n_threads - 1 {
            nb_bins_last_thread
        } else {
            nb_bins_per_thread
        };
        // Perform sequential binary search for the end of the bins (of the thread)
        (0..nb_bins_thread).map(move |i| {
            let start_idx: usize = idx; // Start index of the bin (previous end index)

            // Update the search value
            let search_value: T = T::from_f64(arr0_thr + val_step * (i + 1) as f64).unwrap();
            if arr[start_idx] >= search_value {
                // If the first value of the bin is already >= the search value,
                // then the bin is empty.
                return None;
            }
            idx = binary_search(arr, search_value, idx, arr.len() - 1); // End index of the bin
            Some((start_idx, idx))
        })
    })
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use super::*;

    use dev_utils::utils::get_random_array;

    // Template for nb_bins
    #[template]
    #[rstest]
    #[case(99)]
    #[case(100)]
    #[case(101)]
    fn nb_bins(#[case] nb_bins: usize) {}

    #[test]
    fn test_search_sorted_identicial_to_np_linspace_searchsorted() {
        // Create a 0..9999 array
        let arr: [u32; 10_000] = core::array::from_fn(|i| i.as_());
        assert!(arr.len() == 10_000);
        let iterator = get_equidistant_bin_idx_iterator(&arr, 4);
        // Check the iterator
        let mut idx: usize = 0;
        for bin in iterator {
            let (start_idx, end_idx) = bin.unwrap();
            assert!(start_idx == idx);
            assert!(end_idx == idx + 2_500);
            idx += 2_500;
        }
    }

    #[test]
    fn test_binary_search() {
        let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(binary_search(&arr, 0, 0, arr.len() - 1), 0);
        assert_eq!(binary_search(&arr, 1, 0, arr.len() - 1), 1);
        assert_eq!(binary_search(&arr, 2, 0, arr.len() - 1), 2);
        assert_eq!(binary_search(&arr, 3, 0, arr.len() - 1), 3);
        assert_eq!(binary_search(&arr, 4, 0, arr.len() - 1), 4);
        assert_eq!(binary_search(&arr, 5, 0, arr.len() - 1), 5);
        assert_eq!(binary_search(&arr, 6, 0, arr.len() - 1), 6);
        assert_eq!(binary_search(&arr, 7, 0, arr.len() - 1), 7);
        assert_eq!(binary_search(&arr, 8, 0, arr.len() - 1), 8);
        assert_eq!(binary_search(&arr, 9, 0, arr.len() - 1), 9);
        assert_eq!(binary_search(&arr, 10, 0, arr.len() - 1), 10);
        assert_eq!(binary_search(&arr, 11, 0, arr.len() - 1), 10);
    }

    #[test]
    fn test_binary_search_with_mid() {
        let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(binary_search_with_mid(&arr, 0, 0, arr.len() - 1, 0), 0);
        assert_eq!(binary_search_with_mid(&arr, 1, 0, arr.len() - 1, 0), 1);
        assert_eq!(binary_search_with_mid(&arr, 2, 0, arr.len() - 1, 1), 2);
        assert_eq!(binary_search_with_mid(&arr, 3, 0, arr.len() - 1, 2), 3);
        assert_eq!(binary_search_with_mid(&arr, 4, 0, arr.len() - 1, 3), 4);
        assert_eq!(binary_search_with_mid(&arr, 5, 0, arr.len() - 1, 4), 5);
        assert_eq!(binary_search_with_mid(&arr, 6, 0, arr.len() - 1, 5), 6);
        assert_eq!(binary_search_with_mid(&arr, 7, 0, arr.len() - 1, 6), 7);
        assert_eq!(binary_search_with_mid(&arr, 8, 0, arr.len() - 1, 7), 8);
        assert_eq!(binary_search_with_mid(&arr, 9, 0, arr.len() - 1, 8), 9);
        assert_eq!(binary_search_with_mid(&arr, 10, 0, arr.len() - 1, 9), 10);
        // this line causes the code to crash -> because value higher than arr[mid]
        // assert_eq!(binary_search_with_mid(&arr, 11, 0, arr.len() - 1, 9), 10);
    }

    #[test]
    fn test_get_equidistant_bin_idxs() {
        let expected_indices = vec![0, 4, 7];

        let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let bin_idxs_iter = get_equidistant_bin_idx_iterator(&arr, 3);
        let bin_idxs = bin_idxs_iter.map(|x| x.unwrap().0).collect::<Vec<usize>>();
        assert_eq!(bin_idxs, expected_indices);

        let bin_idxs_iter = get_equidistant_bin_idx_iterator_parallel(&arr, 3);
        let bin_idxs = bin_idxs_iter
            .map(|x| x.map(|x| x.unwrap().0).collect::<Vec<usize>>())
            .flatten()
            .collect::<Vec<usize>>();
        assert_eq!(bin_idxs, expected_indices);
    }

    #[apply(nb_bins)]
    fn test_many_random_same_result(nb_bins: usize) {
        let n = 5_000;

        for _ in 0..100 {
            let mut arr = get_random_array::<i32>(n, i32::MIN, i32::MAX);
            // Sort the array
            arr.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Calculate the bin indexes
            let bin_idxs_iter = get_equidistant_bin_idx_iterator(&arr[..], nb_bins);
            let bin_idxs = bin_idxs_iter.map(|x| x.unwrap().0).collect::<Vec<usize>>();

            // Calculate the bin indexes in parallel
            let bin_idxs_iter = get_equidistant_bin_idx_iterator_parallel(&arr[..], nb_bins);
            let bin_idxs_parallel = bin_idxs_iter
                .map(|x| x.map(|x| x.unwrap().0).collect::<Vec<usize>>())
                .flatten()
                .collect::<Vec<usize>>();

            // Check that the results are the same
            assert_eq!(bin_idxs, bin_idxs_parallel);
        }
    }
}
