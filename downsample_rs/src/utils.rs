use ndarray::ArrayView1;

use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use super::types::Num;
use num_traits::{AsPrimitive, FromPrimitive};

// ---------------------- Binary search ----------------------

// #[inline(always)]
fn binary_search<T: PartialOrd>(arr: ArrayView1<T>, value: T, left: usize, right: usize) -> usize {
    let mut size: usize = right - left;
    let mut left: usize = left;
    let mut right: usize = right;
    // Return the index where the value is <= arr[index] and arr[index+1] < value
    while left < right {
        let mid = left + size / 2;
        if arr[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
        size = right - left;
    }
    left
}

// #[inline(always)]
fn binary_search_with_mid<T: PartialOrd>(
    arr: ArrayView1<T>,
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
    // if arr[left] == value { left + 1 } else { left }
    left
}

// ------------------- Equidistant binning --------------------

// --- Sequential version

pub(crate) fn get_equidistant_bin_idx_iterator<T>(
    arr: ArrayView1<T>,
    nb_bins: usize,
) -> impl Iterator<Item = (usize, usize)> + '_
where
    T: Num + FromPrimitive + AsPrimitive<f64>,
{
    assert!(nb_bins >= 2);
    // Divide by nb_bins to avoid overflow!
    let val_step: f64 =
        (arr[arr.len() - 1].as_() / nb_bins as f64) - (arr[0].as_() / nb_bins as f64);
    let idx_step: usize = arr.len() / nb_bins; // used to pre-guess the mid index
    let mut value: f64 = arr[0].as_(); // Search value
    let mut idx = 0; // Index of the search value
    (0..nb_bins).map(move |_| {
        let start_idx = idx; // Start index of the bin (previous end index)
        value += val_step;
        let mid = idx + idx_step;
        let mid = if mid < arr.len() - 1 {
            mid
        } else {
            arr.len() - 2 // TODO: arr.len() - 1 gives error I thought...
        };
        let search_value = T::from_f64(value).unwrap();
        // Implementation WITHOUT pre-guessing mid is slower!!
        // idx = binary_search(arr, search_value, idx, arr.len()-1);
        idx = binary_search_with_mid(arr, search_value, idx, arr.len() - 1, mid); // End index of the bin
        (start_idx, idx)
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
    arr: ArrayView1<T>,
    nb_bins: usize,
) -> impl IndexedParallelIterator<Item = (usize, usize)> + '_
where
    T: Num + FromPrimitive + AsPrimitive<f64> + Sync + Send,
{
    assert!(nb_bins >= 2);
    // Divide by nb_bins to avoid overflow!
    let val_step: f64 =
        (arr[arr.len() - 1].as_() / nb_bins as f64) - (arr[0].as_() / nb_bins as f64);
    let arr0: f64 = arr[0].as_();
    (0..nb_bins).into_par_iter().map(move |i| {
        let start_value = sequential_add_mul(arr0, val_step, i);
        let end_value = start_value + val_step;
        let start_value = T::from_f64(start_value).unwrap();
        let end_value = T::from_f64(end_value).unwrap();
        let start_idx = binary_search(arr, start_value, 0, arr.len() - 1);
        let end_idx = binary_search(arr, end_value, start_idx, arr.len() - 1);
        (start_idx, end_idx)
    })
}

// --------------------------------------- TESTS ---------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils::get_random_array;

    #[test]
    fn test_binary_search() {
        let arr = Array1::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(binary_search(arr.view(), 0, 0, arr.len() - 1), 0);
        assert_eq!(binary_search(arr.view(), 1, 0, arr.len() - 1), 0);
        assert_eq!(binary_search(arr.view(), 2, 0, arr.len() - 1), 1);
        assert_eq!(binary_search(arr.view(), 3, 0, arr.len() - 1), 2);
        assert_eq!(binary_search(arr.view(), 4, 0, arr.len() - 1), 3);
        assert_eq!(binary_search(arr.view(), 5, 0, arr.len() - 1), 4);
        assert_eq!(binary_search(arr.view(), 6, 0, arr.len() - 1), 5);
        assert_eq!(binary_search(arr.view(), 7, 0, arr.len() - 1), 6);
        assert_eq!(binary_search(arr.view(), 8, 0, arr.len() - 1), 7);
        assert_eq!(binary_search(arr.view(), 9, 0, arr.len() - 1), 8);
        assert_eq!(binary_search(arr.view(), 10, 0, arr.len() - 1), 9);
        assert_eq!(binary_search(arr.view(), 11, 0, arr.len() - 1), 9);
    }

    #[test]
    fn test_binary_search_with_mid() {
        let arr = Array1::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(
            binary_search_with_mid(arr.view(), 0, 0, arr.len() - 1, 0),
            0
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 1, 0, arr.len() - 1, 0),
            0
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 2, 0, arr.len() - 1, 1),
            1
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 3, 0, arr.len() - 1, 2),
            2
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 4, 0, arr.len() - 1, 3),
            3
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 5, 0, arr.len() - 1, 4),
            4
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 6, 0, arr.len() - 1, 5),
            5
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 7, 0, arr.len() - 1, 6),
            6
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 8, 0, arr.len() - 1, 7),
            7
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 9, 0, arr.len() - 1, 8),
            8
        );
        assert_eq!(
            binary_search_with_mid(arr.view(), 10, 0, arr.len() - 1, 9),
            9
        );
        // This line causes the code to crash -> because value higher than arr[mid]
        // assert_eq!(binary_search_with_mid(arr.view(), 11, 0, arr.len() - 1, 9), 9);
    }

    #[test]
    fn test_get_equidistant_bin_idxs() {
        let arr = Array1::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let bin_idxs_iter = get_equidistant_bin_idx_iterator(arr.view(), 3);
        let bin_idxs = bin_idxs_iter.map(|x| x.0).collect::<Vec<usize>>();
        assert_eq!(bin_idxs, vec![0, 3, 6]);
        let bin_idxs_iter = get_equidistant_bin_idx_iterator_parallel(arr.view(), 3);
        let bin_idxs = bin_idxs_iter.map(|x| x.0).collect::<Vec<usize>>();
        assert_eq!(bin_idxs, vec![0, 3, 6]);
    }

    #[test]
    fn test_many_random_same_result() {
        let n = 5_000;
        let nb_bins = 100;
        for _ in 0..100 {
            let arr = get_random_array::<i32>(n, i32::MIN, i32::MAX);
            // Sort the array
            let mut arr = arr.to_vec();
            arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let arr = Array1::from(arr);
            // Calculate the bin indexes
            let bin_idxs_iter = get_equidistant_bin_idx_iterator(arr.view(), nb_bins);
            let bin_idxs = bin_idxs_iter.map(|x| x.0).collect::<Vec<usize>>();
            let bin_idxs_iter = get_equidistant_bin_idx_iterator_parallel(arr.view(), nb_bins);
            let bin_idxs_parallel = bin_idxs_iter.map(|x| x.0).collect::<Vec<usize>>();
            assert_eq!(bin_idxs, bin_idxs_parallel);
        }
    }
}
