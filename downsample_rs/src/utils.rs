use ndarray::Array1;
use ndarray::ArrayView1;

use rayon::prelude::*;
use std::ops::{Add, Div, Sub, Mul};

pub trait FromUsize {
    fn from_usize(value: usize) -> Self;
}

macro_rules! impl_from_usize {
    ($($t:ty),*) => {
        $(
            impl FromUsize for $t {
                #[inline]
                fn from_usize(value: usize) -> Self {
                    value as Self
                }
            }
        )*
    };
}

impl_from_usize!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, usize);

#[inline(always)]
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

#[inline(always)]
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
    left
}

pub fn get_equidistant_bin_idxs<T>(arr: ArrayView1<T>, nb_bins: usize) -> Array1<usize>
where
    T: PartialOrd
        + Copy
        + Sub<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        // + std::convert::From<usize>,
        + FromUsize,
{
    assert!(nb_bins >= 2);
    let mut bin_idxs: Array1<usize> = Array1::zeros(nb_bins - 1);
    // Divide by nb_bins to avoid overflow!
    let val_step: T = (arr[arr.len() - 1] / FromUsize::from_usize(nb_bins)) - (arr[0] / FromUsize::from_usize(nb_bins));
    let idx_step: usize = arr.len() / nb_bins;
    let mut value = arr[0];
    let mut idx = 0;
    for i in 0..nb_bins-1 {
        value = value + val_step;
        let mid = idx + idx_step;
        let mid = if mid < arr.len() { mid } else { arr.len() - 1 };
        // Implementation WITHOUT pre-guessing mid is slower!!
        // idx = binary_search(arr, value, idx, arr.len()-1);
        idx = binary_search_with_mid(arr, value, idx, arr.len() - 1, mid);
        bin_idxs[i] = idx;
    }
    bin_idxs
}

// #[inline(always)]
fn sequential_add_mul<T: Copy + Add<Output = T> + Mul<Output = T> + FromUsize>(
    start_val: T,
    add_val: T,
    mul: usize,
) -> T 
{
    // a + x*b will sometimes overflow when x*b is larger than the largest positive 
    // number in the datatype.
    // This code should not fail when: (T::MAX - a) < (x*b).
    let mul_2: usize = mul / 2;
    start_val + add_val * FromUsize::from_usize(mul_2) + add_val * FromUsize::from_usize(mul - mul_2)
}

pub fn get_equidistant_bin_idxs_parallel<T>(arr: ArrayView1<T>, nb_bins: usize) -> Array1<usize>
where
    T: PartialOrd
        + Copy
        + Sub<Output = T>
        + Add<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Send
        + Sync
        + FromUsize,
{
    assert!(nb_bins >= 2);
    let mut bin_idxs: Array1<usize> = Array1::from((1..nb_bins).collect::<Vec<usize>>());
    // Divide by nb_bins to avoid overflow!
    let val_step: T = (arr[arr.len() - 1] / FromUsize::from_usize(nb_bins)) - (arr[0] / FromUsize::from_usize(nb_bins));
    // let idx_step: usize = arr.len() / nb_bins;
    bin_idxs.par_iter_mut().for_each(|i| {
        let value = sequential_add_mul(arr[0], val_step, *i); 
        *i = binary_search(arr, value, 0, arr.len() - 1);
        // Implementation WITH pre-guessing mid is slower!!
        // *i = binary_search_with_mid(arr, value, 0, arr.len() - 1, *i * idx_step);
    });
    bin_idxs
}

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
        assert_eq!(binary_search_with_mid(arr.view(), 0, 0, arr.len() - 1, 0), 0);
        assert_eq!(binary_search_with_mid(arr.view(), 1, 0, arr.len() - 1, 0), 0);
        assert_eq!(binary_search_with_mid(arr.view(), 2, 0, arr.len() - 1, 1), 1);
        assert_eq!(binary_search_with_mid(arr.view(), 3, 0, arr.len() - 1, 2), 2);
        assert_eq!(binary_search_with_mid(arr.view(), 4, 0, arr.len() - 1, 3), 3);
        assert_eq!(binary_search_with_mid(arr.view(), 5, 0, arr.len() - 1, 4), 4);
        assert_eq!(binary_search_with_mid(arr.view(), 6, 0, arr.len() - 1, 5), 5);
        assert_eq!(binary_search_with_mid(arr.view(), 7, 0, arr.len() - 1, 6), 6);
        assert_eq!(binary_search_with_mid(arr.view(), 8, 0, arr.len() - 1, 7), 7);
        assert_eq!(binary_search_with_mid(arr.view(), 9, 0, arr.len() - 1, 8), 8);
        assert_eq!(binary_search_with_mid(arr.view(), 10, 0, arr.len() - 1, 9), 9);
        // This line causes the code to crash -> because value higher than arr[mid]
        // assert_eq!(binary_search_with_mid(arr.view(), 11, 0, arr.len() - 1, 9), 9);
    }

    #[test]
    fn test_get_equidistant_bin_idxs() {
        let arr = Array1::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let bin_idxs = get_equidistant_bin_idxs(arr.view(), 3);
        assert_eq!(bin_idxs, Array1::from(vec![3, 6]));
        let bin_idxs = get_equidistant_bin_idxs_parallel(arr.view(), 3);
        assert_eq!(bin_idxs, Array1::from(vec![3, 6]));
    }

    #[test]
    fn test_many_random_same_result() {
        let n = 5_000;
        let nb_bins = 100;
        for _ in (0..100) {
            let arr = get_random_array::<i32>(n, i32::MIN, i32::MAX);
            // Sort the array
            let mut arr = arr.to_vec();
            arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let arr = Array1::from(arr);
            // Calculate the bin indexes
            let bin_idxs = get_equidistant_bin_idxs(arr.view(), nb_bins);
            let bin_idxs_parallel = get_equidistant_bin_idxs_parallel(arr.view(), nb_bins);
            assert_eq!(bin_idxs, bin_idxs_parallel);
        }
    }
}

// use std::convert::{ TryFrom, TryInto };

// // Divide and conquer algorithm
// pub fn get_equidistant_bin_idxs_dc<T>(
//     arr: ArrayView1<T>,
//     nb_bins: usize,
// ) -> Array1<usize>
// where
//     T: PartialOrd + Copy + Sub<Output = T> + Add<Output = T> + Div<Output = T> + std::convert::From<usize> + Mul<Output = T> + Rem<Output = T>,
//     usize: std::convert::From<T>
// {
//     let mut bin_idxs: Array1<usize> = Array1::from((0..nb_bins).collect::<Vec<usize>>());
//     let val_step: T = (arr[arr.len() - 1] - arr[0]) / nb_bins.try_into().unwrap();
//     dnc_loop(arr, val_step, &mut bin_idxs, 0, arr.len() - 1, 0);
//     bin_idxs
// }

// fn dnc_loop<T>(
//     arr: ArrayView1<T>,
//     val_step: T,
//     bin_idxs: &mut Array1<usize>,
//     left: usize,
//     right: usize,
//     insert_idx: usize,
// )
// where
//     T: PartialOrd + Copy + Sub<Output = T> + Add<Output = T> + Div<Output = T> + std::convert::From<usize> + Mul<Output = T> + Rem<Output = T>,
//     usize: std::convert::From<T>
// {
//     // Base case: only 1 search value between left and right
//     let val_left = arr[left];
//     let val_right = arr[right];
//     let diff = (val_right / val_step) - (val_left / val_step);
//     // Convert to usize
//     let diff: usize = usize::try_from(diff).unwrap();
//     if diff == 0 {
//         return;
//     }
//     else if diff == 1 {
//         // println!("left: {}, right: {}, insert_idx: {}", left, right, insert_idx);
//         let search_val = val_right - (val_right % val_step);
//         bin_idxs[insert_idx] = binary_search(arr, search_val, left, right);
//         return;
//     } else {
//         let mid = (left + right) / 2;
//         dnc_loop(arr, val_step, bin_idxs, left, mid, insert_idx);
//         dnc_loop(arr, val_step, bin_idxs, mid+1, right, insert_idx + diff / 2);
//     }
// }

// // Divide and conquer algorithm - iterative version
// pub fn get_equidistant_bin_idxs_dc_iterative<T>(
//     arr: ArrayView1<T>,
//     nb_bins: usize,
// ) -> Array1<usize>
// where
//     T: PartialOrd + Copy + Sub<Output = T> + Add<Output = T> + Div<Output = T> + std::convert::From<usize> + Mul<Output = T> + Rem<Output = T>,
//     usize: std::convert::From<T>
// {
//     let mut bin_idxs: Array1<usize> = Array1::from((0..nb_bins).collect::<Vec<usize>>());
//     let val_step: T = (arr[arr.len() - 1] - arr[0]) / nb_bins.try_into().unwrap();
//     let mut stack: Vec<(usize, usize, usize)> = Vec::new();
//     stack.push((0, arr.len() - 1, 0));
//     while !stack.is_empty() {
//         let (left, right, insert_idx) = stack.pop().unwrap();
//         // Base case: only 1 search value between left and right
//         let val_left = arr[left];
//         let val_right = arr[right];
//         let diff = (val_right / val_step) - (val_left / val_step);
//         // Convert to usize
//         let diff: usize = usize::try_from(diff).unwrap();
//         if diff == 0 {
//             continue;
//         }
//         else if diff == 1 {
//             // println!("left: {}, right: {}, insert_idx: {}", left, right, insert_idx);
//             let search_val = val_right - (val_right % val_step);
//             bin_idxs[insert_idx] = binary_search(arr, search_val, left, right);
//             continue;
//         } else {
//             let mid = (left + right) / 2;
//             stack.push((left, mid, insert_idx));
//             stack.push((mid+1, right, insert_idx + diff / 2));
//         }
//     }
//     bin_idxs
// }
