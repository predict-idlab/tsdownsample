use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use argminmax::ArgMinMax;
use num_traits::{AsPrimitive, FromPrimitive};

use super::searchsorted::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::types::Num;

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn min_max_with_x<Tx, Ty>(x: &[Tx], arr: &[Ty], n_out: usize) -> Vec<usize>
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

pub fn min_max_without_x<T: Copy + PartialOrd>(arr: &[T], n_out: usize) -> Vec<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    assert_eq!(n_out % 2, 0);
    min_max_generic(arr, n_out, |arr| arr.argminmax())
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn min_max_with_x_parallel<Tx, Ty>(
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

pub fn min_max_without_x_parallel<T: Copy + PartialOrd + Send + Sync>(
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

// ----------------------------------- GENERICS ------------------------------------

// --------------------- WITHOUT X

#[inline(always)]
pub(crate) fn min_max_generic<T: Copy>(
    arr: &[T],
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return (0..arr.len()).collect::<Vec<usize>>();
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 2) as f64;

    let mut sampled_indices = vec![usize::default(); n_out];

    let mut start_idx: usize = 0;
    for i in 0..n_out / 2 {
        // Decided to use multiplication instead of adding to the accumulator (end)
        // as multiplication seems to be less prone to rounding errors.
        let end: f64 = block_size * (i + 1) as f64;
        let end_idx: usize = end as usize + 1;

        let (min_index, max_index) = f_argminmax(&arr[start_idx..end_idx]);

        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_indices[2 * i] = min_index + start_idx;
            sampled_indices[2 * i + 1] = max_index + start_idx;
        } else {
            sampled_indices[2 * i] = max_index + start_idx;
            sampled_indices[2 * i + 1] = min_index + start_idx;
        }

        start_idx = end_idx;
    }

    sampled_indices
}

#[inline(always)]
pub(crate) fn min_max_generic_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: &[T],
    n_out: usize,
    n_threads: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return (0..arr.len()).collect::<Vec<usize>>();
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 2) as f64;

    // Store the enumerated indexes in the output array
    // let mut sampled_indices: Array1<usize> = Array1::from_vec((0..n_out).collect::<Vec<usize>>());
    let mut sampled_indices: Vec<usize> = (0..n_out).collect::<Vec<usize>>();

    // to limit the amounts of threads Rayon uses, an explicit threadpool needs to be created
    // in which the required code is "installed". This limits the amount of used threads.
    // https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html#method.install
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build();

    let func = || {
        for chunk in sampled_indices.chunks_exact_mut(2) {
            let i: f64 = unsafe { *chunk.get_unchecked(0) >> 1 } as f64;
            let start_idx: usize = (block_size * i) as usize + (i != 0.0) as usize;
            let end_idx: usize = (block_size * (i + 1.0)) as usize + 1;

            let (min_index, max_index) = f_argminmax(&arr[start_idx..end_idx]);

            // Add the indexes in sorted order
            if min_index < max_index {
                chunk[0] = min_index + start_idx;
                chunk[1] = max_index + start_idx;
            } else {
                chunk[0] = max_index + start_idx;
                chunk[1] = min_index + start_idx;
            }
        }
    };

    pool.unwrap().install(func); // allow panic if pool could not be created

    sampled_indices.to_vec()
}

// --------------------- WITH X

#[inline(always)]
pub(crate) fn min_max_generic_with_x<T: Copy>(
    arr: &[T],
    bin_idx_iterator: impl Iterator<Item = Option<(usize, usize)>>,
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return (0..arr.len()).collect::<Vec<usize>>();
    }

    let mut sampled_indices: Vec<usize> = Vec::with_capacity(n_out);

    bin_idx_iterator.for_each(|bin| {
        if let Some((start, end)) = bin {
            if end <= start + 2 {
                // If the bin has <= 2 elements, just add them all
                for i in start..end {
                    sampled_indices.push(i);
                }
            } else {
                // If the bin has at least two elements, add the argmin and argmax
                let step = &arr[start..end];
                let (min_index, max_index) = f_argminmax(step);

                // Add the indexes in sorted order
                if min_index < max_index {
                    sampled_indices.push(min_index + start);
                    sampled_indices.push(max_index + start);
                } else {
                    sampled_indices.push(max_index + start);
                    sampled_indices.push(min_index + start);
                }
            }
        }
    });

    sampled_indices
}

#[inline(always)]
pub(crate) fn min_max_generic_with_x_parallel<T: Copy + Send + Sync>(
    arr: &[T],
    bin_idx_iterator: impl IndexedParallelIterator<Item = impl Iterator<Item = Option<(usize, usize)>>>,
    n_out: usize,
    n_threads: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return (0..arr.len()).collect::<Vec<usize>>();
    }

    // to limit the amounts of threads Rayon uses, an explicit threadpool needs to be created
    // in which the required code is "installed". This limits the amount of used threads.
    // https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html#method.install
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build();

    let iter_func = || {
        bin_idx_iterator
            .flat_map(|bin_idx_iterator| {
                bin_idx_iterator
                    .map(|bin| {
                        match bin {
                            Some((start, end)) => {
                                if end <= start + 2 {
                                    // If the bin has <= 2 elements, just return them all
                                    return (start..end).collect::<Vec<usize>>();
                                }

                                // If the bin has at least two elements, return the argmin and argmax
                                let step = &arr[start..end];
                                let (min_index, max_index) = f_argminmax(step);

                                // Return the indexes in sorted order
                                if min_index < max_index {
                                    vec![min_index + start, max_index + start]
                                } else {
                                    vec![max_index + start, min_index + start]
                                }
                            } // If the bin is empty, return empty Vec
                            None => {
                                vec![]
                            }
                        }
                    })
                    .collect::<Vec<Vec<usize>>>()
            })
            .flatten()
            .collect::<Vec<usize>>()
    };

    pool.unwrap().install(iter_func) // allow panic if pool could not be created
}

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use super::{min_max_with_x, min_max_without_x};
    use super::{min_max_with_x_parallel, min_max_without_x_parallel};

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
    fn test_min_max_scalar_without_x_correct() {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_without_x(&arr, 10);
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
    fn test_min_max_scalar_without_x_parallel_correct(n_threads: usize) {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_without_x_parallel(&arr, 10, n_threads);
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
    fn test_min_max_scalar_with_x_correct() {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_with_x(&x, &arr, 10);
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
    fn test_min_max_scalar_with_x_parallel_correct(n_threads: usize) {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_with_x_parallel(&x, &arr, 10, n_threads);
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
    fn test_min_max_scalar_with_x_gap() {
        // We will create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_with_x(&x, &arr, 10);
        assert_eq!(sampled_indices.len(), 8); // One full gap
        let expected_indices = vec![0, 29, 30, 50, 51, 69, 70, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|i| if i > 101 { i + 50 } else { i });

        let sampled_indices = min_max_with_x(&x, &arr, 10);
        assert_eq!(sampled_indices.len(), 9); // Gap with 1 value
        let expected_indices = vec![0, 39, 40, 50, 51, 52, 59, 60, 99];
        assert_eq!(sampled_indices, expected_indices);
    }

    #[apply(threads)]
    fn test_min_max_scalar_with_x_parallel_gap(n_threads: usize) {
        // Create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = min_max_with_x_parallel(&x, &arr, 10, n_threads);
        assert_eq!(sampled_indices.len(), 8); // One full gap
        let expected_indices = vec![0, 29, 30, 50, 51, 69, 70, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|i| if i > 101 { i + 50 } else { i });

        let sampled_indices = min_max_with_x_parallel(&x, &arr, 10, n_threads);
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
            let idxs1 = min_max_without_x(arr.as_slice(), N_OUT);
            let idxs2 = min_max_without_x_parallel(arr.as_slice(), N_OUT, n_threads);
            let idxs3 = min_max_with_x(&x, arr.as_slice(), N_OUT);
            let idxs4 = min_max_with_x_parallel(&x, arr.as_slice(), N_OUT, n_threads);
            assert_eq!(idxs1, idxs2);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4);
        }
    }
}
