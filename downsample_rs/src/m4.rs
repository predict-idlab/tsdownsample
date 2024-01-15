use argminmax::ArgMinMax;
use num_traits::{AsPrimitive, FromPrimitive};
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use super::searchsorted::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::types::Num;
use super::POOL;

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn m4_with_x<Tx, Ty>(x: &[Tx], arr: &[Ty], n_out: usize) -> Vec<usize>
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

pub fn m4_without_x<T: Copy + PartialOrd>(arr: &[T], n_out: usize) -> Vec<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    assert_eq!(n_out % 4, 0);
    m4_generic(arr, n_out, |arr| arr.argminmax())
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn m4_with_x_parallel<Tx, Ty>(x: &[Tx], arr: &[Ty], n_out: usize) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
    Tx: Num + FromPrimitive + AsPrimitive<f64> + Send + Sync,
    Ty: Copy + PartialOrd + Send + Sync,
{
    assert_eq!(n_out % 4, 0);
    let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out / 4);
    m4_generic_with_x_parallel(arr, bin_idx_iterator, n_out, |arr| arr.argminmax())
}

// ----------- WITHOUT X

pub fn m4_without_x_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: &[T],
    n_out: usize,
) -> Vec<usize>
where
    for<'a> &'a [T]: ArgMinMax,
{
    assert_eq!(n_out % 4, 0);
    m4_generic_parallel(arr, n_out, |arr| arr.argminmax())
}

// TODO: check for duplicate data in the output array
// -> In the current implementation we always add 4 datapoints per bin (if of
//    course the bin has >= 4 datapoints). However, the argmin and argmax might
//    be the start and end of the bin, which would result in duplicate data in
//    the output array. (this is for example the case for monotonic data).

// ----------------------------------- GENERICS ------------------------------------

// --------------------- WITHOUT X

#[inline(always)]
pub(crate) fn m4_generic<T: Copy + PartialOrd>(
    arr: &[T],
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return (0..arr.len()).collect();
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 4) as f64;

    let mut sampled_indices: Vec<usize> = vec![usize::default(); n_out];

    let mut start_idx: usize = 0;
    for i in 0..n_out / 4 {
        // Decided to use multiplication instead of adding to the accumulator (end)
        // as multiplication seems to be less prone to rounding errors.
        let end: f64 = block_size * (i + 1) as f64;
        let end_idx: usize = end as usize + 1;

        let (min_index, max_index) = f_argminmax(&arr[start_idx..end_idx]);

        // Add the indexes in sorted order
        sampled_indices[4 * i] = start_idx;
        if min_index < max_index {
            sampled_indices[4 * i + 1] = min_index + start_idx;
            sampled_indices[4 * i + 2] = max_index + start_idx;
        } else {
            sampled_indices[4 * i + 1] = max_index + start_idx;
            sampled_indices[4 * i + 2] = min_index + start_idx;
        }
        sampled_indices[4 * i + 3] = end_idx - 1;

        start_idx = end_idx;
    }

    sampled_indices
}

#[inline(always)]
pub(crate) fn m4_generic_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: &[T],
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return (0..arr.len()).collect::<Vec<usize>>();
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 4) as f64;

    // Store the enumerated indexes in the output array
    // These indexes are used to calculate the start and end indexes of each bin in
    // the multi-threaded execution
    let mut sampled_indices: Vec<usize> = (0..n_out).collect::<Vec<usize>>();

    POOL.install(|| {
        sampled_indices
            .par_chunks_exact_mut(4)
            .for_each(|sampled_index_chunk| {
                let i: f64 = unsafe { *sampled_index_chunk.get_unchecked(0) >> 2 } as f64;
                let start_idx: usize = (block_size * i) as usize + (i != 0.0) as usize;
                let end_idx: usize = (block_size * (i + 1.0)) as usize + 1;

                let (min_index, max_index) = f_argminmax(&arr[start_idx..end_idx]);

                sampled_index_chunk[0] = start_idx;
                // Add the indexes in sorted order
                if min_index < max_index {
                    sampled_index_chunk[1] = min_index + start_idx;
                    sampled_index_chunk[2] = max_index + start_idx;
                } else {
                    sampled_index_chunk[1] = max_index + start_idx;
                    sampled_index_chunk[2] = min_index + start_idx;
                }
                sampled_index_chunk[3] = end_idx - 1;
            })
    });

    sampled_indices
}

// --------------------- WITH X

#[inline(always)]
pub(crate) fn m4_generic_with_x<T: Copy>(
    arr: &[T],
    bin_idx_iterator: impl Iterator<Item = Option<(usize, usize)>>,
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return (0..arr.len()).collect::<Vec<usize>>();
    }

    let mut sampled_indices: Vec<usize> = Vec::with_capacity(n_out);

    bin_idx_iterator.for_each(|bin| {
        if let Some((start, end)) = bin {
            if end <= start + 4 {
                // If the bin has <= 4 elements, just add them all
                for i in start..end {
                    sampled_indices.push(i);
                }
            } else {
                // If the bin has > 4 elements, add the first and last + argmin and argmax
                let step = &arr[start..end];
                let (min_index, max_index) = f_argminmax(step);

                sampled_indices.push(start);

                // Add the indexes in sorted order
                if min_index < max_index {
                    sampled_indices.push(min_index + start);
                    sampled_indices.push(max_index + start);
                } else {
                    sampled_indices.push(max_index + start);
                    sampled_indices.push(min_index + start);
                }

                sampled_indices.push(end - 1);
            }
        }
    });

    sampled_indices
}

#[inline(always)]
pub(crate) fn m4_generic_with_x_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: &[T],
    bin_idx_iterator: impl IndexedParallelIterator<Item = impl Iterator<Item = Option<(usize, usize)>>>,
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return (0..arr.len()).collect::<Vec<usize>>();
    }

    POOL.install(|| {
        bin_idx_iterator
            .flat_map(|bin_idx_iterator| {
                bin_idx_iterator
                    .map(|bin| {
                        match bin {
                            Some((start, end)) => {
                                if end <= start + 4 {
                                    // If the bin has <= 4 elements, just return them all
                                    return (start..end).collect::<Vec<usize>>();
                                }

                                // If the bin has > 4 elements, return the first and last + argmin and argmax
                                let step = &arr[start..end];
                                let (min_index, max_index) = f_argminmax(step);

                                // Return the indexes in sorted order
                                let mut sampled_index = vec![start, 0, 0, end - 1];
                                if min_index < max_index {
                                    sampled_index[1] = min_index + start;
                                    sampled_index[2] = max_index + start;
                                } else {
                                    sampled_index[1] = max_index + start;
                                    sampled_index[2] = min_index + start;
                                }
                                sampled_index
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
    })
}

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use super::{m4_with_x, m4_without_x};
    use super::{m4_with_x_parallel, m4_without_x_parallel};

    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Vec<f32> {
        utils::get_random_array(n, f32::MIN, f32::MAX)
    }

    // Template for n_out
    #[template]
    #[rstest]
    #[case(196)]
    #[case(200)]
    #[case(204)]
    fn n_outs(#[case] n_out: usize) {}

    #[test]
    fn test_m4_scalar_without_x_correct() {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_without_x(&arr, 12);
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
    fn test_m4_scalar_without_x_parallel_correct() {
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_without_x_parallel(&arr, 12);
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
    fn test_m4_scalar_with_x_correct() {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_with_x(&x, &arr, 12);
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
    fn test_m4_scalar_with_x_parallel_correct() {
        let x: [i32; 100] = core::array::from_fn(|i| i.as_());
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_with_x_parallel(&x, &arr, 12);
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
    fn test_m4_scalar_with_x_gap() {
        // We will create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_with_x(&x, &arr, 20);
        assert_eq!(sampled_indices.len(), 16); // One full gap
        let expected_indices = vec![0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|x| if x > 101 { x + 50 } else { x });

        let sampled_indices = m4_with_x(&x, &arr, 20);
        assert_eq!(sampled_indices.len(), 17); // Gap with 1 value
        let expected_indices = vec![
            0, 0, 39, 39, 40, 40, 50, 50, 51, 52, 52, 59, 59, 60, 60, 99, 99,
        ];
        assert_eq!(sampled_indices, expected_indices);
    }

    #[test]
    fn test_m4_scalar_with_x_gap_parallel() {
        // We will create a gap in the middle of the array
        // Increment the second half of the array by 50
        let x: [i32; 100] = core::array::from_fn(|i| if i > 50 { (i + 50).as_() } else { i.as_() });
        let arr: [f32; 100] = core::array::from_fn(|i| i.as_());

        let sampled_indices = m4_with_x_parallel(&x, &arr, 20);
        assert_eq!(sampled_indices.len(), 16); // One full gap
        let expected_indices = vec![0, 0, 29, 29, 30, 30, 50, 50, 51, 51, 69, 69, 70, 70, 99, 99];
        assert_eq!(sampled_indices, expected_indices);

        // Increment the second half of the array by 50 again
        let x = x.map(|x| if x > 101 { x + 50 } else { x });

        let sampled_indices = m4_with_x_parallel(&x, &arr, 20);
        assert_eq!(sampled_indices.len(), 17); // Gap with 1 value
        let expected_indices = vec![
            0, 0, 39, 39, 40, 40, 50, 50, 51, 52, 52, 59, 59, 60, 60, 99, 99,
        ];
        assert_eq!(sampled_indices, expected_indices);
    }

    #[apply(n_outs)]
    fn test_many_random_runs_correct(n_out: usize) {
        const N: usize = 20_003;
        let x: [i32; N] = core::array::from_fn(|i| i.as_());
        for _ in 0..100 {
            let arr = get_array_f32(N);
            let idxs1 = m4_without_x(arr.as_slice(), n_out);
            let idxs2 = m4_with_x(&x, arr.as_slice(), n_out);
            assert_eq!(idxs1, idxs2);
            let idxs3 = m4_without_x_parallel(arr.as_slice(), n_out);
            let idxs4 = m4_with_x_parallel(&x, arr.as_slice(), n_out);
            assert_eq!(idxs1, idxs3);
            assert_eq!(idxs1, idxs4); // TODO: this fails when nb. of threads = 16
        }
    }
}
