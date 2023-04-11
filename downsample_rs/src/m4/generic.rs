use ndarray::Zip;
use ndarray::{s, Array1, ArrayView1};

use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

// TODO: check for duplicate data in the output array
// -> In the current implementation we always add 4 datapoints per bin (if of
//    course the bin has >= 4 datapoints). However, the argmin and argmax might
//    be the start and end of the bin, which would result in duplicate data in
//    the output array. (this is for example the case for monotonic data).

// --------------------- WITHOUT X

#[inline(always)]
pub(crate) fn m4_generic<T: Copy + PartialOrd>(
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 4) as f64;
    let arr_ptr = arr.as_ptr();

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    let mut start_idx: usize = 0;
    // let mut end: f64 = 0.0;
    for i in 0..n_out / 4 {
        // end += block_size;
        // if end.fract() > 1.0 - 1.0 / (n_out as f64) {
        //     // Is necessary to avoid rounding errors
        //     end = end.ceil();
        // }
        // Decided to use multiplication instead of adding to the accumulator (end)
        // as multiplication seems to be less prone to rounding errors.
        let end: f64 = block_size * (i + 1) as f64;
        let end_idx: usize = end as usize + 1;
        let (min_index, max_index) = f_argminmax(unsafe {
            ArrayView1::from_shape_ptr((end_idx - start_idx,), arr_ptr.add(start_idx))
        });

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
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 4) as f64;

    // Store the enumerated indexes in the output array
    let mut sampled_indices: Array1<usize> = Array1::from_vec((0..n_out).collect::<Vec<usize>>());

    Zip::from(sampled_indices.exact_chunks_mut(4)).par_for_each(|mut sampled_index| {
        let i: usize = unsafe { *sampled_index.uget(0) >> 2 };
        let start_idx = if i == 0 {
            0
        } else {
            (block_size * i as f64) as usize + 1
        };
        let end_idx = (block_size * (i + 1) as f64) as usize + 1;

        let (min_index, max_index) = f_argminmax(unsafe {
            ArrayView1::from_shape_ptr((end_idx - start_idx,), arr.as_ptr().add(start_idx))
        });

        sampled_index[0] = start_idx;
        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_index[1] = min_index + start_idx;
            sampled_index[2] = max_index + start_idx;
        } else {
            sampled_index[1] = max_index + start_idx;
            sampled_index[2] = min_index + start_idx;
        }
        sampled_index[3] = end_idx - 1;
    });

    sampled_indices
}

// --------------------- WITH X

#[inline(always)]
pub(crate) fn m4_generic_with_x<T: Copy>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl Iterator<Item = Option<(usize, usize)>>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let arr_ptr = arr.as_ptr();
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
                let step = unsafe { ArrayView1::from_shape_ptr(end - start, arr_ptr.add(start)) };
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

    Array1::from_vec(sampled_indices)
}

#[inline(always)]
pub(crate) fn m4_generic_with_x_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl IndexedParallelIterator<Item = impl Iterator<Item = Option<(usize, usize)>>>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 4
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    Array1::from_vec(
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
                                let step = unsafe {
                                    ArrayView1::from_shape_ptr(end - start, arr.as_ptr().add(start))
                                };
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
            .collect::<Vec<usize>>(),
    )
}
