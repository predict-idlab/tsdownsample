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

    let block_size = arr.len() as f64 / (n_out as f64) * 4.0;
    let block_size = block_size.floor() as usize;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    arr.slice(s![..block_size * n_out / 4])
        .exact_chunks(block_size)
        .into_iter()
        .enumerate()
        .for_each(|(i, step)| {
            let (min_index, max_index) = f_argminmax(step);

            let start_idx = block_size * i;
            sampled_indices[4 * i] = start_idx;

            // Add the indexes in sorted order
            if min_index < max_index {
                sampled_indices[4 * i + 1] = min_index + start_idx;
                sampled_indices[4 * i + 2] = max_index + start_idx;
            } else {
                sampled_indices[4 * i + 1] = max_index + start_idx;
                sampled_indices[4 * i + 2] = min_index + start_idx;
            }
            sampled_indices[4 * i + 3] = start_idx + block_size - 1;
        });

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

    let block_size = arr.len() as f64 / (n_out as f64) * 4.0;
    let block_size = block_size.floor() as usize;

    // Store the enumerated indexes in the output array
    let mut sampled_indices: Array1<usize> = Array1::from_vec((0..n_out).collect::<Vec<usize>>());

    // Iterate over the sample_index pointers and the array chunks
    Zip::from(
        arr.slice(s![..block_size * n_out / 4])
            .exact_chunks(block_size),
    )
    .and(sampled_indices.exact_chunks_mut(4))
    .par_for_each(|step, mut sampled_index| {
        let (min_index, max_index) = f_argminmax(step);

        let start_idx = block_size * unsafe { *sampled_index.uget(0) >> 2 };
        sampled_index[0] = start_idx;

        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_index[1] = min_index + start_idx;
            sampled_index[2] = max_index + start_idx;
        } else {
            sampled_index[1] = max_index + start_idx;
            sampled_index[2] = min_index + start_idx;
        }
        sampled_index[3] = start_idx + block_size - 1;
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
