use ndarray::Zip;
use ndarray::{Array1, ArrayView1};

use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

// --------------------- WITHOUT X

#[inline(always)]
pub(crate) fn min_max_generic<T: Copy>(
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 2) as f64;
    let arr_ptr = arr.as_ptr();

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    let mut start_idx: usize = 0;
    for i in 0..n_out / 2 {
        // Decided to use multiplication instead of adding to the accumulator (end)
        // as multiplication seems to be less prone to rounding errors.
        let end: f64 = block_size * (i + 1) as f64;
        let end_idx: usize = end as usize + 1;

        let (min_index, max_index) = f_argminmax(unsafe {
            ArrayView1::from_shape_ptr((end_idx - start_idx,), arr_ptr.add(start_idx))
        });

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
    arr: ArrayView1<T>,
    n_out: usize,
    n_threads: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    // arr.len() - 1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out / 2) as f64;

    // Store the enumerated indexes in the output array
    let mut sampled_indices: Array1<usize> = Array1::from_vec((0..n_out).collect::<Vec<usize>>());

    // to limit the amounts of threads Rayon uses, an explicit threadpool needs to be created
    // in which the required code is "installed". This limits the amount of used threads.
    // https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html#method.install
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build();

    let zip_func = || {
        Zip::from(sampled_indices.exact_chunks_mut(2)).par_for_each(|mut sampled_index| {
            let i: f64 = unsafe { *sampled_index.uget(0) >> 1 } as f64;
            let start_idx: usize = (block_size * i) as usize + (i != 0.0) as usize;
            let end_idx: usize = (block_size * (i + 1.0)) as usize + 1;

            let (min_index, max_index) = f_argminmax(unsafe {
                ArrayView1::from_shape_ptr((end_idx - start_idx,), arr.as_ptr().add(start_idx))
            });

            // Add the indexes in sorted order
            if min_index < max_index {
                sampled_index[0] = min_index + start_idx;
                sampled_index[1] = max_index + start_idx;
            } else {
                sampled_index[0] = max_index + start_idx;
                sampled_index[1] = min_index + start_idx;
            }
        });
    };

    pool.unwrap().install(zip_func); // allow panic if pool could not be created

    sampled_indices
}

// --------------------- WITH X

#[inline(always)]
pub(crate) fn min_max_generic_with_x<T: Copy>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl Iterator<Item = Option<(usize, usize)>>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let ptr = arr.as_ptr();
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
                let step = unsafe { ArrayView1::from_shape_ptr(end - start, ptr.add(start)) };
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

    Array1::from_vec(sampled_indices)
}

#[inline(always)]
pub(crate) fn min_max_generic_with_x_parallel<T: Copy + Send + Sync>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl IndexedParallelIterator<Item = impl Iterator<Item = Option<(usize, usize)>>>,
    n_out: usize,
    n_threads: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    // to limit the amounts of threads Rayon uses, an explicit threadpool needs to be created
    // in which the required code is "installed". This limits the amount of used threads.
    // https://docs.rs/rayon/latest/rayon/struct.ThreadPool.html#method.install
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build();

    let iter_func = || {
        Array1::from_vec(
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
                                    let step = unsafe {
                                        ArrayView1::from_shape_ptr(
                                            end - start,
                                            arr.as_ptr().add(start),
                                        )
                                    };
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
                .collect::<Vec<usize>>(),
        )
    };

    pool.unwrap().install(iter_func) // allow panic if pool could not be created
}
