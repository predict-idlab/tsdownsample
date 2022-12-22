// use ndarray::parallel::prelude::*;
use ndarray::Zip;
use ndarray::{s, Array1, ArrayView1};

use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// --------------------- WITHOUT X

#[inline(always)]
pub(crate) fn min_max_generic<T: Copy>(
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Non-parallel implementation
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let block_size = (arr.len() - 2) as f64 / (n_out - 2) as f64 * 2.0;
    let block_size = block_size.floor() as usize;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);
    // Always add the first point
    sampled_indices[0] = 0;

    arr.slice(s![1..block_size * (n_out - 2) / 2 + 1])
        .exact_chunks(block_size)
        .into_iter()
        .enumerate()
        .for_each(|(i, step)| {
            // TODO over sampled_indexes itereren voor efficiente mut pointers door te geven
            let (min_index, max_index) = f_argminmax(step);
            let offset = (block_size * i) + 1;

            // Add the indexes in sorted order
            if min_index < max_index {
                sampled_indices[2 * i + 1] = min_index + offset;
                sampled_indices[2 * i + 2] = max_index + offset;
            } else {
                sampled_indices[2 * i + 1] = max_index + offset;
                sampled_indices[2 * i + 2] = min_index + offset;
            }
        });

    // Always add the last point
    sampled_indices[n_out - 1] = arr.len() - 1;

    sampled_indices
}

#[inline(always)]
pub(crate) fn min_max_generic_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Parallel implementation
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let block_size = (arr.len() - 2) as f64 / (n_out - 2) as f64 * 2.0;
    let block_size = block_size.floor() as usize;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);
    // Always add the first point
    sampled_indices[0] = 0;

    // Create step array
    let idxs = Array1::from((0..(n_out - 2) / 2).collect::<Vec<usize>>());

    // Iterate over the sample_index pointers and the array chunks
    Zip::from(
        arr.slice(s![1..block_size * (n_out - 2) / 2 + 1])
            .exact_chunks(block_size),
    )
    .and(
        sampled_indices
            .slice_mut(s![1..n_out - 1])
            .exact_chunks_mut(2),
    )
    .and(idxs.view())
    .par_for_each(|step, mut sampled_index, i| {
        let (min_index, max_index) = f_argminmax(step);
        let offset = (block_size * i) + 1;

        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_index[0] = min_index + offset;
            sampled_index[1] = max_index + offset;
        } else {
            sampled_index[0] = max_index + offset;
            sampled_index[1] = min_index + offset
        }
    });

    // Always add the last point
    sampled_indices[n_out - 1] = arr.len() - 1;

    sampled_indices
}

// --------------------- WITH X

pub(crate) fn min_max_generic_with_x<T: Copy>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl Iterator<Item = (usize, usize)>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Non-parallel implementation
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);
    // Always add the first point
    sampled_indices[0] = 0;

    // let mut prev_end:usize = 0;
    let offset: usize = 1;
    bin_idx_iterator.enumerate().for_each(|(i, (start, end))| {
        let start = start + offset;
        let end = end + offset;
        let (min_index, max_index) = f_argminmax(arr.slice(s![start..end]));

        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_indices[2 * i + 1] = min_index + start;
            sampled_indices[2 * i + 2] = max_index + start;
        } else {
            sampled_indices[2 * i + 1] = max_index + start;
            sampled_indices[2 * i + 2] = min_index + start;
        }
        // prev_end = end
    });

    // Always add the last point
    sampled_indices[n_out - 1] = arr.len() - 1;

    sampled_indices
}

pub(crate) fn min_max_generic_with_x_parallel<T: Copy + Send + Sync>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl IndexedParallelIterator<Item = (usize, usize)>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Non-parallel implementation
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    // Create a mutex to store the sampled indices
    let sampled_indices = Arc::new(Mutex::new(Array1::<usize>::default(n_out)));

    // Always add the first point
    sampled_indices.lock().unwrap()[0] = 0;

    // iterate over the bins
    bin_idx_iterator.enumerate().for_each(|(i, (start, end))| {
        let start = start + 1;
        let end = end + 1;
        let (min_index, max_index) = f_argminmax(arr.slice(s![start..end]));

        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_indices.lock().unwrap()[2 * i + 1] = min_index + start;
            sampled_indices.lock().unwrap()[2 * i + 2] = max_index + start;
        } else {
            sampled_indices.lock().unwrap()[2 * i + 1] = max_index + start;
            sampled_indices.lock().unwrap()[2 * i + 2] = min_index + start;
        }
    });

    // Always add the last point
    sampled_indices.lock().unwrap()[n_out - 1] = arr.len() - 1;

    // Remove the mutex and return the sampled indices
    Arc::try_unwrap(sampled_indices)
        .unwrap()
        .into_inner()
        .unwrap()
}
