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
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let block_size = arr.len() as f64 / (n_out as f64) * 2.0;
    let block_size = block_size.floor() as usize;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    let mut i: usize = 0; // TODO: for some reason is this faster than enumerate
    arr.slice(s![..block_size * n_out / 2])
        .exact_chunks(block_size)
        .into_iter()
        .for_each(|step| {
            let (min_index, max_index) = f_argminmax(step);
            let offset = block_size * i;

            // Add the indexes in sorted order
            if min_index < max_index {
                sampled_indices[2 * i] = min_index + offset;
                sampled_indices[2 * i + 1] = max_index + offset;
            } else {
                sampled_indices[2 * i] = max_index + offset;
                sampled_indices[2 * i + 1] = min_index + offset;
            }
            i += 1;
        });

    sampled_indices
}

#[inline(always)]
pub(crate) fn min_max_generic_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let block_size = arr.len() as f64 / (n_out as f64) * 2.0;
    let block_size = block_size.floor() as usize;

    // Store the enumerated indexes in the output array
    let mut sampled_indices: Array1<usize> = Array1::from_vec((0..n_out).collect::<Vec<usize>>());

    Zip::from(
        arr.slice(s![..block_size * n_out / 2])
            .exact_chunks(block_size),
    )
    .and(sampled_indices.exact_chunks_mut(2))
    .par_for_each(|step, mut sampled_index| {
        let (min_index, max_index) = f_argminmax(step);

        // Add the indexes in sorted order
        let offset = block_size * unsafe { *sampled_index.uget(0) >> 1 };
        if min_index < max_index {
            sampled_index[0] = min_index + offset;
            sampled_index[1] = max_index + offset;
        } else {
            sampled_index[0] = max_index + offset;
            sampled_index[1] = min_index + offset;
        }
    });

    sampled_indices
}

// --------------------- WITH X

#[inline(always)]
pub(crate) fn min_max_generic_with_x<T: Copy>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl Iterator<Item = (usize, usize)>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let ptr = arr.as_ptr();
    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out);

    let mut i: usize = 0; // TODO: for some reason is this faster than enumerate
    bin_idx_iterator.for_each(|(start, end)| {
        let step = unsafe { ArrayView1::from_shape_ptr(end - start, ptr.add(start)) };
        let (min_index, max_index) = f_argminmax(step);

        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_indices[2 * i] = min_index + start;
            sampled_indices[2 * i + 1] = max_index + start;
        } else {
            sampled_indices[2 * i] = max_index + start;
            sampled_indices[2 * i + 1] = min_index + start;
        }
        i += 1;
    });

    sampled_indices
}

#[inline(always)]
pub(crate) fn min_max_generic_with_x_parallel<T: Copy + Send + Sync>(
    arr: ArrayView1<T>,
    bin_idx_iterator: impl IndexedParallelIterator<Item = (usize, usize)>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Assumes n_out is a multiple of 2
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    // Create a mutex to store the sampled indices
    let sampled_indices = Arc::new(Mutex::new(Array1::<usize>::default(n_out)));

    // Iterate over the bins
    bin_idx_iterator.enumerate().for_each(|(i, (start, end))| {
        let (min_index, max_index) = f_argminmax(arr.slice(s![start..end]));

        // Add the indexes in sorted order
        if min_index < max_index {
            sampled_indices.lock().unwrap()[2 * i] = min_index + start;
            sampled_indices.lock().unwrap()[2 * i + 1] = max_index + start;
        } else {
            sampled_indices.lock().unwrap()[2 * i] = max_index + start;
            sampled_indices.lock().unwrap()[2 * i + 1] = min_index + start;
        }
    });

    // Remove the mutex and return the sampled indices
    Arc::try_unwrap(sampled_indices)
        .unwrap()
        .into_inner()
        .unwrap()
}
