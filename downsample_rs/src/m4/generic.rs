// use ndarray::parallel::prelude::*;
use ndarray::Zip;
use ndarray::{s, Array1, ArrayView1};

#[inline(always)]
pub(crate) fn m4_generic<T: Copy + PartialOrd>(
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Non-parallel implementation
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let block_size = (arr.len() - 1) as f64 / (n_out - 1) as f64 * 4.0;
    let block_size = block_size.floor() as usize;

    let n_out_actual = (arr.len() / block_size) * 4 + 1;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out_actual);

    arr.slice(s![..block_size * (n_out_actual - 1) / 4])
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

    // Always add the last point
    sampled_indices[n_out_actual - 1] = arr.len() - 1;

    sampled_indices
}

pub(crate) fn m4_generic_parallel<T: Copy + PartialOrd + Send + Sync>(
    arr: ArrayView1<T>,
    n_out: usize,
    f_argminmax: fn(ArrayView1<T>) -> (usize, usize),
) -> Array1<usize> {
    // Non-parallel implementation
    if n_out >= arr.len() {
        return Array1::from((0..arr.len()).collect::<Vec<usize>>());
    }

    let block_size = (arr.len() - 1) as f64 / (n_out - 1) as f64 * 4.0;
    let block_size = block_size.floor() as usize;

    let n_out_actual = (arr.len() / block_size) * 4 + 1;

    let mut sampled_indices: Array1<usize> = Array1::<usize>::default(n_out_actual);

    // Create step array
    let idxs = Array1::from((0..n_out_actual / 4).collect::<Vec<usize>>());

    // Iterate over the sample_index pointers and the array chunks
    Zip::from(
        arr.slice(s![..block_size * (n_out_actual - 1) / 4])
            .exact_chunks(block_size),
    )
    .and(
        sampled_indices
            .slice_mut(s![..n_out_actual - 1])
            .exact_chunks_mut(4),
    )
    .and(idxs.view())
    .par_for_each(|step, mut sampled_index, i| {
        let (min_index, max_index) = f_argminmax(step);

        let start_idx = block_size * i;
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

    // Always add the last point
    sampled_indices[n_out_actual - 1] = arr.len() - 1;

    sampled_indices
}

// TODO: just returning an array of T might be more efficient?
