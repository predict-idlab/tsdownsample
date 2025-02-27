// use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

use argminmax::{ArgMinMax, NaNArgMinMax};
use num_traits::{AsPrimitive, FromPrimitive};

// use crate::minmax;

use super::searchsorted::{
    get_equidistant_bin_idx_iterator, get_equidistant_bin_idx_iterator_parallel,
};
use super::types::Num;
use super::POOL;

// ----------------------------------- Helper datastructures ------------------------------------
struct Point<Ty> {
    x: usize,
    y: Ty,
}

impl<Ty> Point<Ty> {
    fn update(&mut self, x: usize, y: Ty) {
        self.x = x;
        self.y = y;
    }
}

#[derive(PartialEq, Eq)]
enum Flag {
    Min,
    Max,
    None,
}

// ----------------------------------- NON-PARALLEL ------------------------------------
// ----------- WITH X
macro_rules! fpcs_with_x {
    ($func_name:ident, $trait:ident, $f_fpcs:expr) => {
        pub fn $func_name<Tx, Ty>(x: &[Tx], y: &[Ty], n_out: usize) -> Vec<usize>
        where
            for<'a> &'a [Ty]: $trait,
            Tx: Num + AsPrimitive<f64> + FromPrimitive,
            Ty: Num + AsPrimitive<f64>,
        {
            let bin_idx_iterator = get_equidistant_bin_idx_iterator(x, n_out - 2);
            fpcs_generic(y, bin_idx_iterator, n_out, $f_fpcs)
        }
    };
}

fpcs_with_x!(fpcs_with_x, ArgMinMax, |arr| arr.argminmax());
fpcs_with_x!(fpcs_with_x_nan, NaNArgMinMax, |arr| arr.nanargminmax());

// ----------- WITHOUT X
macro_rules! fpcs_without_x {
    ($func_name:ident, $trait:path, $f_argminmax:expr) => {
        pub fn $func_name<T: Copy + PartialOrd>(arr: &[T], n_out: usize) -> Vec<usize>
        where
            for<'a> &'a [T]: $trait,
        {
            fpcs_generic_without_x(arr, n_out, $f_argminmax)
        }
    };
}

fpcs_without_x!(fpcs_without_x, ArgMinMax, |arr| arr.argminmax());
fpcs_without_x!(fpcs_without_x_nan, NaNArgMinMax, |arr| arr.nanargminmax());

// ------------------------------------- PARALLEL --------------------------------------
// ----------- WITH X
macro_rules! fpcs_with_x_parallel {
    ($func_name:ident, $trait:path, $f_argminmax:expr) => {
        pub fn $func_name<Tx, Ty>(x: &[Tx], y: &[Ty], n_out: usize) -> Vec<usize>
        where
            for<'a> &'a [Ty]: $trait,
            Tx: Num + AsPrimitive<f64> + FromPrimitive + Send + Sync,
            Ty: Num + AsPrimitive<f64> + Send + Sync,
        {
            // collect the parrallel iterator to a vector
            let bin_idx_iterator = get_equidistant_bin_idx_iterator_parallel(x, n_out - 2);
            fpcs_generic_parallel(y, bin_idx_iterator, n_out, $f_argminmax)
        }
    };
}

fpcs_with_x_parallel!(fpcs_with_x_parallel, ArgMinMax, |arr| arr.argminmax());
fpcs_with_x_parallel!(fpcs_with_x_parallel_nan, NaNArgMinMax, |arr| arr
    .nanargminmax());

// ----------- WITHOUT X
macro_rules! fpcs_without_x_parallel {
    ($func_name:ident, $trait:path, $f_argminmax:expr) => {
        pub fn $func_name<Ty: Copy + PartialOrd + Send + Sync>(
            arr: &[Ty],
            n_out: usize,
        ) -> Vec<usize>
        where
            for<'a> &'a [Ty]: $trait,
            // Ty: Num + AsPrimitive<f64> + Send + Sync,
        {
            fpcs_generic_without_x_parallel(arr, n_out, $f_argminmax)
        }
    };
}

fpcs_without_x_parallel!(fpcs_without_x_parallel, ArgMinMax, |arr| arr.argminmax());
fpcs_without_x_parallel!(fpcs_without_x_parallel_nan, NaNArgMinMax, |arr| arr
    .nanargminmax());

// ----------- WITHOUT X

// ------------------------------------- GENERICS --------------------------------------
// move the inner loop to a separate function to reduce code duplication
fn fpcs_inner_loop<Ty: PartialOrd + Copy>(
    y: &[Ty],
    min_idx: usize,
    max_idx: usize,
    min_point: &mut Point<Ty>,
    max_point: &mut Point<Ty>,
    potential_point: &mut Point<Ty>,
    previous_min_flag: &mut Flag,
    sampled_indices: &mut Vec<usize>,
) {
    // only update when the min/max is more extreme than the current min/max
    if y[max_idx] > max_point.y {
        max_point.update(max_idx, y[max_idx]);
    }
    if y[min_idx] < min_point.y {
        min_point.update(min_idx, y[min_idx]);
    }

    // if the min is to the left of the max
    if min_point.x < max_point.x {
        // if the min was selected in the previos bin
        if *previous_min_flag == Flag::Min && min_point.x != potential_point.x {
            sampled_indices.push(potential_point.x);
        }
        sampled_indices.push(min_point.x);

        potential_point.update(max_point.x, max_point.y);
        min_point.update(max_point.x, max_point.y);
        *previous_min_flag = Flag::Min;
    } else {
        if *previous_min_flag == Flag::Max && max_point.x != potential_point.x {
            sampled_indices.push(potential_point.x as usize);
        }
        sampled_indices.push(max_point.x as usize);

        potential_point.update(min_point.x, min_point.y);
        max_point.update(min_point.x, min_point.y);
        *previous_min_flag = Flag::Max;
    }
}

// ----------- WITH X
#[inline(always)]
pub(crate) fn fpcs_generic<T: PartialOrd + Copy>(
    y: &[T],
    bin_idx_iterator: impl Iterator<Item = Option<(usize, usize)>>,
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    let mut sampled_indices: Vec<usize> = Vec::with_capacity(n_out * 2);

    let mut previous_min_flag: Flag = Flag::None;
    let mut potential_point: Point<T> = Point { x: 0, y: y[0] };
    let mut max_point: Point<T> = Point { x: 0, y: y[0] };
    let mut min_point: Point<T> = Point { x: 0, y: y[0] };

    bin_idx_iterator.for_each(|bin| {
        if let Some((start, end)) = bin {
            if end <= start + 2 {
                // if the bin has <= 2 elements, we just add them all
                for i in start..end {
                    sampled_indices.push(i);
                }
            } else {
                // if the bin has at least two elements, we find the min and max
                let (min_idx, max_idx) = f_argminmax(&y[start..end]);

                fpcs_inner_loop(
                    y,
                    min_idx,
                    max_idx,
                    &mut min_point,
                    &mut max_point,
                    &mut potential_point,
                    &mut previous_min_flag,
                    &mut sampled_indices,
                );
            }
        }
    });
    // add the last sample
    sampled_indices.push(y.len() as usize - 1);
    sampled_indices
}

#[inline(always)]
pub(crate) fn fpcs_generic_parallel<T: PartialOrd + Copy + Send + Sync>(
    arr: &[T],
    bin_idx_iterator: impl IndexedParallelIterator<Item = impl Iterator<Item = Option<(usize, usize)>>>,
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // Pre-allocate the vector with the right capacity - this will store min-max
    // from each bin to be used in the FPCS algorithm
    let mut minmax_idxs: Vec<usize> = Vec::with_capacity((n_out - 2) * 2);

    POOL.install(|| {
        // Process bins in parallel to find min-max index pairs from each bin
        // Each result contains (bin_position, [min_idx, max_idx]) where:
        // - bin_position ensures results maintain correct order when collected
        // - min_idx is the index of minimum value in the bin
        // - max_idx is the index of maximum value in the bin
        let results: Vec<(usize, [usize; 2])> = bin_idx_iterator
            .enumerate()
            .flat_map(|(bin_index, bin_idx_iterator)| {
                bin_idx_iterator
                    .filter_map(|bin| {
                        bin.and_then(|(start, end)| {
                            match end - start {
                                0 => None, // Empty bin
                                1 => {
                                    // Single element - add it to the result
                                    Some((bin_index * 2, [start, start]))
                                }
                                2 => {
                                    // Two elements - determine min and max
                                    let (min_idx, max_idx) = if arr[start] <= arr[start + 1] {
                                        (start, start + 1)
                                    } else {
                                        (start + 1, start)
                                    };
                                    Some((bin_index * 2, [min_idx, max_idx]))
                                }
                                _ => {
                                    // Larger bins - find min and max
                                    let (min_idx, max_idx) = f_argminmax(&arr[start..end]);
                                    Some((bin_index * 2, [min_idx + start, max_idx + start]))
                                }
                            }
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // Only process results if we have any valid bins
        if !results.is_empty() {
            // Ensure we have enough capacity to avoid reallocations
            minmax_idxs.reserve(results.len() * 2);

            // Populate minmax_idxs with alternating min-max indices
            // These will be processed in pairs by fpcs_inner_loop
            for (_, [min_idx, max_idx]) in results {
                minmax_idxs.push(min_idx);
                minmax_idxs.push(max_idx);
            }
        }
    });

    let mut sampled_indices: Vec<usize> = Vec::with_capacity(n_out * 2);

    let mut previous_min_flag: Flag = Flag::None;
    let mut potential_point: Point<T> = Point { x: 0, y: arr[0] };
    let mut max_point: Point<T> = Point { x: 0, y: arr[0] };
    let mut min_point: Point<T> = Point { x: 0, y: arr[0] };

    // Prepend first and last point
    sampled_indices.push(0);

    // Process the min/max pairs
    for chunk in minmax_idxs.chunks(2) {
        if chunk.len() == 2 {
            let min_idx = chunk[0];
            let max_idx = chunk[1];

            fpcs_inner_loop(
                arr,
                min_idx,
                max_idx,
                &mut min_point,
                &mut max_point,
                &mut potential_point,
                &mut previous_min_flag,
                &mut sampled_indices,
            );
        } else if chunk.len() == 1 {
            // Handle any remaining single element
            sampled_indices.push(chunk[0]);
        }
    }

    // add the last sample
    sampled_indices.push(arr.len() as usize - 1);
    sampled_indices
}

// ----------- WITHOUT X
#[inline(always)]
pub(crate) fn fpcs_generic_without_x<Ty: PartialOrd + Copy>(
    y: &[Ty],
    n_out: usize, // handle n_out behavior on a higher level
    f_argminmax: fn(&[Ty]) -> (usize, usize),
) -> Vec<usize> {
    if n_out >= y.len() {
        return (0..y.len()).collect::<Vec<usize>>();
    }
    assert!(n_out >= 3); // avoid division by 0
                         // let f_argminmax = y.argminmax();

    let block_size: f64 = y.len() as f64 / (n_out - 2) as f64;

    let mut sampled_indices: Vec<usize> = Vec::with_capacity(n_out * 2);

    let mut previous_min_flag: Flag = Flag::None;
    let mut potential_point: Point<Ty> = Point { x: 0, y: y[0] };
    let mut max_point: Point<Ty> = Point { x: 0, y: y[0] };
    let mut min_point: Point<Ty> = Point { x: 0, y: y[0] };

    let mut lower: usize = 1;
    let mut upper: usize = (block_size as usize) + 1;
    while upper < y.len() {
        let (mut min_idx, mut max_idx) = f_argminmax(&y[lower..upper]);
        min_idx += lower;
        max_idx += lower;

        fpcs_inner_loop(
            y,
            min_idx,
            max_idx,
            &mut min_point,
            &mut max_point,
            &mut potential_point,
            &mut previous_min_flag,
            &mut sampled_indices,
        );

        lower = upper;
        upper = (upper as f64 + block_size) as usize;
    }

    // add the last sample
    sampled_indices.push(y.len() as usize - 1);
    sampled_indices
}

#[inline(always)]
pub(crate) fn fpcs_generic_without_x_parallel<T: PartialOrd + Copy + Send + Sync>(
    arr: &[T],
    n_out: usize,
    f_argminmax: fn(&[T]) -> (usize, usize),
) -> Vec<usize> {
    // --------- 1. parallelize the min-max search
    // arr.len() -1 is used to match the delta of a range-index (0..arr.len()-1)
    let block_size: f64 = (arr.len() - 1) as f64 / (n_out - 2) as f64;

    // let mut minmax_idxs: Vec<usize> = Vec::with_capacity((n_out - 2) * 2);
    let mut minmax_idxs: Vec<usize> = vec![0; (n_out - 2) * 2];

    POOL.install(|| {
        minmax_idxs
            .par_chunks_exact_mut(2)
            .for_each(|min_max_index_chunk| {
                let i: f64 = unsafe { *min_max_index_chunk.get_unchecked(0) >> 1 } as f64;
                let start_idx: usize = (block_size * i) as usize + (i != 0.0) as usize;
                let end_idx: usize = (block_size * (i + 1.0)) as usize + 1;

                let (min_index, max_index) = f_argminmax(&arr[start_idx..end_idx]);

                // Add the indexes in min-max order
                min_max_index_chunk[0] = min_index + start_idx;
                min_max_index_chunk[1] = max_index + start_idx;
            })
    });

    let mut sampled_indices: Vec<usize> = Vec::with_capacity(n_out * 2);

    let mut previous_min_flag: Flag = Flag::None;
    let mut potential_point: Point<T> = Point { x: 0, y: arr[0] };
    let mut max_point: Point<T> = Point { x: 0, y: arr[0] };
    let mut min_point: Point<T> = Point { x: 0, y: arr[0] };

    // Prepend first and last point
    sampled_indices.push(0);

    // Process the min/max pairs
    for chunk in minmax_idxs.chunks(2) {
        if chunk.len() == 2 {
            let min_idx = chunk[0];
            let max_idx = chunk[1];

            fpcs_inner_loop(
                arr,
                min_idx,
                max_idx,
                &mut min_point,
                &mut max_point,
                &mut potential_point,
                &mut previous_min_flag,
                &mut sampled_indices,
            );
        } else if chunk.len() == 1 {
            // Handle any remaining single element
            sampled_indices.push(chunk[0]);
        }
    }

    // add the last sample
    sampled_indices.push(arr.len() as usize - 1);
    sampled_indices
}

pub fn vanilla_fpcs_without_x<Ty: Num + std::cmp::PartialOrd + Copy>(
    y: &[Ty],
    n_out: usize, // NOTE: there will be between [n_out, 2*n_out] output points
) -> Vec<usize> {
    if n_out >= y.len() {
        return (0..y.len()).collect::<Vec<usize>>();
    }
    assert!(n_out >= 3); // avoid division by 0

    // TODO -> we don't know the size of the output vector
    // TODO -> make this more concise in a future version
    let mut retain_idxs: Vec<usize> = Vec::with_capacity(n_out * 2);

    let mut meta_counter: u32 = 1;
    // TODO -> check
    let threshold: f64 = y.len() as f64 / (n_out - 2) as f64;
    let mut previous_min_flag: Flag = Flag::None;
    let mut potential_point: Point<Ty> = Point { x: 0, y: y[0] };
    let mut max_point: Point<Ty> = Point { x: 0, y: y[0] };
    let mut min_point: Point<Ty> = Point { x: 0, y: y[0] };

    for idx in 1..y.len() {
        let val = y[idx];

        if val > max_point.y {
            max_point.update(idx, val);
        }
        if val < min_point.y {
            min_point.update(idx, val);
        }

        if (idx > ((meta_counter as f64) * threshold) as usize) || (idx == y.len() - 1) {
            // edge case -> previous flag is None
            meta_counter += 1;

            // if the min is to the left of the max
            if min_point.x < max_point.x {
                // if the min was selected in the previos bin
                if previous_min_flag == Flag::Min && min_point.x != potential_point.x {
                    retain_idxs.push(potential_point.x as usize);
                }
                retain_idxs.push(min_point.x as usize);

                potential_point.update(max_point.x, max_point.y);
                min_point.update(max_point.x, max_point.y);
                previous_min_flag = Flag::Min;
            } else {
                if previous_min_flag == Flag::Max && max_point.x != potential_point.x {
                    retain_idxs.push(potential_point.x as usize);
                }
                retain_idxs.push(max_point.x as usize);

                potential_point.update(min_point.x, min_point.y);
                max_point.update(min_point.x, min_point.y);
                previous_min_flag = Flag::Max;
            }
        }
    }

    // add the last sample
    retain_idxs.push(y.len() as usize - 1);
    return retain_idxs;
}

// --------------------------------------- TESTS ---------------------------------------
#[cfg(test)]
mod tests {
    use dev_utils::utils;

    use super::fpcs_without_x;

    #[test]
    fn test_fpcs_with_x() {
        let x = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let n_out = 3;
        // let sampled_indices = fpcs_without_x(&x, &y, n_out);
        // TODO -> fix after we have a library
        // assert eq(sampled_indices, vec![0, 9])
        // utils::print_vec(&res);
    }
}
