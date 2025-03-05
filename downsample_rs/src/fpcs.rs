use super::minmax;
use super::types::Num;
use argminmax::{ArgMinMax, NaNArgMinMax};
use num_traits::{AsPrimitive, FromPrimitive};
use std::{fmt::Debug, ops::Not};

// ------------------------------- Helper datastructures -------------------------------
// NOTE: the pub(crate) is added to match the visibility of the fpcs_generic function
pub(crate) struct Point<Ty> {
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

// --------------------------------- helper functions ----------------------------------
#[inline(always)]
fn fpcs_inner_core<Ty: PartialOrd + Copy>(
    min_point: &mut Point<Ty>,
    max_point: &mut Point<Ty>,
    potential_point: &mut Point<Ty>,
    previous_min_flag: &mut Flag,
    sampled_indices: &mut Vec<usize>,
) {
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

#[inline(always)]
fn fpcs_inner_comp<Ty: PartialOrd + Copy>(
    y: &[Ty],
    min_idx: usize,
    max_idx: usize,
    min_point: &mut Point<Ty>,
    max_point: &mut Point<Ty>,
) {
    // NOTE: the > and >= stem from the pseudocode details of the FPCS algorithm
    // NOTE: this comparison is inverted as nan comparisons always return false
    if (max_point.y > y[max_idx]).not() {
        max_point.update(max_idx, y[max_idx]);
    }
    if (min_point.y <= y[min_idx]).not() {
        min_point.update(min_idx, y[min_idx]);
    }
}

#[inline(always)]
fn fpcs_outer_loop<Ty: PartialOrd + Copy + Debug>(
    arr: &[Ty],
    minmax_idxs: Vec<usize>,
    n_out: usize,
    inner_comp_fn: fn(&[Ty], usize, usize, &mut Point<Ty>, &mut Point<Ty>),
) -> Vec<usize> {
    let mut previous_min_flag: Flag = Flag::None;
    let mut potential_point: Point<Ty> = Point { x: 0, y: arr[0] };
    let mut max_point: Point<Ty> = Point { x: 0, y: arr[0] };
    let mut min_point: Point<Ty> = Point { x: 0, y: arr[0] };

    let mut sampled_indices: Vec<usize> = Vec::with_capacity(n_out * 2);
    // Prepend first point
    sampled_indices.push(0);

    println!("minmax_idxs: {:?}", minmax_idxs);
    // Process the min/max pairs
    for chunk in minmax_idxs.chunks(2) {
        if chunk.len() == 2 {
            let mut min_idx = chunk[0];
            let mut max_idx = chunk[1];

            // NOTE: the minmax_idxs are not ordered!!
            // So we need to check if the min is actually the min
            if arr[min_idx] > arr[max_idx] {
                min_idx = chunk[1];
                max_idx = chunk[0];
            }

            inner_comp_fn(arr, min_idx, max_idx, &mut min_point, &mut max_point);

            fpcs_inner_core(
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

// -------------------------------------- MACROs ---------------------------------------
// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

macro_rules! fpcs_with_x {
    ($func_name:ident, $trait:ident, $f_minmax:expr, $f_fpcs_inner_comp:expr) => {
        pub fn $func_name<Tx, Ty>(x: &[Tx], y: &[Ty], n_out: usize) -> Vec<usize>
        where
            for<'a> &'a [Ty]: $trait,
            Tx: Num + FromPrimitive + AsPrimitive<f64>,
            Ty: Copy + PartialOrd + Debug,
        {
            fpcs_generic(x, y, n_out, $f_minmax, $f_fpcs_inner_comp)
        }
    };
}

fpcs_with_x!(
    fpcs_with_x,
    ArgMinMax,
    minmax::min_max_with_x,
    fpcs_inner_comp
);
fpcs_with_x!(
    fpcs_with_x_nan,
    NaNArgMinMax,
    minmax::min_max_with_x_nan,
    fpcs_inner_comp
);

// ----------- WITHOUT X

macro_rules! fpcs_without_x {
    ($func_name:ident, $trait:path, $f_minmax:expr, $f_fpcs_inner_comp:expr) => {
        pub fn $func_name<T: Copy + PartialOrd + Debug>(arr: &[T], n_out: usize) -> Vec<usize>
        where
            for<'a> &'a [T]: $trait,
        {
            fpcs_generic_without_x(arr, n_out, $f_minmax, $f_fpcs_inner_comp)
        }
    };
}

fpcs_without_x!(
    fpcs_without_x,
    ArgMinMax,
    minmax::min_max_without_x,
    fpcs_inner_comp
);
fpcs_without_x!(
    fpcs_without_x_nan,
    NaNArgMinMax,
    minmax::min_max_without_x_nan,
    fpcs_inner_comp
);

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

macro_rules! fpcs_with_x_parallel {
    ($func_name:ident, $trait:path, $f_argminmax:expr, $f_fpcs_inner_comp:expr) => {
        pub fn $func_name<Tx, Ty>(x: &[Tx], y: &[Ty], n_out: usize) -> Vec<usize>
        where
            for<'a> &'a [Ty]: $trait,
            Tx: Num + FromPrimitive + AsPrimitive<f64> + Send + Sync,
            Ty: Num + Copy + PartialOrd + Send + Sync + Debug,
        {
            // collect the parrallel iterator to a vector
            fpcs_generic(x, y, n_out, $f_argminmax, $f_fpcs_inner_comp)
        }
    };
}

fpcs_with_x_parallel!(
    fpcs_with_x_parallel,
    ArgMinMax,
    minmax::min_max_with_x_parallel,
    fpcs_inner_comp
);
fpcs_with_x_parallel!(
    fpcs_with_x_parallel_nan,
    NaNArgMinMax,
    minmax::min_max_with_x_parallel_nan,
    fpcs_inner_comp
);

// ----------- WITHOUT X

macro_rules! fpcs_without_x_parallel {
    ($func_name:ident, $trait:path, $f_argminmax:expr, $f_fpcs_inner_comp:expr) => {
        pub fn $func_name<Ty: Copy + PartialOrd + Send + Sync + Debug>(
            arr: &[Ty],
            n_out: usize,
        ) -> Vec<usize>
        where
            for<'a> &'a [Ty]: $trait,
            // Ty: Num + AsPrimitive<f64> + Send + Sync,
        {
            fpcs_generic_without_x(arr, n_out, $f_argminmax, $f_fpcs_inner_comp)
        }
    };
}

fpcs_without_x_parallel!(
    fpcs_without_x_parallel,
    ArgMinMax,
    minmax::min_max_without_x_parallel,
    fpcs_inner_comp
);
fpcs_without_x_parallel!(
    fpcs_without_x_parallel_nan,
    NaNArgMinMax,
    minmax::min_max_without_x_parallel_nan,
    fpcs_inner_comp
);

// ------------------------------------- GENERICS --------------------------------------

// ----------- WITH X

#[inline(always)]
pub(crate) fn fpcs_generic<Tx: Num + AsPrimitive<f64>, Ty: PartialOrd + Copy + Debug>(
    x: &[Tx],
    y: &[Ty],
    n_out: usize,
    f_minmax: fn(&[Tx], &[Ty], usize) -> Vec<usize>,
    fpcs_inner_comp: fn(&[Ty], usize, usize, &mut Point<Ty>, &mut Point<Ty>),
) -> Vec<usize> {
    assert_eq!(x.len(), y.len());
    let mut minmax_idxs = f_minmax(&x[1..(x.len() - 1)], &y[1..(x.len() - 1)], (n_out - 2) * 2);
    minmax_idxs.iter_mut().for_each(|elem| *elem += 1); // inplace + 1
    return fpcs_outer_loop(y, minmax_idxs, n_out, fpcs_inner_comp);
}

// ----------- WITHOUT X

#[inline(always)]
pub(crate) fn fpcs_generic_without_x<Ty: PartialOrd + Copy + Debug>(
    arr: &[Ty],
    n_out: usize,
    f_minmax: fn(&[Ty], usize) -> Vec<usize>,
    fpcs_inner_comp: fn(&[Ty], usize, usize, &mut Point<Ty>, &mut Point<Ty>),
) -> Vec<usize> {
    let mut minmax_idxs: Vec<usize> = f_minmax(&arr[1..(arr.len() - 1)], (n_out - 2) * 2);
    minmax_idxs.iter_mut().for_each(|elem| *elem += 1); // inplace + 1
    return fpcs_outer_loop(arr, minmax_idxs, n_out, fpcs_inner_comp);
}

// -
// ------------------------------------- LEGACY --------------------------------------
// -

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
