use ndarray::{s, Array1, ArrayView1};

use super::super::helpers::Average;
use super::super::lttb::{lttb_with_x, lttb_without_x};
use super::super::types::Num;
use num_traits::AsPrimitive;

// types to make function signatures easier to read
type ThreadCount = usize;
type OutputCount = usize;

pub enum MinMaxFunctionWithX<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>> {
    Serial(fn(ArrayView1<Tx>, ArrayView1<Ty>, OutputCount) -> Array1<usize>),
    Parallel(fn(ArrayView1<Tx>, ArrayView1<Ty>, OutputCount, ThreadCount) -> Array1<usize>),
}

pub enum MinMaxFunctionWithoutX<Ty: Num + AsPrimitive<f64>> {
    Serial(fn(ArrayView1<Ty>, OutputCount) -> Array1<usize>),
    Parallel(fn(ArrayView1<Ty>, OutputCount, ThreadCount) -> Array1<usize>),
}

#[inline(always)]
pub(crate) fn minmaxlttb_generic<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    n_threads: Option<usize>,
    f_minmax: MinMaxFunctionWithX<Tx, Ty>,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: Average,
{
    assert_eq!(x.len(), y.len());
    assert!(minmax_ratio > 1);
    let n_threads = clip_threadcount(n_threads);
    // Apply first min max aggregation (if above ratio)
    if x.len() / n_out > minmax_ratio {
        // Get index of min max points
        let mut index = match f_minmax {
            MinMaxFunctionWithX::Serial(func) => {
                func(x.slice(s![1..-1]), y.slice(s![1..-1]), n_out * minmax_ratio)
            }
            MinMaxFunctionWithX::Parallel(func) => func(
                x.slice(s![1..-1]),
                y.slice(s![1..-1]),
                n_out * minmax_ratio,
                n_threads.unwrap_or(1), // n_threads cannot be None
            ),
        };
        // inplace + 1
        index.mapv_inplace(|i| i + 1);
        let mut index: Vec<usize> = index.into_raw_vec();
        // Prepend first and last point
        index.insert(0, 0);
        index.push(x.len() - 1);
        let index = Array1::from_vec(index);
        // Get x and y values at index
        let x = unsafe { index.mapv(|i| *x.uget(i)) };
        let y = unsafe { index.mapv(|i| *y.uget(i)) };
        // Apply lttb on the reduced data
        let index_points_selected = lttb_with_x(x.view(), y.view(), n_out);
        // Return the original index
        return index_points_selected.mapv(|i| index[i]);
    }
    // Apply lttb on all data when requirement is not met
    lttb_with_x(x, y, n_out)
}

#[inline(always)]
pub(crate) fn minmaxlttb_generic_without_x<Ty: Num + AsPrimitive<f64>>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    n_threads: Option<usize>,
    f_minmax: MinMaxFunctionWithoutX<Ty>,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: Average,
{
    assert!(minmax_ratio > 1);
    let n_threads = clip_threadcount(n_threads);
    // Apply first min max aggregation (if above ratio)
    if y.len() / n_out > minmax_ratio {
        // Get index of min max points
        let mut index = match f_minmax {
            MinMaxFunctionWithoutX::Serial(func) => func(y.slice(s![1..-1]), n_out * minmax_ratio),
            MinMaxFunctionWithoutX::Parallel(func) => func(
                y.slice(s![1..-1]),
                n_out * minmax_ratio,
                n_threads.unwrap_or(1), // n_threads cannot be None
            ),
        };
        // inplace + 1
        index.mapv_inplace(|i| i + 1);
        let mut index: Vec<usize> = index.into_raw_vec();
        // Prepend first and last point
        index.insert(0, 0);
        index.push(y.len() - 1);
        let index = Array1::from_vec(index);
        // Get y values at index
        let y = unsafe { index.mapv(|i| *y.uget(i)) };
        // Apply lttb on the reduced data
        let index_points_selected = lttb_without_x(y.view(), n_out);
        // Return the original index
        return index_points_selected.mapv(|i| index[i]);
    }
    // Apply lttb on all data when requirement is not met
    lttb_without_x(y.view(), n_out)
}
