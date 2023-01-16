use ndarray::{s, Array1, ArrayView1};

use super::super::helpers::Average;
use super::super::lttb::{lttb_with_x, lttb_without_x};
use super::super::types::Num;
use num_traits::AsPrimitive;

#[inline(always)]
pub(crate) fn minmaxlttb_generic<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    f_minmax: fn(ArrayView1<Tx>, ArrayView1<Ty>, usize) -> Array1<usize>,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: Average,
{
    assert_eq!(x.len(), y.len());
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if x.len() / n_out > minmax_ratio {
        // Get index of min max points
        let mut index: Array1<usize> =
            f_minmax(x.slice(s![1..-1]), y.slice(s![1..-1]), n_out * minmax_ratio);
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
    f_minmax: fn(ArrayView1<Ty>, usize) -> Array1<usize>,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: Average,
{
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if y.len() / n_out > minmax_ratio {
        // Get index of min max points
        let mut index: Array1<usize> = f_minmax(y.slice(s![1..-1]), n_out * minmax_ratio);
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
