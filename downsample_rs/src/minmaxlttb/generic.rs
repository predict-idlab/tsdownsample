use ndarray::{Array1, ArrayView1};

use super::super::lttb::{lttb_with_x, lttb_without_x};
use super::super::types::{Num, ToF64};
use num_traits::cast::AsPrimitive;

#[inline(always)]
pub(crate) fn minmaxlttb_generic<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    f_minmax: fn(ArrayView1<Tx>, ArrayView1<Ty>, usize) -> Array1<usize>,
) -> Array1<usize> {
    assert_eq!(x.len(), y.len());
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if x.len() / n_out > minmax_ratio {
        let mut index = f_minmax(x, y, n_out * minmax_ratio).into_raw_vec();
        // Prepend first and last point if not already in index
        if index[0] != 0 {
            index.insert(0, 0);
        }
        if index[index.len() - 1] != x.len() - 1 {
            index.push(x.len() - 1);
        }
        let x = Array1::from_shape_vec(
            (index.len(),),
            index.iter().map(|i| x[*i]).collect::<Vec<_>>(),
        )
        .unwrap();
        let y = Array1::from_shape_vec(
            (index.len(),),
            index.iter().map(|i| y[*i]).collect::<Vec<_>>(),
        )
        .unwrap();
        // let x = index.mapv(|i| x[i]);
        // let y = index.mapv(|i| y[i]);
        let index_points_selected = lttb_with_x(x.view(), y.view(), n_out);
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
) -> Array1<usize> {
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if y.len() / n_out > minmax_ratio {
        let mut index = f_minmax(y, n_out * minmax_ratio).into_raw_vec();
        // Prepend first and last point if not already in index
        if index[0] != 0 {
            index.insert(0, 0);
        }
        if index[index.len() - 1] != y.len() - 1 {
            index.push(y.len() - 1);
        }
        let index = Array1::from_shape_vec((index.len(),), index).unwrap();
        let y = index.mapv(|i| y[i]);
        let index_points_selected = lttb_with_x(index.view(), y.view(), n_out);
        return index_points_selected.mapv(|i| index[i]);
    }
    // Apply lttb on all data when requirement is not met
    lttb_without_x(y.view(), n_out)
}
