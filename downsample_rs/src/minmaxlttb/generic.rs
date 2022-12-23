use ndarray::{Array1, ArrayView1};

use super::super::lttb::{lttb, lttb_without_x};
use super::super::types::Num;

#[inline(always)]
pub(crate) fn minmaxlttb_generic<
    Tx: Num,
    Ty: Num + PartialOrd, // TODO: check if partialord is needed
>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    f_minmax: fn(ArrayView1<Ty>, usize) -> Array1<usize>,
) -> Array1<usize> {
    assert_eq!(x.len(), y.len());
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if x.len() / n_out > minmax_ratio {
        let index = f_minmax(y, n_out * minmax_ratio);
        let x = index.mapv(|i| x[i]);
        let y = index.mapv(|i| y[i]);
        let index_points_selected = lttb(x.view(), y.view(), n_out);
        return index_points_selected.mapv(|i| index[i]);
    }
    // Apply lttb on all data when requirement is not met
    lttb(x, y, n_out)
}

#[inline(always)]
pub(crate) fn minmaxlttb_generic_without_x<Ty: Num>(
    y: ArrayView1<Ty>,
    n_out: usize,
    minmax_ratio: usize,
    f_minmax: fn(ArrayView1<Ty>, usize) -> Array1<usize>,
) -> Array1<usize> {
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if y.len() / n_out > minmax_ratio {
        let index = f_minmax(y, n_out * minmax_ratio);
        let y = index.mapv(|i| y[i]);
        let index_points_selected = lttb(index.view(), y.view(), n_out);
        return index_points_selected.mapv(|i| index[i]);
    }
    // Apply lttb on all data when requirement is not met
    lttb_without_x(y.view(), n_out)
}
