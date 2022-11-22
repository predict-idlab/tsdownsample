use ndarray::{Array1, ArrayView1};

use super::super::lttb::utils::Num;
use super::super::lttb::{lttb, lttb_without_x};

const SIZE_THRESHOLD: usize = 10_000_000;
const RATIO_THRESHOLD: usize = 100;
const MINMAX_RATIO: usize = 30;

#[inline]
pub(crate) fn minmaxlttb_generic<
    Tx: Num,
    Ty: Num + PartialOrd, // TODO: check if partialord is needed
>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
    f_minmax: fn(ArrayView1<Ty>, usize) -> Array1<usize>,
) -> Array1<usize> {
    assert_eq!(x.len(), y.len());
    // Apply first min max aggregation (if above threshold & ratio)
    if x.len() > SIZE_THRESHOLD && x.len() / n_out > RATIO_THRESHOLD {
        let index = f_minmax(y, n_out * MINMAX_RATIO);
        let x = index.mapv(|i| x[i]);
        let y = index.mapv(|i| y[i]);
        let index_points_selected = lttb(x.view(), y.view(), n_out);
        return index_points_selected.mapv(|i| index[i]);
    }
    // Apply lttb on all data when requirements are not met
    lttb(x, y, n_out)
}

#[inline]
pub(crate) fn minmaxlttb_generic_without_x<Ty: Num>(
    y: ArrayView1<Ty>,
    n_out: usize,
    f_minmax: fn(ArrayView1<Ty>, usize) -> Array1<usize>,
) -> Array1<usize> {
    // Apply first min max aggregation (if above threshold & ratio)
    if y.len() > SIZE_THRESHOLD && y.len() / n_out > RATIO_THRESHOLD {
        let index = f_minmax(y, n_out * 30);
        let y = index.mapv(|i| y[i]);
        let index_points_selected = lttb(index.view(), y.view(), n_out);
        return index_points_selected.mapv(|i| index[i]);
    }
    // Apply lttb on all data when requirements are not met
    lttb_without_x(y.view(), n_out)
}
