use argminmax::ArgMinMax;

use super::super::lttb::{lttb_with_x, lttb_without_x};
use super::super::types::Num;
use num_traits::AsPrimitive;

// types to make function signatures easier to read
type ThreadCount = usize;
type OutputCount = usize;

pub enum MinMaxFunctionWithX<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>> {
    Serial(fn(&[Tx], &[Ty], OutputCount) -> Vec<usize>),
    Parallel(fn(&[Tx], &[Ty], OutputCount, ThreadCount) -> Vec<usize>),
}

pub enum MinMaxFunctionWithoutX<Ty: Num + AsPrimitive<f64>> {
    Serial(fn(&[Ty], OutputCount) -> Vec<usize>),
    Parallel(fn(&[Ty], OutputCount, ThreadCount) -> Vec<usize>),
}

#[inline(always)]
pub(crate) fn minmaxlttb_generic<Tx: Num + AsPrimitive<f64>, Ty: Num + AsPrimitive<f64>>(
    x: &[Tx],
    y: &[Ty],
    n_out: usize,
    minmax_ratio: usize,
    n_threads: Option<usize>,
    f_minmax: MinMaxFunctionWithX<Tx, Ty>,
) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
{
    assert_eq!(x.len(), y.len());
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if x.len() / n_out > minmax_ratio {
        // Get index of min max points
        let mut index = match f_minmax {
            MinMaxFunctionWithX::Serial(func) => func(
                &x[1..(x.len() - 1)],
                &y[1..(x.len() - 1)],
                n_out * minmax_ratio,
            ),
            MinMaxFunctionWithX::Parallel(func) => func(
                &x[1..(x.len() - 1)],
                &y[1..(x.len() - 1)],
                n_out * minmax_ratio,
                n_threads.unwrap(), // n_threads cannot be None
            ),
        };
        // inplace + 1
        index.iter_mut().for_each(|elem| *elem += 1);
        // Prepend first and last point
        index.insert(0, 0);
        index.push(x.len() - 1);
        // Get x and y values at index
        let x = unsafe {
            index
                .iter()
                .map(|i| *x.get_unchecked(*i))
                .collect::<Vec<Tx>>()
        };
        let y = unsafe {
            index
                .iter()
                .map(|i| *y.get_unchecked(*i))
                .collect::<Vec<Ty>>()
        };
        // Apply lttb on the reduced data
        let index_points_selected = lttb_with_x(x.as_slice(), y.as_slice(), n_out);
        // Return the original index
        return index_points_selected
            .iter()
            .map(|i| index[*i])
            .collect::<Vec<usize>>();
    }
    // Apply lttb on all data when requirement is not met
    lttb_with_x(x, y, n_out)
}

#[inline(always)]
pub(crate) fn minmaxlttb_generic_without_x<Ty: Num + AsPrimitive<f64>>(
    y: &[Ty],
    n_out: usize,
    minmax_ratio: usize,
    n_threads: Option<usize>,
    f_minmax: MinMaxFunctionWithoutX<Ty>,
) -> Vec<usize>
where
    for<'a> &'a [Ty]: ArgMinMax,
{
    assert!(minmax_ratio > 1);
    // Apply first min max aggregation (if above ratio)
    if y.len() / n_out > minmax_ratio {
        // Get index of min max points
        let mut index = match f_minmax {
            MinMaxFunctionWithoutX::Serial(func) => {
                func(&y[1..(y.len() - 1)], n_out * minmax_ratio)
            }
            MinMaxFunctionWithoutX::Parallel(func) => func(
                &y[1..(y.len() - 1)],
                n_out * minmax_ratio,
                n_threads.unwrap(), // n_threads cannot be None
            ),
        };
        // inplace + 1
        index.iter_mut().for_each(|elem| *elem += 1);
        // Prepend first and last point
        index.insert(0, 0);
        index.push(y.len() - 1);
        // Get y values at index
        let y = unsafe {
            index
                .iter()
                .map(|i| *y.get_unchecked(*i))
                .collect::<Vec<Ty>>()
        };
        // Apply lttb on the reduced data
        let index_points_selected = lttb_without_x(y.as_slice(), n_out);
        // Return the original index
        return index_points_selected
            .iter()
            .map(|i| index[*i])
            .collect::<Vec<usize>>();
    }
    // Apply lttb on all data when requirement is not met
    lttb_without_x(y, n_out).to_vec()
}
