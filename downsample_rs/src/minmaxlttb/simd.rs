use super::super::minmax;
use super::generic::{minmaxlttb_generic, minmaxlttb_generic_without_x};

// use num_traits::{Num, ToPrimitive};
use super::super::lttb::utils::Num;
use ndarray::{Array1, ArrayView1};

extern crate argminmax;
use argminmax::ArgMinMax;

// ----------------------------------- NON-PARALLEL ------------------------------------

// ----------- WITH X

pub fn minmaxlttb_simd<Tx: Num, Ty: Num + PartialOrd>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
{
    minmaxlttb_generic(x, y, n_out, minmax::min_max_simd)
}

// ----------- WITHOUT X

pub fn minmaxlttb_simd_without_x<Ty: Num + PartialOrd>(
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
{
    minmaxlttb_generic_without_x(y, n_out, minmax::min_max_simd)
}

// ------------------------------------- PARALLEL --------------------------------------

// ----------- WITH X

pub fn minmaxlttb_simd_parallel<Tx: Num, Ty: Num + PartialOrd + Send + Sync>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
{
    minmaxlttb_generic(x, y, n_out, minmax::min_max_simd_parallel)
}

// ----------- WITHOUT X

pub fn minmaxlttb_simd_without_x_parallel<Ty: Num + PartialOrd + Send + Sync>(
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    for<'a> ArrayView1<'a, Ty>: ArgMinMax,
{
    minmaxlttb_generic_without_x(y, n_out, minmax::min_max_simd_parallel)
}
