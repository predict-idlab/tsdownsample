use super::super::lttb::utils::Num;
use super::super::minmax;
use super::generic::{minmaxlttb_generic, minmaxlttb_generic_without_x};
use ndarray::{Array1, ArrayView1};

extern crate argminmax;
use argminmax::{ScalarArgMinMax, SCALAR};

// ----------------------------------- NON-PARALLEL ------------------------------------

pub fn minmaxlttb_scalar<Tx: Num, Ty: Num + PartialOrd>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic(x, y, n_out, minmax::min_max_scalar)
}

pub fn minmaxlttb_scalar_without_x<Ty: Num + PartialOrd>(
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic_without_x(y, n_out, minmax::min_max_scalar)
}

// ------------------------------------- PARALLEL --------------------------------------

pub fn minmaxlttb_scalar_parallel<Tx: Num + Send + Sync, Ty: Num + PartialOrd + Send + Sync>(
    x: ArrayView1<Tx>,
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic(x, y, n_out, minmax::min_max_scalar_parallel)
}

pub fn minmaxlttb_scalar_without_x_parallel<Ty: Num + PartialOrd + Send + Sync>(
    y: ArrayView1<Ty>,
    n_out: usize,
) -> Array1<usize>
where
    SCALAR: ScalarArgMinMax<Ty>,
{
    minmaxlttb_generic_without_x(y, n_out, minmax::min_max_scalar_parallel)
}

// ---- TEST

#[cfg(test)]
mod tests {
    extern crate dev_utils;

    use dev_utils::utils;

    use super::{minmaxlttb_scalar, minmaxlttb_scalar_without_x};
    use ndarray::{array, s, Array1};

    // TODO
}
