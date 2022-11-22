extern crate downsample_rs;
extern crate paste;

// TODO
// - support bool
//      => issue: does not have to_primitive (necessary for lttb)
// - m4 & minmax should determine bin size on x-range!
//      code now assumes equal bin size

use half::f16;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use paste::paste;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;

/// ------------------------- MACROS -------------------------

// Create macros to avoid duplicate code for the various resample functions over the
// different data types.

// ----- Helper macros -----

// Without x-range

macro_rules! _create_pyfunc_without_x {
    ($name:ident, $resample_mod:ident, $resample_fn:ident, $type:ty, $mod:ident) => {
        // Create the Python function
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            y: PyReadonlyArray1<$type>,
            n_out: usize,
        ) -> &'py PyArray1<usize> {
            let y = y.as_array();
            let sampled_indices = $resample_mod::$resample_fn(y, n_out);
            sampled_indices.into_pyarray(py)
        }
        // Add the function to the module
        $mod.add_wrapped(wrap_pyfunction!($name))?;
    }; // ($name:ident, $resample_mod:ident, $resample_fn:ident, $type:ty, $mod:ident, $cast_type:ty) => {
       //     // Create the Python function
       //     #[pyfunction]
       //     fn $name<'py>(
       //         py: Python<'py>,
       //         y: PyReadonlyArray1<$type>,
       //         n_out: usize,
       //     ) -> &'py PyArray1<usize> {
       //         let y = y.as_array().mapv(|v| v as $cast_type);
       //         let sampled_indices = $resample_mod::$resample_fn(y.view(), n_out);
       //         sampled_indices.into_pyarray(py)
       //     }
       //     // Add the function to the module
       //     $mod.add_wrapped(wrap_pyfunction!($name))?;
       // };
}

macro_rules! _create_pyfuncs_without_x {
    ($resample_mod:ident, $resample_fn:ident, $mod:ident, $($t:ty)*) => {
        $(
            paste! {
                _create_pyfunc_without_x!([<downsample_ $t>], $resample_mod, $resample_fn, $t, $mod);
            }
        )*
    };
}

// With x-range

macro_rules! _create_pyfunc_with_x {
    ($name:ident, $resample_mod:ident, $resample_fn:ident, $type_x:ty, $type_y:ty, $mod:ident) => {
        // Create the Python function
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<$type_x>,
            y: PyReadonlyArray1<$type_y>,
            n_out: usize,
        ) -> &'py PyArray1<usize> {
            let x = x.as_array();
            let y = y.as_array();
            let sampled_indices = $resample_mod::$resample_fn(x, y, n_out);
            sampled_indices.into_pyarray(py)
        }
        // Add the function to the module
        $mod.add_wrapped(wrap_pyfunction!($name))?;
    };
}

macro_rules! _create_pyfuncs_with_x {
    ($resample_mod:ident, $resample_fn:ident, $mod:ident, $($t:ty)*) => {
        // When only one expression of type $t is passed, the macro will implement the
        // function for all combinations of $t (for type x and y).
        _create_pyfuncs_with_x!($resample_mod, $resample_fn, $mod, $($t)*, $($t)*);
    };
    ($resample_mod:ident, $resample_fn:ident, $mod:ident, $($tx:ty)*, $($ty:ty)*) => {
        $(
            paste! {
                _create_pyfunc_with_x!([<downsample_ $tx _ $ty>], $resample_mod, $resample_fn, $tx, $ty, $mod);
            }
        )*
    };
}

// ------ Main macros ------

macro_rules! create_pyfuncs_without_x {
    ($resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_without_x!($resample_mod, $resample_fn, $mod, f16 f32 f64 i16 i32 i64 u16 u32 u64);
    };
}

macro_rules! create_pyfuncs_with_x {
    ($resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_with_x!($resample_mod, $resample_fn, $mod, f16 f32 f64 i16 i32 i64 u16 u32 u64);
    };
}

// -------------------------------------- MINMAX ---------------------------------------

use downsample_rs::minmax as minmax_mod;

// Create a sub module for the minmax algorithm
#[pymodule]
fn min_max(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // ----------------- SCALAR

    let scalar_mod = PyModule::new(_py, "scalar")?;
    {
        create_pyfuncs_without_x!(minmax_mod, min_max_scalar, scalar_mod);
    }

    // ----------------- SCALAR PARALLEL

    let scalar_parallel_mod = PyModule::new(_py, "scalar_parallel")?;
    {
        create_pyfuncs_without_x!(minmax_mod, min_max_scalar_parallel, scalar_parallel_mod);
    }

    // ----------------- SIMD

    let simd_mod = PyModule::new(_py, "simd")?;
    {
        create_pyfuncs_without_x!(minmax_mod, min_max_simd, simd_mod);
    }

    // ----------------- SIMD PARALLEL

    let simd_parallel_mod = PyModule::new(_py, "simd_parallel")?;
    {
        create_pyfuncs_without_x!(minmax_mod, min_max_simd_parallel, simd_parallel_mod);
    }

    // Add the sub modules to the module
    m.add_submodule(scalar_mod)?;
    m.add_submodule(scalar_parallel_mod)?;
    m.add_submodule(simd_mod)?;
    m.add_submodule(simd_parallel_mod)?;

    Ok(())
}

// --------------------------------------- M4 ------------------------------------------

use downsample_rs::m4 as m4_mod;

// Create a sub module for the M4 algorithm
#[pymodule]
fn m4(_py: Python, m: &PyModule) -> PyResult<()> {
    // ----------------- SCALAR

    let scalar_mod = PyModule::new(_py, "scalar")?;
    {
        create_pyfuncs_without_x!(m4_mod, m4_scalar, scalar_mod);
    }

    // ----------------- SCALAR PARALLEL

    let scalar_parallel_mod = PyModule::new(_py, "scalar_parallel")?;
    {
        create_pyfuncs_without_x!(m4_mod, m4_scalar_parallel, scalar_parallel_mod);
    }

    // ----------------- SIMD

    let simd_mod = PyModule::new(_py, "simd")?;
    {
        create_pyfuncs_without_x!(m4_mod, m4_simd, simd_mod);
    }

    // ----------------- SIMD PARALLEL

    let simd_parallel_mod = PyModule::new(_py, "simd_parallel")?;
    {
        create_pyfuncs_without_x!(m4_mod, m4_simd_parallel, simd_parallel_mod);
    }

    // Add the sub modules to the module
    m.add_submodule(scalar_mod)?;
    m.add_submodule(scalar_parallel_mod)?;
    m.add_submodule(simd_mod)?;
    m.add_submodule(simd_parallel_mod)?;

    Ok(())
}

// -------------------------------------- LTTB -----------------------------------------

use downsample_rs::lttb as lttb_mod;

// Create a sub module for the LTTB algorithm
#[pymodule]
fn lttb(_py: Python, m: &PyModule) -> PyResult<()> {
    // ----------------- SCALAR

    let scalar_mod = PyModule::new(_py, "scalar")?;

    // Create the Python functions for the module
    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(lttb_mod, lttb_without_x, scalar_mod);
    }

    // ----- WITH X TODO
    {
        create_pyfuncs_with_x!(lttb_mod, lttb, scalar_mod);
    }

    // Add the sub modules to the module
    m.add_submodule(scalar_mod)?;

    Ok(())
}

// -------------------------------------- MINMAXLTTB -----------------------------------------

use downsample_rs::minmaxlttb as minmaxlttb_mod;

// Create a sub module for the MINMAXLTTB algorithm
#[pymodule]
fn minmaxlttb(_py: Python, m: &PyModule) -> PyResult<()> {
    // ----------------- SCALAR

    let scalar_mod = PyModule::new(_py, "scalar")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(minmaxlttb_mod, minmaxlttb_scalar_without_x, scalar_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(minmaxlttb_mod, minmaxlttb_scalar, scalar_mod);
    }

    // ----------------- SCALAR PARALLEL

    let scalar_parallel_mod = PyModule::new(_py, "scalar_parallel")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(
            minmaxlttb_mod,
            minmaxlttb_scalar_without_x_parallel,
            scalar_parallel_mod
        );
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(
            minmaxlttb_mod,
            minmaxlttb_scalar_parallel,
            scalar_parallel_mod
        );
    }

    // ----------------- SIMD

    let simd_mod = PyModule::new(_py, "simd")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(minmaxlttb_mod, minmaxlttb_simd_without_x, simd_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(minmaxlttb_mod, minmaxlttb_simd, simd_mod);
    }

    // ----------------- SIMD PARALLEL

    let simd_parallel_mod = PyModule::new(_py, "simd_parallel")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(
            minmaxlttb_mod,
            minmaxlttb_simd_without_x_parallel,
            simd_parallel_mod
        );
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(minmaxlttb_mod, minmaxlttb_simd_parallel, simd_parallel_mod);
    }

    // Add the submodules to the module
    m.add_submodule(scalar_mod)?;
    m.add_submodule(scalar_parallel_mod)?;
    m.add_submodule(simd_mod)?;
    m.add_submodule(simd_parallel_mod)?;

    Ok(())
}

// ------------------------------- DOWNSAMPLING MODULE ------------------------------ //

#[pymodule] // The super module
fn tsdownsample_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(min_max))?;
    m.add_wrapped(wrap_pymodule!(m4))?;
    m.add_wrapped(wrap_pymodule!(lttb))?;
    m.add_wrapped(wrap_pymodule!(minmaxlttb))?;

    _py.run(
        "\
import sys
sys.modules['tsdownsample_rs.min_max'] = min_max
sys.modules['tsdownsample_rs.m4'] = m4
sys.modules['tsdownsample_rs.lttb'] = lttb
sys.modules['tsdownsample_rs.minmaxlttb'] = minmaxlttb
            ",
        None,
        Some(m.dict()),
    )?;

    Ok(())
}
