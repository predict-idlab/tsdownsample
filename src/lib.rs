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
            let y = y.as_slice().unwrap();
            let sampled_indices = $resample_mod::$resample_fn(y, n_out);
            sampled_indices.into_pyarray(py)
        }
        // Add the function to the module
        $mod.add_wrapped(wrap_pyfunction!($name))?;
    };
}

macro_rules! _create_pyfunc_without_x_with_ratio {
    ($name:ident, $resample_mod:ident, $resample_fn:ident, $type:ty, $mod:ident) => {
        // Create the Python function
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            y: PyReadonlyArray1<$type>,
            n_out: usize,
            ratio: usize,
        ) -> &'py PyArray1<usize> {
            let y = y.as_slice().unwrap();
            let sampled_indices = $resample_mod::$resample_fn(y, n_out, ratio);
            sampled_indices.into_pyarray(py)
        }
        // Add the function to the module
        $mod.add_wrapped(wrap_pyfunction!($name))?;
    };
}

macro_rules! _create_pyfuncs_without_x_generic {
    ($create_macro:ident, $resample_mod:ident, $resample_fn:ident, $mod:ident, $($t:ty)*) => {
        $(
            paste! {
                $create_macro!([<downsample_ $t>], $resample_mod, $resample_fn, $t, $mod);
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
            let x = x.as_slice().unwrap();
            let y = y.as_slice().unwrap();
            let sampled_indices = $resample_mod::$resample_fn(x, y, n_out);
            sampled_indices.into_pyarray(py)
        }
        // Add the function to the module
        $mod.add_wrapped(wrap_pyfunction!($name))?;
    };
}

macro_rules! _create_pyfunc_with_x_with_ratio {
    ($name:ident, $resample_mod:ident, $resample_fn:ident, $type_x:ty, $type_y:ty, $mod:ident) => {
        // Create the Python function
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            x: PyReadonlyArray1<$type_x>,
            y: PyReadonlyArray1<$type_y>,
            n_out: usize,
            ratio: usize,
        ) -> &'py PyArray1<usize> {
            let x = x.as_slice().unwrap();
            let y = y.as_slice().unwrap();
            let sampled_indices = $resample_mod::$resample_fn(x, y, n_out, ratio);
            sampled_indices.into_pyarray(py)
        }
        // Add the function to the module
        $mod.add_wrapped(wrap_pyfunction!($name))?;
    };
}

macro_rules! _create_pyfuncs_with_x_generic {
    // ($create_macro:ident, $resample_mod:ident, $resample_fn:ident, $mod:ident, $($t:ty)+) => {
    //     // The macro will implement the function for all combinations of $t (for type x and y).
    //     // (duplicate the list of types to iterate over all combinations)
    //     _create_pyfuncs_with_x_generic!(@inner $create_macro, $resample_mod, $resample_fn, $mod, $($t)+; $($t),+);
    // };

    ($create_macro:ident, $resample_mod:ident, $resample_fn:ident, $mod:ident, $($tx:ty)+, $($ty:ty)+) => {
        // The macro will implement the function for all combinations of $tx and $ty (for respectively type x and y).
        _create_pyfuncs_with_x_generic!(@inner $create_macro, $resample_mod, $resample_fn, $mod, $($tx)+; $($ty),+);
    };

    // Base case: there is only one type (for y) left
    (@inner $create_macro:ident, $resample_mod:ident, $resample_fn:ident, $mod:ident, $($tx:ty)+; $ty:ty) => {
        $(
            paste! {
                $create_macro!([<downsample_ $tx _ $ty>], $resample_mod, $resample_fn, $tx, $ty, $mod);
            }
        )*
    };
    // The head/tail recursion: pick the first element -> apply the base case, and recurse over the rest.
    (@inner $create_macro:ident, $resample_mod:ident, $resample_fn:ident, $mod:ident, $($tx:ty)+; $ty_head:ty, $($ty_rest:ty),+) => {
        _create_pyfuncs_with_x_generic!(@inner $create_macro, $resample_mod, $resample_fn, $mod, $($tx)+; $ty_head);
        _create_pyfuncs_with_x_generic!(@inner $create_macro, $resample_mod, $resample_fn, $mod, $($tx)+; $($ty_rest),+);
    };

    // Huge thx to https://stackoverflow.com/a/54552848
    // and https://users.rust-lang.org/t/tail-recursive-macros/905/3
}

// ------ Main macros ------

macro_rules! _create_pyfuncs_without_x_helper {
    ($pyfunc_fn:ident, $resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_without_x_generic!($pyfunc_fn, $resample_mod, $resample_fn, $mod, f16 f32 f64 i8 i16 i32 i64 u8 u16 u32 u64);
    };
}

macro_rules! create_pyfuncs_without_x {
    // Use @threaded to differentiate between the single and multithreaded versions
    ($resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_without_x_helper!(
            _create_pyfunc_without_x,
            $resample_mod,
            $resample_fn,
            $mod
        );
    };
}

macro_rules! create_pyfuncs_without_x_with_ratio {
    // Use @threaded to differentiate between the single and multithreaded versions
    ($resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_without_x_helper!(
            _create_pyfunc_without_x_with_ratio,
            $resample_mod,
            $resample_fn,
            $mod
        );
    };
}

macro_rules! _create_pyfuncs_with_x_helper {
    ($pyfunc_fn:ident, $resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_with_x_generic!($pyfunc_fn, $resample_mod, $resample_fn, $mod, f32 f64 i16 i32 i64 u16 u32 u64, f16 f32 f64 i8 i16 i32 i64 u8 u16 u32 u64);
    };
}

macro_rules! create_pyfuncs_with_x {
    // Use @threaded to differentiate between the single and multithreaded versions
    ($resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_with_x_helper!(_create_pyfunc_with_x, $resample_mod, $resample_fn, $mod);
    };
}

macro_rules! create_pyfuncs_with_x_with_ratio {
    // Use @threaded to differentiate between the single and multithreaded versions
    ($resample_mod:ident, $resample_fn:ident, $mod:ident) => {
        _create_pyfuncs_with_x_helper!(
            _create_pyfunc_with_x_with_ratio,
            $resample_mod,
            $resample_fn,
            $mod
        );
    };
}

// -------------------------------------- MINMAX ---------------------------------------

use downsample_rs::minmax as minmax_mod;

// Create a sub module for the minmax algorithm
#[pymodule]
fn minmax(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // ----------------- SEQUENTIAL

    let sequential_mod = PyModule::new(_py, "sequential")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(minmax_mod, min_max_without_x, sequential_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(minmax_mod, min_max_with_x, sequential_mod);
    }

    // ----------------- PARALLEL

    let parallel_mod = PyModule::new(_py, "parallel")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(minmax_mod, min_max_without_x_parallel, parallel_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(minmax_mod, min_max_with_x_parallel, parallel_mod);
    }

    // Add the sub modules to the module
    m.add_submodule(sequential_mod)?;
    m.add_submodule(parallel_mod)?;

    Ok(())
}

// --------------------------------------- M4 ------------------------------------------

use downsample_rs::m4 as m4_mod;

// Create a sub module for the M4 algorithm
#[pymodule]
fn m4(_py: Python, m: &PyModule) -> PyResult<()> {
    // ----------------- SEQUENTIAL

    let sequential_mod = PyModule::new(_py, "sequential")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(m4_mod, m4_without_x, sequential_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(m4_mod, m4_with_x, sequential_mod);
    }

    // ----------------- PARALLEL

    let parallel_mod = PyModule::new(_py, "parallel")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(m4_mod, m4_without_x_parallel, parallel_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(m4_mod, m4_with_x_parallel, parallel_mod);
    }

    // Add the sub modules to the module
    m.add_submodule(sequential_mod)?;
    m.add_submodule(parallel_mod)?;

    Ok(())
}

// -------------------------------------- LTTB -----------------------------------------

use downsample_rs::lttb as lttb_mod;

// Create a sub module for the LTTB algorithm
#[pymodule]
fn lttb(_py: Python, m: &PyModule) -> PyResult<()> {
    // ----------------- SEQUENTIAL

    let sequential_mod = PyModule::new(_py, "sequential")?;

    // Create the Python functions for the module
    // ----- WITHOUT X
    {
        create_pyfuncs_without_x!(lttb_mod, lttb_without_x, sequential_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x!(lttb_mod, lttb_with_x, sequential_mod);
    }

    // Add the sub modules to the module
    m.add_submodule(sequential_mod)?;

    Ok(())
}

// -------------------------------------- MINMAXLTTB -----------------------------------------

use downsample_rs::minmaxlttb as minmaxlttb_mod;

// Create a sub module for the MINMAXLTTB algorithm
#[pymodule]
fn minmaxlttb(_py: Python, m: &PyModule) -> PyResult<()> {
    // ----------------- SEQUENTIAL

    let sequential_mod = PyModule::new(_py, "sequential")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x_with_ratio!(minmaxlttb_mod, minmaxlttb_without_x, sequential_mod);
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x_with_ratio!(minmaxlttb_mod, minmaxlttb_with_x, sequential_mod);
    }

    // ----------------- PARALLEL

    let parallel_mod = PyModule::new(_py, "parallel")?;

    // ----- WITHOUT X
    {
        create_pyfuncs_without_x_with_ratio!(
            minmaxlttb_mod,
            minmaxlttb_without_x_parallel,
            parallel_mod
        );
    }

    // ----- WITH X
    {
        create_pyfuncs_with_x_with_ratio!(minmaxlttb_mod, minmaxlttb_with_x_parallel, parallel_mod);
    }

    // Add the submodules to the module
    m.add_submodule(sequential_mod)?;
    m.add_submodule(parallel_mod)?;

    Ok(())
}

// ------------------------------- DOWNSAMPLING MODULE ------------------------------ //

#[pymodule] // The super module
#[pyo3(name = "_tsdownsample_rs")] // How the module is imported in Python: https://github.com/PyO3/maturin/issues/256#issuecomment-1038576218
fn tsdownsample(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(minmax))?;
    m.add_wrapped(wrap_pymodule!(m4))?;
    m.add_wrapped(wrap_pymodule!(lttb))?;
    m.add_wrapped(wrap_pymodule!(minmaxlttb))?;

    Ok(())
}
