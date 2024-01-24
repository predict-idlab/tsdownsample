import numpy as np
import pytest

from tsdownsample import (
    LTTBDownsampler,
    M4Downsampler,
    MinMaxDownsampler,
    NaNM4Downsampler,
    NaNMinMaxDownsampler,
)
from tsdownsample._python.downsamplers import (
    LTTB_py,
    M4_py,
    MinMax_py,
    NaNM4_py,
    NaNMinMax_py,
)


@pytest.mark.parametrize(
    "rust_python_pair",
    [
        (MinMaxDownsampler(), MinMax_py()),
        (M4Downsampler(), M4_py()),
        (LTTBDownsampler(), LTTB_py()),
        # Include NaN downsamplers
        (NaNMinMaxDownsampler(), NaNMinMax_py()),
        (NaNM4Downsampler(), NaNM4_py()),
    ],
)
@pytest.mark.parametrize("n", [10_000, 10_032, 20_321, 23_489])
@pytest.mark.parametrize("n_out", [100, 200, 252])
def test_resampler_accordance(rust_python_pair, n, n_out):
    rust_downsampler, python_downsampler = rust_python_pair
    x = np.arange(n)
    y = np.random.randn(n)
    # Without x passed to the rust downsampler
    assert np.allclose(
        rust_downsampler.downsample(y, n_out=n_out),
        python_downsampler.downsample(x, y, n_out=n_out),
    )
    # With x passed to the rust downsampler
    assert np.allclose(
        rust_downsampler.downsample(x, y, n_out=n_out),
        python_downsampler.downsample(x, y, n_out=n_out),
    )


@pytest.mark.parametrize(
    "rust_python_pair",
    [(NaNMinMaxDownsampler(), NaNMinMax_py()), (NaNM4Downsampler(), NaNM4_py())],
)
@pytest.mark.parametrize("n", [10_000, 10_032, 20_321, 23_489])
@pytest.mark.parametrize("n_random_nans", [100, 200, 500, 2000, 5000])
@pytest.mark.parametrize("n_out", [100, 200, 252])
def test_nan_resampler_accordance(rust_python_pair, n, n_random_nans, n_out):
    rust_downsampler, python_downsampler = rust_python_pair
    x = np.arange(n)
    y = np.random.randn(n)
    y[np.random.choice(y.size, n_random_nans, replace=False)] = np.nan
    # Without x passed to the rust downsampler
    rust_result = rust_downsampler.downsample(y, n_out=n_out)
    python_result = python_downsampler.downsample(x, y, n_out=n_out)
    assert np.allclose(rust_result, python_result)
    # With x passed to the rust downsampler
    assert np.allclose(
        rust_downsampler.downsample(x, y, n_out=n_out),
        python_downsampler.downsample(x, y, n_out=n_out),
    )
