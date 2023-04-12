import numpy as np
import pytest

from tsdownsample import LTTBDownsampler, M4Downsampler, MinMaxDownsampler
from tsdownsample._python.downsamplers import LTTB_py, M4_py, MinMax_py


@pytest.mark.parametrize(
    "rust_python_pair",
    [
        (MinMaxDownsampler(), MinMax_py()),
        (M4Downsampler(), M4_py()),
        (LTTBDownsampler(), LTTB_py()),
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
