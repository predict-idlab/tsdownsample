import numpy as np
import pytest

from tsdownsample import (
    EveryNthDownsampler,
    LTTBDownsampler,
    M4Downsampler,
    MinMaxDownsampler,
    MinMaxLTTBDownsampler,
)

NB_SAMPLES = ["100,000", "1,000,000"]
N_OUT = ["100", "1,000", "5,000"]
Y_DTYPES = [np.float32, np.float64] + [np.int32, np.int64]


# --------------------------------------------------------------------------- #
#                               MinMaxDownsampler
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="minmax")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_minmax_no_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the MinMaxDownsampler."""
    downsampler = MinMaxDownsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, y, n_out=n_out, parallel=parallel)


@pytest.mark.benchmark(group="minmax")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_minmax_with_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the MinMaxDownsampler."""
    downsampler = MinMaxDownsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    x = np.arange(n_samples)
    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, x, y, n_out=n_out, parallel=parallel)


# --------------------------------------------------------------------------- #
#                               M4Downsampler
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="m4")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_m4_no_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the M4Downsampler."""
    downsampler = M4Downsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, y, n_out=n_out, parallel=parallel)


@pytest.mark.benchmark(group="m4")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_m4_with_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the M4Downsampler."""
    downsampler = M4Downsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    x = np.arange(n_samples)
    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, x, y, n_out=n_out, parallel=parallel)


# --------------------------------------------------------------------------- #
#                              LTTBDownsampler
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="lttb")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_lttb_no_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the LTTBDownsampler."""
    downsampler = LTTBDownsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, y, n_out=n_out, parallel=parallel)


@pytest.mark.benchmark(group="lttb")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_lttb_with_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the LTTBDownsampler."""
    downsampler = LTTBDownsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    x = np.arange(n_samples)
    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, x, y, n_out=n_out, parallel=parallel)


# --------------------------------------------------------------------------- #
#                          MinMaxLTTBDownsampler
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="minmaxlttb")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_minmaxlttb_no_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the MinMaxLTTBDownsampler."""
    downsampler = MinMaxLTTBDownsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, y, n_out=n_out, parallel=parallel)


@pytest.mark.benchmark(group="minmaxlttb")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
@pytest.mark.parametrize("dtype", Y_DTYPES)
@pytest.mark.parametrize("parallel", [False, True])
def test_minmaxlttb_with_x(benchmark, n_samples, n_out, dtype, parallel):
    """Test the MinMaxLTTBDownsampler."""
    downsampler = MinMaxLTTBDownsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    x = np.arange(n_samples)
    y = np.random.randn(n_samples).astype(dtype)

    benchmark(downsampler.downsample, x, y, n_out=n_out, parallel=parallel)


# --------------------------------------------------------------------------- #
#                             EveryNthDownsampler
# --------------------------------------------------------------------------- #


@pytest.mark.benchmark(group="everynth")
@pytest.mark.parametrize("n_samples", NB_SAMPLES)
@pytest.mark.parametrize("n_out", N_OUT)
def test_everynth(benchmark, n_samples, n_out):
    """Test the EveryNthDownsampler."""
    downsampler = EveryNthDownsampler()
    n_samples = int(n_samples.replace(",", ""))
    n_out = int(n_out.replace(",", ""))

    y = np.random.randn(n_samples)

    benchmark(downsampler.downsample, y, n_out=n_out)
