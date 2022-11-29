import numpy as np
import pandas as pd

from tsdownsample import (
    EveryNthDownsampler,
    LTTBDownsampler,
    M4Downsampler,
    MeanDownsampler,
    MedianDownsampler,
    MinMaxDownsampler,
    MinMaxLTTBDownsampler,
)

# Very basic & poor tests
# TODO: Improve tests
#   - test with and without x
#   - test with and without parallel
#   - test with different data types
#   - test with different downsamplers
#   - compare implementations with existing plotly_resampler implementations


def test_m4_downsampler():
    """Test M4 downsampler."""
    arr = np.array(np.arange(10_000))
    s = pd.Series(arr)
    s_downsampled = M4Downsampler.downsample(s, 100)
    assert s_downsampled.values[0] == 0
    assert s_downsampled.values[-1] == len(arr) - 1


def test_minmax_downsampler():
    """Test MinMax downsampler."""
    arr = np.array(np.arange(10_000))
    s = pd.Series(arr)
    s_downsampled = MinMaxDownsampler.downsample(s, 100)
    assert s_downsampled.values[0] == 0
    assert s_downsampled.values[-1] == len(arr) - 1


def test_lttb_downsampler():
    """Test LTTB downsampler."""
    arr = np.array(np.arange(10_000))
    s = pd.Series(arr)
    s_downsampled = LTTBDownsampler.downsample(s, 100)
    assert s_downsampled.values[0] == 0
    assert s_downsampled.values[-1] == len(arr) - 1


def test_minmaxlttb_downsampler():
    """Test MinMaxLTTB downsampler."""
    arr = np.array(np.arange(10_000))
    s = pd.Series(arr)
    s_downsampled = MinMaxLTTBDownsampler.downsample(s, 100, 30)
    assert s_downsampled.values[0] == 0
    assert s_downsampled.values[-1] == len(arr) - 1


def test_mean_downsampler():
    """Test Mean downsampler."""
    arr = np.array(np.arange(10_000))
    s = pd.Series(arr)
    s_downsampled = MeanDownsampler.downsample(s, 100)
    assert s_downsampled.values[0] == 49.5
    assert s_downsampled.values[-1] == 9_949.5


def test_median_downsampler():
    """Test Median downsampler."""
    arr = np.array(np.arange(10_000))
    s = pd.Series(arr)
    s_downsampled = MedianDownsampler.downsample(s, 100)
    assert s_downsampled.values[0] == 49.5
    assert s_downsampled.values[-1] == 9_949.5


def test_everynth_downsampler():
    """Test EveryNth downsampler."""
    arr = np.array(np.arange(10_000))
    s = pd.Series(arr)
    s_downsampled = EveryNthDownsampler.downsample(s, 100)
    assert s_downsampled.values[0] == 0
    assert s_downsampled.values[-1] == 9_900


## Parallel downsampling

rust_downsamplers = [
    MinMaxDownsampler,
    M4Downsampler,
    LTTBDownsampler,
    MinMaxLTTBDownsampler,
]


def test_parallel_downsampling():
    """Test parallel downsampling."""
    arr = np.random.randn(10_000).astype(np.float32)
    s = pd.Series(arr)
    for downsampler in rust_downsamplers:
        args = []
        if downsampler == MinMaxLTTBDownsampler:
            args = [30]
        s_downsampled = downsampler.downsample(s, 100, *args, parallel=False)
        s_downsampled_p = downsampler.downsample(s, 100, *args, parallel=True)
        assert s_downsampled.equals(s_downsampled_p)


## Data types

supported_dtypes = [
    np.float16,
    np.float32,
    np.float64,
    np.int16,
    np.int32,
    np.int64,
    np.uint16,
    np.uint32,
    np.uint64,
]


def test_downsampling_different_dtypes():
    """Test downsampling with different data types."""
    arr = np.random.randn(10_000)
    for dtype in supported_dtypes:
        s = pd.Series(arr.astype(dtype))
        s_downsampled = MinMaxDownsampler.downsample(s, 100)
        assert s_downsampled.dtype == dtype
