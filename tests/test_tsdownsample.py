import numpy as np
import pytest
from test_config import supported_dtypes_x, supported_dtypes_y

from tsdownsample import (  # MeanDownsampler,; MedianDownsampler,
    EveryNthDownsampler,
    LTTBDownsampler,
    M4Downsampler,
    MinMaxDownsampler,
    MinMaxLTTBDownsampler,
)

# TODO: Improve tests
#   - compare implementations with existing plotly_resampler implementations


def test_m4_downsampler():
    """Test M4 downsampler."""
    arr = np.array(np.arange(10_000))
    s_downsampled = M4Downsampler().downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == len(arr) - 1


def test_minmax_downsampler():
    """Test MinMax downsampler."""
    arr = np.array(np.arange(10_000))
    s_downsampled = MinMaxDownsampler().downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == len(arr) - 1


def test_lttb_downsampler():
    """Test LTTB downsampler."""
    arr = np.array(np.arange(10_000))
    s_downsampled = LTTBDownsampler().downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == len(arr) - 1


def test_minmaxlttb_downsampler():
    """Test MinMaxLTTB downsampler."""
    arr = np.array(np.arange(10_000))
    s_downsampled = MinMaxLTTBDownsampler().downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == len(arr) - 1


def test_everynth_downsampler():
    """Test EveryNth downsampler."""
    arr = np.array(np.arange(10_000))
    s_downsampled = EveryNthDownsampler().downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == 9_900


## Parallel downsampling

rust_downsamplers = [
    MinMaxDownsampler(),
    M4Downsampler(),
    LTTBDownsampler(),
    MinMaxLTTBDownsampler(),
]


def test_parallel_downsampling():
    """Test parallel downsampling."""
    arr = np.random.randn(10_000).astype(np.float32)
    for downsampler in rust_downsamplers:
        s_downsampled = downsampler.downsample(arr, n_out=100, parallel=False)
        s_downsampled_p = downsampler.downsample(arr, n_out=100, parallel=True)
        assert np.all(s_downsampled == s_downsampled_p)


def test_parallel_downsampling_with_x():
    """Test parallel downsampling with x."""
    arr = np.random.randn(10_001).astype(np.float32)  # 10_001 to test edge case
    idx = np.arange(len(arr))
    for downsampler in rust_downsamplers:
        s_downsampled = downsampler.downsample(idx, arr, n_out=100, parallel=False)
        s_downsampled_p = downsampler.downsample(idx, arr, n_out=100, parallel=True)
        assert np.all(s_downsampled == s_downsampled_p)


## Using x

all_downsamplers = rust_downsamplers + [EveryNthDownsampler()]


def test_downsampling_with_x():
    """Test downsampling with x."""
    arr = np.random.randn(2_001).astype(np.float32)  # 2_001 to test edge case
    idx = np.arange(len(arr))
    for downsampler in all_downsamplers:
        s_downsampled = downsampler.downsample(arr, n_out=100)
        s_downsampled_x = downsampler.downsample(idx, arr, n_out=100)
        assert np.all(s_downsampled == s_downsampled_x)


## Data types


def test_downsampling_different_dtypes():
    """Test downsampling with different data types."""
    arr_orig = np.random.randint(0, 100, size=10_000)
    for downsampler in rust_downsamplers:
        res = []
        for dtype in supported_dtypes_y:
            arr = arr_orig.astype(dtype)
            s_downsampled = downsampler.downsample(arr, n_out=100)
            if dtype is not np.bool_:
                res += [s_downsampled]
        for i in range(1, len(res)):
            assert np.all(res[0] == res[i])


def test_downsampling_different_dtypes_with_x():
    """Test downsampling with different data types."""
    arr_orig = np.random.randint(0, 100, size=10_000)
    idx_orig = np.arange(len(arr_orig))
    for downsampler in rust_downsamplers:
        for dtype_x in supported_dtypes_x:
            res = []
            idx = idx_orig.astype(dtype_x)
            for dtype_y in supported_dtypes_y:
                arr = arr_orig.astype(dtype_y)
                s_downsampled = downsampler.downsample(idx, arr, n_out=100)
                if dtype_y is not np.bool_:
                    res += [s_downsampled]
            for i in range(1, len(res)):
                assert np.all(res[0] == res[i])


### Check no out of bounds indexing


def test_downsampling_no_out_of_bounds_different_dtypes():
    """Test no out of bounds issues when downsampling with different data types."""
    arr_orig = np.random.randint(0, 100, size=100)
    for downsampler in rust_downsamplers:
        res = []
        for dtype in supported_dtypes_y:
            arr = arr_orig.astype(dtype)
            s_downsampled = downsampler.downsample(arr, n_out=76)
            s_downsampled_p = downsampler.downsample(arr, n_out=76, parallel=True)
            assert np.all(s_downsampled == s_downsampled_p)
            if dtype is not np.bool_:
                res += [s_downsampled]
        for i in range(1, len(res)):
            assert np.all(res[0] == res[i])


def test_downsampling_no_out_of_bounds_different_dtypes_with_x():
    """Test no out of bounds issues when downsampling with different data types."""
    arr_orig = np.random.randint(0, 100, size=100)
    idx_orig = np.arange(len(arr_orig))
    for downsampler in rust_downsamplers:
        for dtype_x in supported_dtypes_x:
            res = []
            idx = idx_orig.astype(dtype_x)
            for dtype_y in supported_dtypes_y:
                arr = arr_orig.astype(dtype_y)
                s_downsampled = downsampler.downsample(idx, arr, n_out=76)
                s_downsampled_p = downsampler.downsample(
                    idx, arr, n_out=76, parallel=True
                )
                assert np.all(s_downsampled == s_downsampled_p)
                if dtype_y is not np.bool_:
                    res += [s_downsampled]
            for i in range(1, len(res)):
                assert np.all(res[0] == res[i])


### Invalid n_out


def test_invalid_nout():
    """Test invalid n_out."""
    arr = np.random.randint(0, 100, size=10_000)
    with pytest.raises(ValueError):
        LTTBDownsampler().downsample(arr, n_out=-1)
    with pytest.raises(ValueError):
        # Should be even
        MinMaxDownsampler().downsample(arr, n_out=33)
    with pytest.raises(ValueError):
        # Should be multiple of 4
        M4Downsampler().downsample(arr, n_out=34)


### Unsupported dtype


def test_error_unsupported_dtype():
    """Test unsupported dtype."""
    arr = np.random.randint(0, 100, size=10_000)
    arr = arr.astype("object")
    with pytest.raises(ValueError):
        MinMaxDownsampler().downsample(arr, n_out=100)


def test_error_invalid_args():
    """Test invalid arguments."""
    arr = np.random.randint(0, 100, size=10_000)
    # No args
    with pytest.raises(ValueError) as e_msg:
        MinMaxDownsampler().downsample(n_out=100, parallel=True)
    assert "takes 1 or 2 positional arguments" in str(e_msg.value)
    # Too many args
    with pytest.raises(ValueError) as e_msg:
        MinMaxDownsampler().downsample(arr, arr, arr, n_out=100, parallel=True)
    assert "takes 1 or 2 positional arguments" in str(e_msg.value)
    # Invalid y
    with pytest.raises(ValueError) as e_msg:
        MinMaxDownsampler().downsample(arr.reshape(5, 2_000), n_out=100, parallel=True)
    assert "y must be 1D" in str(e_msg.value)
    # Invalid x
    with pytest.raises(ValueError) as e_msg:
        MinMaxDownsampler().downsample(
            arr.reshape(5, 2_000), arr, n_out=100, parallel=True
        )
    assert "x must be 1D" in str(e_msg.value)
    # Invalid x and y (different length)
    with pytest.raises(ValueError) as e_msg:
        MinMaxDownsampler().downsample(arr, arr[:-1], n_out=100, parallel=True)
    assert "x and y must have the same length" in str(e_msg.value)
