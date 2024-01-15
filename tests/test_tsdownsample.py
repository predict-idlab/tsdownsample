from typing import Iterable

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
from tsdownsample.downsampling_interface import AbstractDownsampler

# TODO: Improve tests
#   - compare implementations with existing plotly_resampler implementations


RUST_DOWNSAMPLERS = [
    MinMaxDownsampler(),
    M4Downsampler(),
    LTTBDownsampler(),
    MinMaxLTTBDownsampler(),
]

OTHER_DOWNSAMPLERS = [EveryNthDownsampler()]


def generate_rust_downsamplers() -> Iterable[AbstractDownsampler]:
    for downsampler in RUST_DOWNSAMPLERS:
        yield downsampler


def generate_all_downsamplers() -> Iterable[AbstractDownsampler]:
    for downsampler in RUST_DOWNSAMPLERS + OTHER_DOWNSAMPLERS:
        yield downsampler


@pytest.mark.parametrize("downsampler", generate_all_downsamplers())
def test_serialization_copy(downsampler: AbstractDownsampler):
    """Test serialization."""
    from copy import copy, deepcopy

    dc = copy(downsampler)
    ddc = deepcopy(downsampler)

    arr = np.arange(10_000)
    orig_downsampled = downsampler.downsample(arr, n_out=100)
    dc_downsampled = dc.downsample(arr, n_out=100)
    ddc_downsampled = ddc.downsample(arr, n_out=100)
    assert np.all(orig_downsampled == dc_downsampled)
    assert np.all(orig_downsampled == ddc_downsampled)


@pytest.mark.parametrize("downsampler", generate_all_downsamplers())
def test_serialization_pickle(downsampler: AbstractDownsampler):
    """Test serialization."""
    import pickle

    dc = pickle.loads(pickle.dumps(downsampler))

    arr = np.arange(10_000)
    orig_downsampled = downsampler.downsample(arr, n_out=100)
    dc_downsampled = dc.downsample(arr, n_out=100)
    assert np.all(orig_downsampled == dc_downsampled)


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_rust_downsampler(downsampler: AbstractDownsampler):
    """Test the Rust downsamplers."""
    arr = np.arange(10_000)
    s_downsampled = downsampler.downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == len(arr) - 1


def test_everynth_downsampler():
    """Test EveryNth downsampler."""
    arr = np.arange(10_000)
    downsampler = EveryNthDownsampler()
    s_downsampled = downsampler.downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == 9_900


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_parallel_downsampling(downsampler: AbstractDownsampler):
    """Test parallel downsampling."""
    arr = np.random.randn(10_000).astype(np.float32)
    s_downsampled = downsampler.downsample(arr, n_out=100)
    s_downsampled_p = downsampler.downsample(arr, n_out=100, parallel=True)
    assert np.all(s_downsampled == s_downsampled_p)


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_parallel_downsampling_with_x(downsampler: AbstractDownsampler):
    """Test parallel downsampling with x."""
    arr = np.random.randn(10_001).astype(np.float32)  # 10_001 to test edge case
    idx = np.arange(len(arr))
    s_downsampled = downsampler.downsample(idx, arr, n_out=100)
    s_downsampled_p = downsampler.downsample(idx, arr, n_out=100, parallel=True)
    assert np.all(s_downsampled == s_downsampled_p)


@pytest.mark.parametrize("downsampler", generate_all_downsamplers())
def test_downsampling_with_x(downsampler: AbstractDownsampler):
    """Test downsampling with x."""
    arr = np.random.randn(2_001).astype(np.float32)  # 2_001 to test edge case
    idx = np.arange(len(arr))
    s_downsampled = downsampler.downsample(arr, n_out=100)
    s_downsampled_x = downsampler.downsample(idx, arr, n_out=100)
    assert np.all(s_downsampled == s_downsampled_x)


@pytest.mark.parametrize("downsampler", generate_all_downsamplers())
def test_downsampling_with_gaps_in_x(downsampler: AbstractDownsampler):
    """Test downsampling with gaps in x.

    With gap we do NOT mean a NaN in the array, but a large gap in the x values.
    """
    # TODO: might improve this test, now we just validate that the code does
    # not crash
    arr = np.random.randn(10_000).astype(np.float32)
    idx = np.arange(len(arr))
    idx[: len(idx) // 2] += len(idx) // 2  # add large gap in x
    s_downsampled = downsampler.downsample(idx, arr, n_out=100)
    assert len(s_downsampled) <= 100
    assert len(s_downsampled) >= 66


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_downsampling_different_dtypes(downsampler: AbstractDownsampler):
    """Test downsampling with different data types."""
    arr_orig = np.random.randint(0, 100, size=10_000)
    res = []
    for dtype_y in supported_dtypes_y:
        arr = arr_orig.astype(dtype_y)
        s_downsampled = downsampler.downsample(arr, n_out=100)
        if dtype_y is not np.bool_:
            res += [s_downsampled]
    for i in range(1, len(res)):
        assert np.all(res[0] == res[i])


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_downsampling_different_dtypes_with_x(downsampler: AbstractDownsampler):
    """Test downsampling with x with different data types."""
    arr_orig = np.random.randint(0, 100, size=10_000)
    idx_orig = np.arange(len(arr_orig))
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


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_downsampling_no_out_of_bounds_different_dtypes(
    downsampler: AbstractDownsampler,
):
    """Test no out of bounds issues when downsampling with different data types."""
    arr_orig = np.random.randint(0, 100, size=100)
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


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_downsampling_no_out_of_bounds_different_dtypes_with_x(
    downsampler: AbstractDownsampler,
):
    """Test no out of bounds issues when downsampling with different data types."""
    arr_orig = np.random.randint(0, 100, size=100)
    idx_orig = np.arange(len(arr_orig))
    for dtype_x in supported_dtypes_x:
        res = []
        idx = idx_orig.astype(dtype_x)
        for dtype_y in supported_dtypes_y:
            arr = arr_orig.astype(dtype_y)
            s_downsampled = downsampler.downsample(idx, arr, n_out=76)
            s_downsampled_p = downsampler.downsample(idx, arr, n_out=76, parallel=True)
            assert np.all(s_downsampled == s_downsampled_p)
            if dtype_y is not np.bool_:
                res += [s_downsampled]
        for i in range(1, len(res)):
            assert np.all(res[0] == res[i])


def test_lttb_no_overflow():
    """Test no overflow when calculating average."""
    ### THIS SHOULD NOT OVERFLOW & HAVE THE SAME RESULT
    arr_orig = np.array([2 * 10**5] * 10_000, dtype=np.float64)
    s_downsampled = LTTBDownsampler().downsample(arr_orig, n_out=100)
    arr = arr_orig.astype(np.float32)
    s_downsampled_f32 = LTTBDownsampler().downsample(arr, n_out=100)
    assert np.all(s_downsampled == s_downsampled_f32)
    ### THIS SHOULD OVERFLOW & THUS HAVE A DIFFERENT RESULT...
    # max float32 is 3.4028235 × 1038 (so 2*10**38 is too big when adding 2 values)
    arr_orig = np.array([2 * 10**38] * 10_000, dtype=np.float64)
    s_downsampled = LTTBDownsampler().downsample(arr_orig, n_out=100)
    arr = arr_orig.astype(np.float32)
    s_downsampled_f32 = LTTBDownsampler().downsample(arr, n_out=100)
    assert not np.all(s_downsampled == s_downsampled_f32)  # TODO :(
    # I will leave this test here, but as many (much larger) libraries do not
    # really account for this, I guess it is perhaps less of an issue than I
    # thought. In the end f32 MAX is 3.4028235 × 1038 & f64 MAX is
    # 1.7976931348623157 × 10308 => which is in the end quite a lot.. (and all
    # integer averages are handled using f64) - f32 is only used for f16 & f32
    # (just as in numpy).


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


@pytest.mark.parametrize("downsampler", generate_rust_downsamplers())
def test_non_contiguous_array(downsampler: AbstractDownsampler):
    """Test non contiguous array."""
    arr = np.random.randint(0, 100, size=10_000)
    arr = arr[::2]
    assert not arr.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError) as e_msg:
        downsampler.downsample(arr, n_out=100)
    assert "must be contiguous" in str(e_msg.value)


def test_everynth_non_contiguous_array():
    """Test non contiguous array."""
    arr = np.random.randint(0, 100, size=10_000)
    arr = arr[::2]
    assert not arr.flags["C_CONTIGUOUS"]
    downsampler = EveryNthDownsampler()
    s_downsampled = downsampler.downsample(arr, n_out=100)
    assert s_downsampled[0] == 0
    assert s_downsampled[-1] == 4950
