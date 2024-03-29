import warnings
from typing import Union

import numpy as np

# ------------------ Rust Downsamplers ------------------
from tsdownsample._rust import _tsdownsample_rs  # type: ignore[attr-defined]

from .downsampling_interface import (
    AbstractDownsampler,
    AbstractRustDownsampler,
    AbstractRustNaNDownsampler,
)


class MinMaxDownsampler(AbstractRustDownsampler):
    """Downsampler that uses the MinMax algorithm. If the y data contains NaNs, these
    ignored (i.e. the NaNs are not taken into account when selecting data points).

    For each bin, the indices of the minimum and maximum values are selected.
    """

    @property
    def rust_mod(self):
        return _tsdownsample_rs.minmax

    @staticmethod
    def _check_valid_n_out(n_out: int):
        AbstractRustDownsampler._check_valid_n_out(n_out)
        if n_out % 2 != 0:
            raise ValueError("n_out must be even")


class NaNMinMaxDownsampler(AbstractRustNaNDownsampler):
    """Downsampler that uses the MinMax algorithm. If the y data contains NaNs, the
    indices of these NaNs are returned.

    For each bin, the indices of the minimum and maximum values are selected.
    """

    @property
    def rust_mod(self):
        return _tsdownsample_rs.minmax

    @staticmethod
    def _check_valid_n_out(n_out: int):
        AbstractRustDownsampler._check_valid_n_out(n_out)
        if n_out % 2 != 0:
            raise ValueError("n_out must be even")


class M4Downsampler(AbstractRustDownsampler):
    """Downsampler that uses the M4 algorithm. If the y data contains NaNs, these are
    ignored (i.e. the NaNs are not taken into account when selecting data points).

    For each bin, the indices of the first, last, minimum and maximum values are
    selected.
    """

    @property
    def rust_mod(self):
        return _tsdownsample_rs.m4

    @staticmethod
    def _check_valid_n_out(n_out: int):
        AbstractRustDownsampler._check_valid_n_out(n_out)
        if n_out % 4 != 0:
            raise ValueError("n_out must be a multiple of 4")


class NaNM4Downsampler(AbstractRustNaNDownsampler):
    """Downsampler that uses the M4 algorithm. If the y data contains NaNs, the indices
    of these NaNs are returned.

    For each bin, the indices of the first, last, minimum and maximum values are
    selected.
    """

    @property
    def rust_mod(self):
        return _tsdownsample_rs.m4

    @staticmethod
    def _check_valid_n_out(n_out: int):
        AbstractRustDownsampler._check_valid_n_out(n_out)
        if n_out % 4 != 0:
            raise ValueError("n_out must be a multiple of 4")


class LTTBDownsampler(AbstractRustDownsampler):
    """Downsampler that uses the LTTB algorithm."""

    @property
    def rust_mod(self):
        return _tsdownsample_rs.lttb


class MinMaxLTTBDownsampler(AbstractRustDownsampler):
    """Downsampler that uses the MinMaxLTTB algorithm. If the y data contains NaNs,
    these are ignored (i.e. the NaNs are not taken into account when selecting data
    points).

    MinMaxLTTB paper: https://arxiv.org/abs/2305.00332
    """

    @property
    def rust_mod(self):
        return _tsdownsample_rs.minmaxlttb

    def downsample(
        self, *args, n_out: int, minmax_ratio: int = 4, parallel: bool = False, **_
    ):
        assert minmax_ratio > 0, "minmax_ratio must be greater than 0"
        return super().downsample(
            *args, n_out=n_out, parallel=parallel, ratio=minmax_ratio
        )


class NaNMinMaxLTTBDownsampler(AbstractRustNaNDownsampler):
    """Downsampler that uses the MinMaxLTTB algorithm. If the y data contains NaNs, the
    indices of these NaNs are returned.

    MinMaxLTTB paper: https://arxiv.org/abs/2305.00332
    """

    @property
    def rust_mod(self):
        return _tsdownsample_rs.minmaxlttb

    def downsample(
        self, *args, n_out: int, minmax_ratio: int = 4, parallel: bool = False, **_
    ):
        assert minmax_ratio > 0, "minmax_ratio must be greater than 0"
        return super().downsample(
            *args, n_out=n_out, parallel=parallel, ratio=minmax_ratio
        )


# ------------------ EveryNth Downsampler ------------------


class EveryNthDownsampler(AbstractDownsampler):
    """Downsampler that selects every nth data point"""

    def __init__(self, **kwargs):
        super().__init__(check_contiguous=False, **kwargs)

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **_
    ) -> np.ndarray:
        if x is not None:
            name = self.__class__.__name__
            warnings.warn(
                f"x is passed to downsample method of {name}, but is not taken "
                "into account by the current implementation of the EveryNth algorithm."
            )
        step = max(1, len(y) / n_out)
        return np.arange(start=0, stop=len(y) - 0.1, step=step).astype(np.uint)
