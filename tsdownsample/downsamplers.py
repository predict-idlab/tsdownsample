import math
import warnings
from typing import Union

import numpy as np

# ------------------ Rust Downsamplers ------------------
from tsdownsample._rust import _tsdownsample_rs  # type: ignore[attr-defined]

from .downsampling_interface import AbstractDownsampler, AbstractRustDownsampler


class MinMaxDownsampler(AbstractRustDownsampler):
    def __init__(self) -> None:
        super().__init__(_tsdownsample_rs.minmax)

    @staticmethod
    def _check_valid_n_out(n_out: int):
        AbstractRustDownsampler._check_valid_n_out(n_out)
        if n_out % 2 != 0:
            raise ValueError("n_out must be even")


class M4Downsampler(AbstractRustDownsampler):
    def __init__(self):
        super().__init__(_tsdownsample_rs.m4)

    @staticmethod
    def _check_valid_n_out(n_out: int):
        AbstractRustDownsampler._check_valid_n_out(n_out)
        if n_out % 4 != 0:
            raise ValueError("n_out must be a multiple of 4")


class LTTBDownsampler(AbstractRustDownsampler):
    def __init__(self):
        super().__init__(_tsdownsample_rs.lttb)


class MinMaxLTTBDownsampler(AbstractRustDownsampler):
    def __init__(self):
        super().__init__(_tsdownsample_rs.minmaxlttb)

    def downsample(
        self, *args, n_out: int, minmax_ratio: int = 30, parallel: bool = False, **_
    ):
        assert minmax_ratio > 0, "minmax_ratio must be greater than 0"
        return super().downsample(
            *args, n_out=n_out, parallel=parallel, ratio=minmax_ratio
        )


# ------------------ EveryNth Downsampler ------------------


class EveryNthDownsampler(AbstractDownsampler):
    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **_
    ) -> np.ndarray:
        if x is not None:
            name = self.__class__.__name__
            warnings.warn(
                f"x is passed to downsample method of {name}, but is not taken "
                "into account by the current implementation of the EveryNth algorithm."
            )
        step = max(1, math.ceil(len(y) / n_out))
        return np.arange(0, len(y), step)
