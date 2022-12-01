import warnings
from typing import Union

import numpy as np

# ------------------ Rust Downsamplers ------------------
from tsdownsample._rust import _tsdownsample_rs  # type: ignore[attr-defined]

from .downsampling_interface import AbstractDownsampler, AbstractRustDownsampler

rust_dtypes = [
    "float16",
    "float32",
    "float64",
    "uint16",
    "uint32",
    "uint64",
    "int16",
    "int32",
    "int64",
]

# MinMaxDownsampler = RustDownsampler("MinMax", _tsdownsample_rs.minmax, rust_dtypes)
# M4Downsampler = RustDownsampler("M4", _tsdownsample_rs.m4, rust_dtypes)
# LTTBDownsampler = RustDownsampler("LTTB", _tsdownsample_rs.lttb, rust_dtypes)
# MinMaxLTTBDownsampler = RustDownsampler(
#     "MinMaxLTTB", _tsdownsample_rs.minmaxlttb, rust_dtypes
# )


class MinMaxDownsampler(AbstractRustDownsampler):
    def __init__(self) -> None:
        super().__init__("MinMax", _tsdownsample_rs.minmax, rust_dtypes)

    def _downsample(self, x: Union[np.ndarray, None], *args, **kwargs):
        if x is not None:
            warnings.warn(
                f"x is passed to downsample method of {self.name}, but is not taken "
                f"into account by the current implementation of  {self.name} algorithm."
            )
        return super()._downsample(None, *args, **kwargs)


class M4Downsampler(AbstractRustDownsampler):
    def __init__(self):
        super().__init__("M4", _tsdownsample_rs.m4, rust_dtypes)

    def _downsample(self, x: Union[np.ndarray, None], *args, **kwargs):
        if x is not None:
            warnings.warn(
                f"x is passed to downsample method of {self.name}, but is not taken "
                f"into account by the current implementation of  {self.name} algorithm."
            )
        return super()._downsample(None, *args, **kwargs)


class LTTBDownsampler(AbstractRustDownsampler):
    def __init__(self):
        super().__init__("LTTB", _tsdownsample_rs.lttb, rust_dtypes)


class MinMaxLTTBDownsampler(AbstractRustDownsampler):
    def __init__(self):
        super().__init__("MinMaxLTTB", _tsdownsample_rs.minmaxlttb, rust_dtypes)

    def downsample(
        self, *args, n_out: int, minmax_ratio: int = 30, parallel: bool = False
    ):
        assert minmax_ratio > 0, "minmax_ratio must be greater than 0"
        return super().downsample(
            *args, n_out=n_out, parallel=parallel, ratio=minmax_ratio
        )


# ------------------ EveryNth Downsampler ------------------
import math


class EveryNthDownsampler(AbstractDownsampler):
    def __init__(self) -> None:
        super().__init__("EveryNth")

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int,  **kwargs
    ) -> np.ndarray:
        if x is not None:
            warnings.warn(
                f"x is passed to downsample method of {self.name}, but is not taken "
                f"into account by the current implementation of  {self.name} algorithm."
            )
        step = max(1, math.ceil(len(y) / n_out))
        return np.arange(0, len(y), step)


# ------------------ Function Downsampler ------------------


class FuncDownsampler(AbstractDownsampler):

    pass
