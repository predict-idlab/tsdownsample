import warnings
import numpy as np

from typing import Union
from .downsampling_interface import DownsampleInterface

# ------------------ Rust Downsamplers ------------------
from tsdownsample._rust import _tsdownsample_rs  # type: ignore[attr-defined]

from .downsampling_interface import AbstractRustDownsampler

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
                f"x is passed to downsample method of {self.name}, but is not taken "/
                f"into account by the current implementation of  {self.name} algorithm."
            )
        super()._downsample(x, *args, **kwargs)

class M4Downsampler(AbstractRustDownsampler):

    def __init__(self):
        super().__init__("M4", _tsdownsample_rs.m4, rust_dtypes)

    def _downsample(self, x: Union[np.ndarray, None], *args, **kwargs):
        if x is not None:
            warnings.warn(
                f"x is passed to downsample method of {self.name}, but is not taken "/
                f"into account by the current implementation of  {self.name} algorithm."
            )
        super()._downsample(x, *args, **kwargs)

class LTTBDownsampler(AbstractRustDownsampler):

    def __init__(self):
        super().__init__("LTTB", _tsdownsample_rs.lttb, rust_dtypes)

class MinMaxLTTBDownsampler(AbstractRustDownsampler):

    def __init__(self):
        super().__init__("MinMaxLTTB", _tsdownsample_rs.minmaxlttb, rust_dtypes)


# ------------------ EveryNth Downsampler ------------------
import math

class EveryNthDownsampler(DownsampleInterface):
    def __init__(self) -> None:
        super().__init__("EveryNth")

    def _downsample(
        self,
        _: Union[np.ndarray, None],  # x is not used
        y: np.ndarray,
        n_out: int,
    ) -> np.ndarray:
        step = max(1, math.ceil(len(y) / n_out))
        return np.arange(0, len(y), step)

# ------------------ Function Downsampler ------------------

class FuncDownsampler(DownsampleInterface):

    pass
