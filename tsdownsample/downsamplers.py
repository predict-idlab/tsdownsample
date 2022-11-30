# ------------------ Rust Downsamplers ------------------
from tsdownsample._rust import _tsdownsample_rs  # type: ignore[attr-defined]

from .downsampling_interface import RustDownsampler

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

MinMaxDownsampler = RustDownsampler("MinMax", _tsdownsample_rs.minmax, rust_dtypes)
M4Downsampler = RustDownsampler("M4", _tsdownsample_rs.m4, rust_dtypes)
LTTBDownsampler = RustDownsampler("LTTB", _tsdownsample_rs.lttb, rust_dtypes)
MinMaxLTTBDownsampler = RustDownsampler(
    "MinMaxLTTB", _tsdownsample_rs.minmaxlttb, rust_dtypes
)

# ------------------ EveryNth Downsampler ------------------
import math
import pandas as pd

from .downsampling_interface import DownsampleInterface


class _EveryNthDownsampler(DownsampleInterface):
    def __init__(self) -> None:
        super().__init__("EveryNth")

    def _downsample(
        


EveryNthDownsampler = _EveryNthDownsampler()
