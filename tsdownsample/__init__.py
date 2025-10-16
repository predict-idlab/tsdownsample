"""tsdownsample: high performance downsampling of time series data for visualization."""

from .downsamplers import (
    EveryNthDownsampler,
    FPCSDownsampler,
    LTTBDownsampler,
    M4Downsampler,
    MinMaxDownsampler,
    MinMaxLTTBDownsampler,
    NaNFPCSDownsampler,
    NaNM4Downsampler,
    NaNMinMaxDownsampler,
    NaNMinMaxLTTBDownsampler,
)

__version__ = "0.1.4.1"
__author__ = "Jeroen Van Der Donckt"

__all__ = [
    "EveryNthDownsampler",
    "FPCSDownsampler",
    "MinMaxDownsampler",
    "M4Downsampler",
    "LTTBDownsampler",
    "MinMaxLTTBDownsampler",
    "NaNFPCSDownsampler",
    "NaNMinMaxDownsampler",
    "NaNM4Downsampler",
    "NaNMinMaxLTTBDownsampler",
]
