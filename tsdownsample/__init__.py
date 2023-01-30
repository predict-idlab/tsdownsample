"""tsdownsample: high performance downsampling of time series data for visualization."""

from .downsamplers import (
    EveryNthDownsampler,
    LTTBDownsampler,
    M4Downsampler,
    MinMaxDownsampler,
    MinMaxLTTBDownsampler,
)

__version__ = "0.1.1"
__author__ = "Jeroen Van Der Donckt"

__all__ = [
    "EveryNthDownsampler",
    "MinMaxDownsampler",
    "M4Downsampler",
    "LTTBDownsampler",
    "MinMaxLTTBDownsampler",
]
