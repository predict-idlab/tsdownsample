import tsdownsample._rust.tsdownsample_rs as tsdownsample_rs
from .downsampling_interface import RustDownsamplingInterface

# ------------------ Rust Downsamplers ------------------

MinMaxDownsampler = RustDownsamplingInterface(tsdownsample_rs.minmax)
M4Downsampler = RustDownsamplingInterface(tsdownsample_rs.m4)
LTTBDownsampler = RustDownsamplingInterface(tsdownsample_rs.lttb)
MinMaxLTTBDownsampler = RustDownsamplingInterface(tsdownsample_rs.minmax_lttb)

# ------------------ Python Downsamplers ------------------

MeanDownsampler = PythonDownsamplingInterface(np.mean)