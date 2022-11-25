import warnings
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Callable

import numpy as np
import pandas as pd


class DownsampleInterface(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def _construct_output_series(s: pd.Series, idxs: np.ndarray) -> pd.Series:
        return s.iloc[idxs]

    # def _supports_dtype(self, s: pd.Series):
    #     # base case
    #     if self.dtype_regex_list is None:
    #         return

    #     for dtype_regex_str in self.dtype_regex_list:
    #         m = re.compile(dtype_regex_str).match(str(s.dtype))
    #         if m is not None:  # a match is found
    #             return
    #     raise ValueError(
    #         f"{s.dtype} doesn't match with any regex in {self.dtype_regex_list}"
    #     )

    @abstractmethod
    def downsample(self, s: pd.Series, n_out: int, parallel: bool = False) -> pd.Series:
        """Downsample a pandas series to n_out samples.

        Parameters
        ----------
        s : pd.Series
            The series to downsample.
        n_out : int
            The number of samples to downsample to.
        parallel : bool, optional
            Whether to use parallel processing (if available), by default False.

        Returns
        -------
        pd.Series
            The downsampled series.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name}"


# ------------------- Rust Downsample Interface -------------------
DOWNSAMPLE_F = "downsample"


def _switch_mod_with_y(
    y_dtype: np.dtype, mod: ModuleType, downsample_func: str = DOWNSAMPLE_F
) -> Callable:
    """The x-data is not considered in the downsampling

    Assumes equal binning.

    Parameters
    ----------
    y_dtype : np.dtype
        The dtype of the y-data
    mod : ModuleType
        The module to select the appropriate function from
    downsample_func : str, optional
        The name of the function to use, by default DOWNSAMPLE_FUNC.
    """
    # FLOATS
    if np.issubdtype(y_dtype, np.floating):
        if y_dtype == np.float16:
            return getattr(mod, downsample_func + "_f16")
        elif y_dtype == np.float32:
            return getattr(mod, downsample_func + "_f32")
        elif y_dtype == np.float64:
            return getattr(mod, downsample_func + "_f64")
    # INTS
    elif np.issubdtype(y_dtype, np.integer):
        if y_dtype == np.int16:
            return getattr(mod, downsample_func + "_i16")
        elif y_dtype == np.int32:
            return getattr(mod, downsample_func + "_i32")
        elif y_dtype == np.int64:
            return getattr(mod, downsample_func + "_i64")
    # UINTS
    elif np.issubdtype(y_dtype, np.unsignedinteger):
        if y_dtype == np.uint16:
            return getattr(mod, downsample_func + "_u16")
        elif y_dtype == np.uint32:
            return getattr(mod, downsample_func + "_u32")
        elif y_dtype == np.uint64:
            return getattr(mod, downsample_func + "_u64")
    # BOOLS
    # TODO: support bools
    # elif data_dtype == np.bool:
    # return mod.downsample_bool
    raise ValueError(f"Unsupported data type (for y): {y_dtype}")


def _switch_mod_with_x_and_y(
    x_dtype: np.dtype, y_dtype: np.dtype, mod: ModuleType
) -> Callable:
    """The x-data is considered in the downsampling

    Assumes equal binning.

    Parameters
    ----------
    x_dtype : np.dtype
        The dtype of the x-data
    y_dtype : np.dtype
        The dtype of the y-data
    mod : ModuleType
        The module to select the appropriate function from
    """
    # FLOATS
    if np.issubdtype(x_dtype, np.floating):
        if x_dtype == np.float16:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_f16")
        elif x_dtype == np.float32:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_f32")
        elif x_dtype == np.float64:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_f64")
    # INTS
    elif np.issubdtype(x_dtype, np.integer):
        if x_dtype == np.int16:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_i16")
        elif x_dtype == np.int32:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_i32")
        elif x_dtype == np.int64:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_i64")
    # UINTS
    elif np.issubdtype(x_dtype, np.unsignedinteger):
        if x_dtype == np.uint16:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_u16")
        elif x_dtype == np.uint32:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_u32")
        elif x_dtype == np.uint64:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_u64")
    # BOOLS
    # TODO: support bools
    # elif data_dtype == np.bool:
    # return mod.downsample_bool
    raise ValueError(f"Unsupported data type (for x): {x_dtype}")


class RustDownsamplingInterface(DownsampleInterface):
    def __init__(self, name: str, resampling_mod: ModuleType) -> None:
        super().__init__(name + " [tsdownsample_rs]")
        self.rust_mod = resampling_mod

        # Store the single core sub module
        self.mod_single_core = self.rust_mod.scalar
        if hasattr(self.rust_mod, "simd"):
            # use SIMD implementation if available
            self.mod_single_core = self.rust_mod.simd

        # Store the multi-core sub module (if present)
        self.mod_multi_core = None  # no multi-core implementation (default)
        if hasattr(self.rust_mod, "simd_parallel"):
            # use SIMD implementation if available
            self.mod_multi_core = self.rust_mod.simd_parallel
        elif hasattr(self.rust_mod, "scalar_parallel"):
            # use scalar implementation if available (when no SIMD available)
            self.mod_multi_core = self.rust_mod.scalar_parallel

    def _downsample_without_x(self, s: pd.Series, n_out: int) -> pd.Series:
        downsample_method = _switch_mod_with_y(s.dtype, self.mod_single_core)
        idxs = downsample_method(s.values, n_out)
        return self._construct_output_series(s, idxs)

    def _downsample_with_x(self, s: pd.Series, n_out: int) -> pd.Series:
        downsample_method = _switch_mod_with_x_and_y(
            s.index.dtype, s.dtype, self.mod_single_core
        )
        idxs = downsample_method(s.index.values, s.values, n_out)
        return self._construct_output_series(s, idxs)

    def _downsample_without_x_parallel(self, s: pd.Series, n_out: int) -> pd.Series:
        if self.mod_multi_core is not None:
            downsample_method = _switch_mod_with_y(s.dtype, self.mod_multi_core)
        else:
            warnings.warn(
                "No multi-core implementation available. "
                "Falling back to single core implementation."
            )
            downsample_method = _switch_mod_with_y(s.dtype, self.mod_single_core)
        idxs = downsample_method(s.values, n_out)
        return self._construct_output_series(s, idxs)

    def _downsample_with_x_parallel(self, s: pd.Series, n_out: int) -> pd.Series:
        if self.mod_multi_core is not None:
            downsample_method = _switch_mod_with_x_and_y(
                s.index.dtype, s.dtype, self.mod_multi_core
            )
        else:
            warnings.warn(
                "No multi-core implementation available. "
                "Falling back to single core implementation."
            )
            downsample_method = _switch_mod_with_x_and_y(
                s.index.dtype, s.dtype, self.mod_single_core
            )
        idxs = downsample_method(s.index.values, s.values, n_out)
        return self._construct_output_series(s, idxs)

    def downsample(self, s: pd.Series, n_out: int, parallel: bool = False) -> pd.Series:
        fixed_sr = False
        if isinstance(s.index, pd.RangeIndex) or s.index.freq is not None:
            fixed_sr = True
        if fixed_sr:  # TODO: or the other way around??
            if parallel:
                return self._downsample_without_x_parallel(s, n_out)
            else:
                return self._downsample_without_x(s, n_out)
        else:
            if parallel:
                return self._downsample_with_x_parallel(s, n_out)
            else:
                return self._downsample_with_x(s, n_out)


# ------------------ Numpy Downsample Interface ------------------


class FuncDownsamplingInterface(DownsampleInterface):
    def __init__(self, name: str, downsample_func: Callable) -> None:
        super().__init__("[Func]_" + name)
        self.downsample_func = downsample_func

    def downsample(self, s: pd.Series, n_out: int, parallel: bool = False) -> pd.Series:
        if isinstance(s.index, pd.DatetimeIndex):
            t_start, t_end = s.index[:: len(s) - 1]
            rate = (t_end - t_start) / n_out
            return s.resample(rate).apply(self.downsample_func).dropna()

        # no time index -> use the every nth heuristic
        group_size = max(1, np.ceil(len(s) / n_out))
        s_out = (
            s.groupby(
                # create an array of [0, 0, 0, ...., n_out, n_out]
                # where each value is repeated based $len(s)/n_out$ times
                by=np.repeat(np.arange(n_out), group_size)[: len(s)]
            )
            .agg(self.downsample_func)
            .dropna()
        )
        # Create an index-estimation for real-time data
        # Add one to the index so it's pointed at the end of the window
        # Note: this can be adjusted to .5 to center the data
        # Multiply it with the group size to get the real index-position
        # TODO: add option to select start / middle / end as index
        idx_locs = (np.arange(len(s_out)) + 1) * group_size
        idx_locs[-1] = len(s) - 1
        return pd.Series(
            index=s.iloc[idx_locs.astype(s.index.dtype)].index.astype(s.index.dtype),
            data=s_out.values,
            name=str(s.name),
            copy=False,
        )
