from types import ModuleType
from typing import Callable
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

DOWNSAMPLE_F = 'downsample'

class DownsampleInterface(ABC):

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _construct_output_series(s: pd.Series, idxs: np.ndarray) -> pd.Series:
        return s.iloc[idxs]

    def _supports_dtype(self, s: pd.Series):
        # base case
        if self.dtype_regex_list is None:
            return

        for dtype_regex_str in self.dtype_regex_list:
            m = re.compile(dtype_regex_str).match(str(s.dtype))
            if m is not None:  # a match is found
                return
        raise ValueError(
            f"{s.dtype} doesn't match with any regex in {self.dtype_regex_list}"
        )

    def downsample(self, s: pd.Series, n_out: int, parallel: bool = False) -> pd.Series
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

# ------------------- Rust Downsample Interface -------------------

def _switch_mod_with_y(y_dtype: np.dtype, mod: ModuleType, downsample_func: str = DOWNSAMPLE_F):
    """The x-data is not considered in the downsampling
    
    Assumes equal binning.

    Parameters
    ----------
    y_dtype : np.dtype
        The dtype of the y-data
    mod : Module
        The module to select the appropriate function from
    downsample_func : str, optional
        The name of the function to use, by default DOWNSAMPLE_FUNC.
    """
    # FLOATS
    if np.issubdtype(y_dtype, np.floating):
        if y.dtype == np.float16:
            return getattr(mod, downsample_func + '_f16')
        elif y_dtype == np.float32:
            return getattr(mod, downsample_func + '_f32')
        elif y_dtype == np.float64:
            return getattr(mod, downsample_func + '_f64')
    # INTS
    elif np.issubdtype(y_dtype, np.integer):
        if y_dtype == np.int16:
            return getattr(mod, downsample_func + '_i16')
        elif y_dtype == np.int32:
            return getattr(mod, downsample_func + '_i32')
        elif y_dtype == np.int64:
            return getattr(mod, downsample_func + '_i64')
    # UINTS
    elif np.issubdtype(y_dtype, np.unsignedinteger):
        if y_dtype == np.uint16:
            return getattr(mod, downsample_func + '_u16')
        elif y_dtype == np.uint32:
            return getattr(mod, downsample_func + '_u32')
        elif y_dtype == np.uint64:
            return getattr(mod, downsample_func + '_u64')
    # BOOLS
    # TODO: support bools
    # elif data_dtype == np.bool:
        # return mod.downsample_bool
    raise ValueError(f"Unsupported data type (for y): {y_dtype}")

def _switch_mod_with_x_and_y(x_dtype: np.dtype, y_dtype: np.dtype, mod: ModuleType):
    """The x-data is considered in the downsampling
    
    Assumes equal binning.

    Parameters
    ----------
    x_dtype : np.dtype
        The dtype of the x-data
    y_dtype : np.dtype
        The dtype of the y-data
    mod : Module
        The module to select the appropriate function from
    """
    # FLOATS
    if np.issubdtype(x_dtype, np.floating):
        if x_dtype == np.float16:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_f16')
        elif x_dtype == np.float32:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_f32')
        elif x_dtype == np.float64:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_f64')
    # INTS
    elif np.issubdtype(x_dtype, np.integer):
        if x_dtype == np.int16:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_i16')
        elif x_dtype == np.int32:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_i32')
        elif x_dtype == np.int64:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_i64')
    # UINTS
    elif np.issubdtype(x_dtype, np.unsignedinteger):
        if x_dtype == np.uint16:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_u16')
        elif x_dtype == np.uint32:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_u32')
        elif x_dtype == np.uint64:
            return switch_mod_with_y(y_dtype, mod, f'{DOWNSAMPLE_F}_u64')
    # BOOLS
    # TODO: support bools
    # elif data_dtype == np.bool:
        # return mod.downsample_bool
    raise ValueError(f"Unsupported data type (for x): {x_dtype}")

class RustDownsamplingInterface(DownsampleInterface):

    def __init__(self, resampling_mod: Module) -> None:
        self._mod = resampling_mod
        if hasattr(self.mod, 'simd'):
            self.mod_single_core = self._mod.simd
            self.mod_multi_core = self._mod.simd_parallel
        else:
            self.mod_single_core = self._mod.scalar
            self.mod_multi_core = self._mod.scalar_parallel
        
    def _downsample_without_x(self, s: pd.Series, n_out: int) -> pd.Series:
        downsample_method = _switch_mod_with_y(s.dtype, self.mod_single_core)
        idxs = downsample_method(s.values, n_out)
        return self._construct_output_series(s, idxs)
    
    def _downsample_with_x(self, s: pd.Series, n_out: int) -> pd.Series:
        downsample_method = _switch_mod_with_x_and_y(s.index.dtype, s.dtype, self.mod_single_core)
        idxs = downsample_method(s.index.values, s.values, n_out)
        return self._construct_output_series(s, idxs)

    def _downsample_without_x_parallel(self, s: pd.Series, n_out: int) -> pd.Series:
        downsample_method = _switch_mod_with_y(s.dtype, self.mod_multi_core)
        idxs = downsample_method(s.values, n_out)
        return self._construct_output_series(s, idxs)
    
    def _downsample_with_x_parallel(self, s: pd.Series, n_out: int) -> pd.Series:
        downsample_method = _switch_mod_with_x_and_y(s.index.dtype, s.dtype, self.mod_multi_core)
        idxs = downsample_method(s.index.values, s.values, n_out)
        return self._construct_output_series(s, idxs)

    def downsample(self, s: pd.Series, n_out: int, parallel: bool = False) -> pd.Series:
        if s.index.freq is None:  # TODO: or the other way around??
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

class NumpyDownsamplingInterface():

    def __init__(self, resampling_func: Callable) -> None:
        self._func = resampling_func

    def downsample(self, s: pd.Series, n_out: int, parallel: bool = False) -> pd.Series:
        