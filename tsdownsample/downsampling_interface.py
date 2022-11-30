"""AbstractDownsampler interface-class, subclassed by concrete downsamplers."""

__author__ = "Jeroen Van Der Donckt"

import re
import warnings
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Callable, List, Tuple, Union

import numpy as np


class AbstractDownsampler(ABC):
    """AbstractDownsampler interface-class, subclassed by concrete downsamplers."""

    def __init__(self, name: str, dtype_regex_list: List[str] = None):
        """Initialize the downsampler with a list of regexes to match the data-types to downsample."""
        self.name = name
        self.dtype_regex_list = dtype_regex_list

    def _supports_dtype(self, arr: np.array):
        # base case
        if self.dtype_regex_list is None:
            return

        for dtype_regex_str in self.dtype_regex_list:
            m = re.compile(dtype_regex_str).match(str(arr.dtype))
            if m is not None:  # a match is found
                return
        raise ValueError(
            f"{arr.dtype} doesn't match with any regex in {self.dtype_regex_list}"
        )

    @staticmethod
    def _check_valid_downsample_args(
        *args,
    ) -> Tuple[Union[np.ndarray, None], np.ndarray]:
        if len(args) == 2:
            x, y = args
        elif len(args) == 1:
            x, y = None, args[0]
        else:
            raise ValueError(
                "downsample() takes 1 or 2 positional arguments but "
                f"{len(args)} were given"
            )
        # y must be 1D array
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        # x must be 1D array with same length as y or None
        if x is not None:
            if x.ndim != 1:
                raise ValueError("x must be 1D array")
            if len(x) != len(y):
                raise ValueError("x and y must have the same length")
        return x, y

    @abstractmethod
    def _downsample(
        self,
        x: Union[np.ndarray, None] = None,
        y: Union[np.ndarray, None] = None,
        n_out: int = None,
        **kwargs,
    ) -> np.ndarray:
        """Downsample the data in x and y.

        Returns
        -------
        np.ndarray
            The selected indices.
        """
        raise NotImplementedError

    def downsample(self, *args, n_out: int, **kwargs):  # x and y are optional
        """Downsample y (and x).

        Call signatures::
            downsample([x], y, n_out, **kwargs)


        Parameters
        ----------
        x, y : array-like
            The horizontal / vertical coordinates of the data points.
            *x* values are optional.
            These parameters should be 1D arrays.
            These arguments cannot be passed as keywords.
        n_out : int
            The number of points to keep.
        **kwargs
            Additional keyword arguments are passed to the downsampler.

        Returns
        -------
        np.ndarray
            The selected indices.
        """
        x, y = self._check_valid_downsample_args(*args)
        self._supports_dtype(y)
        if x is not None:
            self._supports_dtype(x)
        return self._downsample(x, y, n_out, **kwargs)

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
    # UINTS
    elif np.issubdtype(y_dtype, np.unsignedinteger):
        if y_dtype == np.uint16:
            return getattr(mod, downsample_func + "_u16")
        elif y_dtype == np.uint32:
            return getattr(mod, downsample_func + "_u32")
        elif y_dtype == np.uint64:
            return getattr(mod, downsample_func + "_u64")
    # INTS (need to be last because uint is subdtype of int)
    elif np.issubdtype(y_dtype, np.integer):
        if y_dtype == np.int16:
            return getattr(mod, downsample_func + "_i16")
        elif y_dtype == np.int32:
            return getattr(mod, downsample_func + "_i32")
        elif y_dtype == np.int64:
            return getattr(mod, downsample_func + "_i64")
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
    # UINTS
    elif np.issubdtype(x_dtype, np.unsignedinteger):
        if x_dtype == np.uint16:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_u16")
        elif x_dtype == np.uint32:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_u32")
        elif x_dtype == np.uint64:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_u64")
    # INTS (need to be last because uint is subdtype of int)
    elif np.issubdtype(x_dtype, np.integer):
        if x_dtype == np.int16:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_i16")
        elif x_dtype == np.int32:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_i32")
        elif x_dtype == np.int64:
            return _switch_mod_with_y(y_dtype, mod, f"{DOWNSAMPLE_F}_i64")
    # BOOLS
    # TODO: support bools
    # elif data_dtype == np.bool:
    # return mod.downsample_bool
    raise ValueError(f"Unsupported data type (for x): {x_dtype}")


class AbstractRustDownsampler(AbstractDownsampler, ABC):
    """RustDownsampler interface-class, subclassed by concrete downsamplers."""

    def __init__(
        self, name: str, resampling_mod: ModuleType, dtype_regex_list: List[str]
    ):
        """Initialize the downsampler with a list of regexes to match the data-types to downsample."""
        super().__init__(name, dtype_regex_list)
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

    def _downsample(
        self,
        x: Union[np.ndarray, None],
        y: Union[np.ndarray, None],
        n_out: int = None,
        parallel: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Downsample the data in x and y."""
        mod = self.mod_single_core
        if parallel:
            if self.mod_multi_core is None:
                warnings.warn(
                    f"No parallel implementation available for {self.name}. "
                    "Falling back to single-core implementation."
                )
        if x is None:
            downsample_f = _switch_mod_with_y(y.dtype, mod)
            return downsample_f(y, n_out, **kwargs)
        downsample_f = _switch_mod_with_x_and_y(x.dtype, y.dtype, mod)
        return downsample_f(x, y, n_out, **kwargs)

    def downsample(
        self,
        *args,  # x and y are optional
        n_out: int,
        parallel: bool = False,
        **kwargs,
    ):
        """Downsample the data in x and y."""
        return super().downsample(*args, n_out=n_out, parallel=parallel, **kwargs)
