"""AbstractDownsampler interface-class, subclassed by concrete downsamplers."""

__author__ = "Jeroen Van Der Donckt"

import re
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from types import ModuleType
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


class AbstractDownsampler(ABC):
    """AbstractDownsampler interface-class, subclassed by concrete downsamplers."""

    def __init__(
        self,
        check_contiguous: bool = True,
        x_dtype_regex_list: Optional[List[str]] = None,
        y_dtype_regex_list: Optional[List[str]] = None,
    ):
        self.check_contiguous = check_contiguous
        self.x_dtype_regex_list = x_dtype_regex_list
        self.y_dtype_regex_list = y_dtype_regex_list

    def _check_contiguous(self, arr: np.ndarray, y: bool = True):
        # necessary for rust downsamplers as they don't support non-contiguous arrays
        # (we call .as_slice().unwrap() on the array) in the lib.rs file
        # which will panic if the array is not contiguous
        if not self.check_contiguous:
            return

        if arr.flags["C_CONTIGUOUS"]:
            return

        raise ValueError(f"{'y' if y else 'x'} array must be contiguous.")

    def _supports_dtype(self, arr: np.ndarray, y: bool = True):
        dtype_regex_list = self.y_dtype_regex_list if y else self.x_dtype_regex_list
        # base case
        if dtype_regex_list is None:
            return

        for dtype_regex_str in dtype_regex_list:
            m = re.compile(dtype_regex_str).match(str(arr.dtype))
            if m is not None:  # a match is found
                return
        raise ValueError(
            f"{arr.dtype} doesn't match with any regex in {dtype_regex_list} "
            f"for the {'y' if y else 'x'}-data"
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

        if x is not None and not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

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

    @staticmethod
    def _check_valid_n_out(n_out: int):
        if n_out <= 0:
            raise ValueError("n_out must be greater than 0")

    @abstractmethod
    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
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
        self._check_valid_n_out(n_out)
        x, y = self._check_valid_downsample_args(*args)
        self._supports_dtype(y, y=True)
        self._check_contiguous(y, y=True)
        if x is not None:
            self._supports_dtype(x, y=False)
            self._check_contiguous(x, y=False)
        return self._downsample(x, y, n_out, **kwargs)


# ------------------- Rust Downsample Interface -------------------
DOWNSAMPLE_F = "downsample"


# the following dtypes are supported by the rust downsamplers (x and y)
_rust_dtypes = [
    "float32",
    "float64",
    "uint16",
    "uint32",
    "uint64",
    "int16",
    "int32",
    "int64",
    "datetime64",
    "timedelta64",
]
# <= 8-bit x-dtypes are not supported as the range of the values is too small to require
# downsampling
_y_rust_dtypes = _rust_dtypes + ["float16", "int8", "uint8", "bool"]


class AbstractRustDownsampler(AbstractDownsampler, ABC):
    """RustDownsampler interface-class, subclassed by concrete downsamplers."""

    def __init__(self):
        super().__init__(True, _rust_dtypes, _y_rust_dtypes)  # same for x and y

    @property
    def _downsample_func_prefix(self) -> str:
        """The prefix of the downsample functions in the rust module."""
        return DOWNSAMPLE_F

    @property
    def rust_mod(self) -> ModuleType:
        """The compiled Rust module for the current downsampler."""
        raise NotImplementedError

    @property
    def mod_single_core(self) -> ModuleType:
        """Get the single-core Rust module.

        Returns
        -------
        ModuleType
            If SIMD compiled module is available, that one is returned. Otherwise, the
            scalar compiled module is returned.
        """
        return self.rust_mod.sequential

    @property
    def mod_multi_core(self) -> Union[ModuleType, None]:
        """Get the multi-core Rust module.

        Returns
        -------
        ModuleType or None
            If SIMD parallel compiled module is available, that one is returned.
            Otherwise, the scalar parallel compiled module is returned.
            If no parallel compiled module is available, None is returned.
        """
        if hasattr(self.rust_mod, "parallel"):
            # use SIMD implementation if available
            return self.rust_mod.parallel
        return None  # no parallel compiled module available

    @staticmethod
    def _view_x(x: np.ndarray) -> np.ndarray:
        """View the x-data as different dtype (if necessary)."""
        if np.issubdtype(x.dtype, np.datetime64):
            # datetime64 is viewed as int64
            return x.view(dtype=np.int64)
        elif np.issubdtype(x.dtype, np.timedelta64):
            # timedelta64 is viewed as int64
            return x.view(dtype=np.int64)
        return x

    @staticmethod
    def _view_y(y: np.ndarray) -> np.ndarray:
        """View the y-data as different dtype (if necessary)."""
        if y.dtype == "bool":
            # bool is viewed as int8
            return y.view(dtype=np.int8)
        elif np.issubdtype(y.dtype, np.datetime64):
            # datetime64 is viewed as int64
            return y.view(dtype=np.int64)
        elif np.issubdtype(y.dtype, np.timedelta64):
            # timedelta64 is viewed as int64
            return y.view(dtype=np.int64)
        return y

    def _switch_mod_with_y(
        self, y_dtype: np.dtype, mod: ModuleType, downsample_func: Optional[str] = None
    ) -> Callable:
        """Select the appropriate function from the rust module for the y-data.

        Assumes equal binning (when no data for x is passed -> only this function is
        executed).
        Equidistant binning is  utilized when a `downsample_func` is passed from the
        `_switch_mod_with_x_and_y` method (since the x-data is considered in the
        downsampling).

        Parameters
        ----------
        y_dtype : np.dtype
            The dtype of the y-data
        mod : ModuleType
            The module to select the appropriate function from
        downsample_func : str, optional
            The name of the function to use, by default DOWNSAMPLE_FUNC.
            This argument is passed from the `_switch_mod_with_x_and_y` method when
            the x-data is considered in the downsampling.
        """
        if downsample_func is None:
            downsample_func = self._downsample_func_prefix
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
            if y_dtype == np.uint8:
                return getattr(mod, downsample_func + "_u8")
            elif y_dtype == np.uint16:
                return getattr(mod, downsample_func + "_u16")
            elif y_dtype == np.uint32:
                return getattr(mod, downsample_func + "_u32")
            elif y_dtype == np.uint64:
                return getattr(mod, downsample_func + "_u64")
        # INTS (need to be last because uint is subdtype of int)
        elif np.issubdtype(y_dtype, np.integer):
            if y_dtype == np.int8:
                return getattr(mod, downsample_func + "_i8")
            elif y_dtype == np.int16:
                return getattr(mod, downsample_func + "_i16")
            elif y_dtype == np.int32:
                return getattr(mod, downsample_func + "_i32")
            elif y_dtype == np.int64:
                return getattr(mod, downsample_func + "_i64")
        # DATETIME -> i64 (datetime64 is viewed as int64)
        # TIMEDELTA -> i64 (timedelta64 is viewed as int64)
        # BOOLS -> int8 (bool is viewed as int8)
        raise ValueError(f"Unsupported data type (for y): {y_dtype}")

    def _switch_mod_with_x_and_y(
        self,  # necessary to access the class its _switch_mod_with_y method
        x_dtype: np.dtype,
        y_dtype: np.dtype,
        mod: ModuleType,
        downsample_func: Optional[str] = None,
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
        downsample_func : str, optional
            The name of the function to use, by default DOWNSAMPLE_FUNC.
        """
        if downsample_func is None:
            downsample_func = self._downsample_func_prefix
        # FLOATS
        if np.issubdtype(x_dtype, np.floating):
            if x_dtype == np.float16:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_f16")
            elif x_dtype == np.float32:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_f32")
            elif x_dtype == np.float64:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_f64")
        # UINTS
        elif np.issubdtype(x_dtype, np.unsignedinteger):
            if x_dtype == np.uint16:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_u16")
            elif x_dtype == np.uint32:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_u32")
            elif x_dtype == np.uint64:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_u64")
        # INTS (need to be last because uint is subdtype of int)
        elif np.issubdtype(x_dtype, np.integer):
            if x_dtype == np.int16:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_i16")
            elif x_dtype == np.int32:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_i32")
            elif x_dtype == np.int64:
                return self._switch_mod_with_y(y_dtype, mod, f"{downsample_func}_i64")
        # DATETIME -> i64 (datetime64 is viewed as int64)
        # TIMEDELTA -> i64 (timedelta64 is viewed as int64)
        raise ValueError(f"Unsupported data type (for x): {x_dtype}")

    def _downsample(
        self,
        x: Union[np.ndarray, None],
        y: np.ndarray,
        n_out: int,
        parallel: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Downsample the data in x and y."""
        mod = self.mod_single_core
        if parallel:
            if self.mod_multi_core is None:
                name = self.__class__.__name__
                warnings.warn(
                    f"No parallel implementation available for {name}. "
                    "Falling back to single-core implementation."
                )
            else:
                mod = self.mod_multi_core
        ## Viewing the y-data as different dtype (if necessary)
        y = self._view_y(y)
        ## Viewing the x-data as different dtype (if necessary)
        if x is None:
            downsample_f = self._switch_mod_with_y(y.dtype, mod)
            return downsample_f(y, n_out, **kwargs)
        x = self._view_x(x)
        ## Getting the appropriate downsample function
        downsample_f = self._switch_mod_with_x_and_y(x.dtype, y.dtype, mod)
        return downsample_f(x, y, n_out, **kwargs)

    def downsample(self, *args, n_out: int, parallel: bool = False, **kwargs):
        """Downsample the data in x and y.

        The x and y arguments are positional-only arguments. If only one argument is
        passed, it is considered to be the y-data. If two arguments are passed, the
        first argument is considered to be the x-data and the second argument is
        considered to be the y-data.
        """
        return super().downsample(*args, n_out=n_out, parallel=parallel, **kwargs)

    def __deepcopy__(self, memo):
        """Deepcopy the object."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k.endswith("_mod") or k.startswith("mod_"):
                # Don't (deep)copy the compiled modules
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


NAN_DOWNSAMPLE_F = "downsample_nan"


class AbstractRustNaNDownsampler(AbstractRustDownsampler, ABC):
    """RustNaNDownsampler interface-class, subclassed by concrete downsamplers."""

    @property
    def _downsample_func_prefix(self) -> str:
        """The prefix of the downsample functions in the rust module."""
        return NAN_DOWNSAMPLE_F

    def _switch_mod_with_y(
        self, y_dtype: np.dtype, mod: ModuleType, downsample_func: Optional[str] = None
    ) -> Callable:
        """Select the appropriate function from the rust module for the y-data.

        Assumes equal binning (when no data for x is passed -> only this function is
        executed).
        Equidistant binning is  utilized when a `downsample_func` is passed from the
        `_switch_mod_with_x_and_y` method (since the x-data is considered in the
        downsampling).

        Parameters
        ----------
        y_dtype : np.dtype
            The dtype of the y-data
        mod : ModuleType
            The module to select the appropriate function from
        downsample_func : str, optional
            The name of the function to use, by default NAN_DOWNSAMPLE_F.
            This argument is passed from the `_switch_mod_with_x_and_y` method when
            the x-data is considered in the downsampling.
        """
        if downsample_func is None:
            downsample_func = self._downsample_func_prefix
        if not np.issubdtype(y_dtype, np.floating):
            # When y is not a float, we need to remove the _nan suffix to use the
            # regular downsample function as the _nan suffix is only used for floats.
            # (Note that NaNs only exist for floats)
            downsample_func = downsample_func.replace("_nan", "")
        return super()._switch_mod_with_y(y_dtype, mod, downsample_func)
