from abc import ABC
from enum import Enum
from typing import NamedTuple, Union

import numpy as np

from ..downsampling_interface import AbstractDownsampler


def _get_bin_idxs(x: np.ndarray, nb_bins: int) -> np.ndarray:
    """Get the equidistant indices of the bins to use for the aggregation.

    Parameters
    ----------
    x : np.ndarray
        The x values of the input data.
    nb_bins : int
        The number of bins.

    Returns
    -------
    np.ndarray
        The indices of the bins to use for the aggregation.
    """
    # Thanks to the `linspace` the data is evenly distributed over the index-range
    # The searchsorted function returns the index positions
    bins = np.searchsorted(x, np.linspace(x[0], x[-1], nb_bins + 1), side="right")
    bins[0] = 0
    bins[-1] = len(x)
    return np.array(bins)


class LTTB_py(AbstractDownsampler):
    @staticmethod
    def _argmax_area(prev_x, prev_y, avg_next_x, avg_next_y, x_bucket, y_bucket) -> int:
        """Vectorized triangular area argmax computation.

        Parameters
        ----------
        prev_x : float
            The previous selected point is x value.
        prev_y : float
            The previous selected point its y value.
        avg_next_x : float
            The x mean of the next bucket
        avg_next_y : float
            The y mean of the next bucket
        x_bucket : np.ndarray
            All x values in the bucket
        y_bucket : np.ndarray
            All y values in the bucket

        Returns
        -------
        int
            The index of the point with the largest triangular area.
        """
        return np.abs(
            x_bucket * (prev_y - avg_next_y)
            + y_bucket * (avg_next_x - prev_x)
            + (prev_x * avg_next_y - avg_next_x * prev_y)
        ).argmax()

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # Bucket size. Leave room for start and end data points
        block_size = (y.shape[0] - 2) / (n_out - 2)
        # Note this 'astype' cast must take place after array creation (and not with the
        # aranage() its dtype argument) or it will cast the `block_size` step to an int
        # before the arange array creation
        offset = np.arange(start=1, stop=y.shape[0], step=block_size).astype(np.int64)

        # Construct the output array
        sampled_x = np.empty(n_out, dtype="int64")
        # Add the first point
        sampled_x[0] = 0

        # Convert x & y to int if it is boolean
        if x.dtype == np.bool_:
            x = x.astype(np.int8)
        if y.dtype == np.bool_:
            y = y.astype(np.int8)

        a = 0
        for i in range(n_out - 3):
            a = (
                LTTB_py._argmax_area(
                    prev_x=x[a],
                    prev_y=y[a],
                    # NOTE: In a 100% correct implementation of LTTB the next x average
                    # should be implemented as the following:
                    # avg_next_x=np.mean(x[offset[i + 1] : offset[i + 2]]),
                    # To improve performance we use the following approximation
                    # which is the average of the first and last point of the next bucket
                    # NOTE: this is not as accurate when x is not sampled equidistant
                    # or when the buckets do not contain tht much data points, but it:
                    # (1) aligns with visual perception (visual middle)
                    # (2) is much faster
                    # (3) is how the LTTB rust implementation works
                    avg_next_x=(x[offset[i + 1]] + x[offset[i + 2] - 1]) / 2.0,
                    avg_next_y=y[offset[i + 1] : offset[i + 2]].mean(),
                    x_bucket=x[offset[i] : offset[i + 1]],
                    y_bucket=y[offset[i] : offset[i + 1]],
                )
                + offset[i]
            )
            sampled_x[i + 1] = a

        # ------------ EDGE CASE ------------
        # next-average of last bucket = last point
        sampled_x[-2] = (
            LTTB_py._argmax_area(
                prev_x=x[a],
                prev_y=y[a],
                avg_next_x=x[-1],  # last point
                avg_next_y=y[-1],
                x_bucket=x[offset[-2] : offset[-1]],
                y_bucket=y[offset[-2] : offset[-1]],
            )
            + offset[-2]
        )
        # Always include the last point
        sampled_x[-1] = x.shape[0] - 1
        return sampled_x


class MinMax_py(AbstractDownsampler):
    """Aggregation method which performs binned min-max aggregation over fully
    overlapping windows.
    """

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 2 == 0, "n_out must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 2)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(lower + np.nanargmin(y_slice))
            rel_idxs.append(lower + np.nanargmax(y_slice))
        return np.unique(rel_idxs)


class NaNMinMax_py(AbstractDownsampler):
    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 2 == 0, "n_out must be a multiple of 2"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 2)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue
            # calculate the argmin(slice) & argmax(slice)
            rel_idxs.append(lower + np.argmin(y_slice))
            rel_idxs.append(lower + np.argmax(y_slice))
        return np.array(sorted(rel_idxs))


class M4_py(AbstractDownsampler):
    """Aggregation method which selects the 4 M-s, i.e y-argmin, y-argmax, x-argmin, and
    x-argmax per bin.

    .. note::
        When `n_out` is 4 * the canvas its pixel widht it should create a pixel-perfect
        visualization w.r.t. the raw data.

    """

    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 4 == 0, "n_out must be a multiple of 4"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 4)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue

            # calculate the min(idx), argmin(slice), argmax(slice), max(idx)
            rel_idxs.append(lower)
            rel_idxs.append(lower + np.nanargmin(y_slice))
            rel_idxs.append(lower + np.nanargmax(y_slice))
            rel_idxs.append(upper - 1)

        # NOTE: we do not use the np.unique so that all indices are retained
        return np.array(sorted(rel_idxs))


class NaNM4_py(AbstractDownsampler):
    @staticmethod
    def _check_valid_n_out(n_out: int):
        assert n_out % 4 == 0, "n_out must be a multiple of 4"

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        """TODO complete docs"""
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        xdt = x.dtype
        if np.issubdtype(xdt, np.datetime64) or np.issubdtype(xdt, np.timedelta64):
            x = x.view(np.int64)

        bins = _get_bin_idxs(x, n_out // 4)

        rel_idxs = []
        for lower, upper in zip(bins, bins[1:]):
            y_slice = y[lower:upper]
            if not len(y_slice):
                continue

            # calculate the min(idx), argmin(slice), argmax(slice), max(idx)
            rel_idxs.append(lower)
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())
            rel_idxs.append(upper - 1)

        # NOTE: we do not use the np.unique so that all indices are retained
        return np.array(sorted(rel_idxs))


class _MinMaxLTTB_py(AbstractDownsampler, ABC):
    def __init__(
        self, check_contiguous=True, x_dtype_regex_list=None, y_dtype_regex_list=None
    ):
        super().__init__(check_contiguous, x_dtype_regex_list, y_dtype_regex_list)
        self.minmax_downsampler: AbstractDownsampler = None
        self.lttb_downsampler: AbstractDownsampler = None

    def _downsample(self, x, y, n_out, **kwargs):
        minmax_ratio = kwargs.get("minmax_ratio", 4)
        kwargs.pop("minmax_ratio", None)  # remove the minmax_ratio from kwargs

        # Is fine for this implementation as this is only used for testing
        if x is None:
            x = np.arange(y.shape[0])

        n_1 = len(x) - 1
        idxs = self.minmax_downsampler.downsample(
            x[1:n_1], y[1:n_1], n_out=n_out * minmax_ratio, **kwargs
        )
        idxs += 1
        idxs = np.concat(([0], idxs, [len(y) - 1])).ravel()
        return idxs[
            self.lttb_downsampler.downsample(x[idxs], y[idxs], n_out=n_out, **kwargs)
        ]


class MinMaxLTTB_py(_MinMaxLTTB_py):
    def __init__(
        self, check_contiguous=True, x_dtype_regex_list=None, y_dtype_regex_list=None
    ):
        super().__init__(check_contiguous, x_dtype_regex_list, y_dtype_regex_list)
        self.minmax_downsampler = MinMax_py()
        self.lttb_downsampler = LTTB_py()


class NaNMinMaxLTTB_py(_MinMaxLTTB_py):
    def __init__(
        self, check_contiguous=True, x_dtype_regex_list=None, y_dtype_regex_list=None
    ):
        super().__init__(check_contiguous, x_dtype_regex_list, y_dtype_regex_list)
        self.minmax_downsampler = NaNMinMax_py()
        self.lttb_downsampler = LTTB_py()


class _FPCS_py(AbstractDownsampler, ABC):
    def __init__(
        self, check_contiguous=True, x_dtype_regex_list=None, y_dtype_regex_list=None
    ):
        super().__init__(check_contiguous, x_dtype_regex_list, y_dtype_regex_list)
        self.minmax_downsampler: AbstractDownsampler = None

    def _downsample(
        self, x: Union[np.ndarray, None], y: np.ndarray, n_out: int, **kwargs
    ) -> np.ndarray:
        # fmt: off
        # ------------------------- Helper datastructures -------------------------
        class Flag(Enum):
            NONE = -1   # -1: no data points have been retained
            MAX = 0     #  0: a max has been retained
            MIN = 1     #  1: a min has been retained

        Point = NamedTuple("point", [("x", int), ("y", y.dtype)])
        # ------------------------------------------------------------------------

        # NOTE: is fine for this implementation as this is only used for testing
        if x is None:
            # Is fine for this implementation as this is only used for testing
            x = np.arange(y.shape[0])

        # 0. Downsample the data using the MinMax algorithm
        MINMAX_FACTOR = 2
        n_1 = len(x) - 1
        # NOTE: as we include the first and last point, we reduce the number of points
        downsampled_idxs = self.minmax_downsampler.downsample(
            x[1:n_1], y[1:n_1], n_out=(n_out - 2) * MINMAX_FACTOR
        )
        downsampled_idxs += 1

        previous_min_flag: Flag = Flag.NONE
        potential_point = Point(0, 0)
        max_point = Point(0, y[0])
        min_point = Point(0, y[0])

        sampled_indices = []
        sampled_indices.append(0)  # prepend the first point
        for i in range(0, len(downsampled_idxs), 2):
            # get the min and max indices and convert them to the correct order
            min_idx, max_idxs = downsampled_idxs[i], downsampled_idxs[i + 1]
            if y[min_idx] > y[max_idxs]:
                min_idx, max_idxs = max_idxs, min_idx
            bin_min = Point(min_idx, y[min_idx])
            bin_max = Point(max_idxs, y[max_idxs])

            # Use the (nan-aware) comparison function to update the min and max points
            # As comparisons with NaN always return False, we inverted the comparison
            # the (inverted) > and <= stem from the pseudo code details in the paper
            if not (max_point.y > bin_max.y):
                max_point = bin_max
            if not (min_point.y <= bin_min.y):
                min_point = bin_min

            # if the min is to the left of the max
            if min_point.x < max_point.x:
                # if the min was not selected in the previous bin
                if previous_min_flag == Flag.MIN and min_point.x != potential_point.x:
                    # Both adjacent samplings retain MinPoint, and PotentialPoint and
                    # MinPoint are not the same point
                    sampled_indices.append(potential_point.x)

                sampled_indices.append(min_point.x)  # receiving min_point b4 max_point -> retain min_point
                potential_point = max_point  # update potential point to unselected max_point
                min_point = max_point  # update min_point to unselected max_point
                previous_min_flag = Flag.MIN  # min_point has been selected

            else:
                if previous_min_flag == Flag.MAX and max_point.x != potential_point.x:
                    # # Both adjacent samplings retain MaxPoint, and PotentialPoint and
                    # MaxPoint are not the same point
                    sampled_indices.append(potential_point.x)

                sampled_indices.append(max_point.x) # receiving max_point b4 min_point -> retain max_point
                potential_point = min_point  # update potential point to unselected min_point
                max_point = min_point  # update max_point to unselected min_point
                previous_min_flag = Flag.MAX  # max_point has been selected

        sampled_indices.append(len(y) - 1)  # append the last point
        # fmt: on
        return np.array(sampled_indices, dtype=np.int64)


class FPCS_py(_FPCS_py):
    def __init__(
        self, check_contiguous=True, x_dtype_regex_list=None, y_dtype_regex_list=None
    ):
        super().__init__(check_contiguous, x_dtype_regex_list, y_dtype_regex_list)
        self.minmax_downsampler = MinMax_py()


class NaNFPCS_py(_FPCS_py):
    def __init__(
        self, check_contiguous=True, x_dtype_regex_list=None, y_dtype_regex_list=None
    ):
        super().__init__(check_contiguous, x_dtype_regex_list, y_dtype_regex_list)
        self.minmax_downsampler = NaNMinMax_py()
