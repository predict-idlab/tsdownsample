from typing import Union

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
    return np.unique(bins)


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
        sampled_x[0] = 0
        sampled_x[-1] = x.shape[0] - 1

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
                    avg_next_x=np.mean(x[offset[i + 1] : offset[i + 2]]),
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
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())
        return np.unique(rel_idxs)


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
            rel_idxs.append(lower + y_slice.argmin())
            rel_idxs.append(lower + y_slice.argmax())
            rel_idxs.append(upper - 1)

        # NOTE: we do not use the np.unique so that all indices are retained
        return np.array(sorted(rel_idxs))
