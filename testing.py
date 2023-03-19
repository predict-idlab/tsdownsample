from datetime import datetime

import numpy as np
import polars as pl

from tsdownsample import LTTBDownsampler, MinMaxDownsampler, M4Downsampler

data = {
    "timestamp": [
        datetime(year=2023, month=1, day=1),
        datetime(year=2023, month=1, day=2),
        datetime(year=2023, month=1, day=3),
    ],
    "value": [1, 2, 3],
}
df = pl.DataFrame(data)

indices = LTTBDownsampler().downsample(np.array(df["timestamp"]), np.array(df["value"]), n_out=3, n_threads=4)
indices = MinMaxDownsampler().downsample(np.array(df["timestamp"]), np.array(df["value"]), n_out=2, n_threads=4)
indices = M4Downsampler().downsample(np.array(df["timestamp"]), np.array(df["value"]), n_out=3, n_threads=4)