# tsdownsample

[![PyPI Latest Release](https://img.shields.io/pypi/v/tsdownsample.svg)](https://pypi.org/project/tsdownsample/)
[![support-version](https://img.shields.io/pypi/pyversions/tsdownsample)](https://img.shields.io/pypi/pyversions/tsdownsample)
[![Downloads](https://pepy.tech/badge/tsdownsample)](https://pepy.tech/project/tsdownsample)
<!-- [![Testing](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml) -->

**📈 Time series downsampling** algorithms for visualization

## Features ✨

* **Fast**: written in rust with pyo3 bindings  
  - leverages optimized [argminmax](https://github.com/jvdd/argminmax) - which is SIMD accelerated with runtime feature detection
  - scales linearly with the number of data points
  - scales multi-threaded with rayon (rust)
* **Efficient**: memory efficient
  - works on views of the data (no copies)
  - no intermediate data structures are created
* **Flexible**: works on any type of data
    - supported datatypes are `f16`, `f32`, `f64`, `i16`, `i32`, `i64`, `u16`, `u32`, `u64`  
    *!! 🚀 `f16` [argminmax](https://github.com/jvdd/argminmax) is 200-300x faster than numpy*
* **Easy to use**: simple API

## Install

> ❗🚨❗ This package is currently under development - no stable release yet ❗🚨❗


```bash
pip install tsdownsample
```

## Usage

```python
import tsdownsample as tsds
import pandas as pd; import numpy as np

# Create a time series
y = np.random.randn(10_000_000)
s = pd.Series(y)

# Downsample to 1000 points
s_ds = tsds.minmaxlttb(s, n_out=1000)
```

---

<p align="center">
👤 <i>Jeroen Van Der Donckt</i>
</p>