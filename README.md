# tsdownsample

[![PyPI Latest Release](https://img.shields.io/pypi/v/tsdownsample.svg)](https://pypi.org/project/tsdownsample/)
[![support-version](https://img.shields.io/pypi/pyversions/tsdownsample)](https://img.shields.io/pypi/pyversions/tsdownsample)
[![Downloads](https://pepy.tech/badge/tsdownsample)](https://pepy.tech/project/tsdownsample)
<!-- [![Testing](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml/badge.svg)](https://github.com/predict-idlab/tsflex/actions/workflows/test.yml) -->

**📈 Time series downsampling** algorithms for visualization

## Features ✨

* **Fast**: written in rust with PyO3 bindings  
  - leverages optimized [argminmax](https://github.com/jvdd/argminmax) - which is SIMD accelerated with runtime feature detection
  - scales linearly with the number of data points
  - multithreaded with Rayon (in Rust)
    <details>
      <summary><i>Why we do not use Python multiprocessing</i></summary>
      Citing the <a href="https://pyo3.rs/v0.17.3/parallelism.html">PyO3 docs on parallelism</a>:<br>
      <blockquote>
          CPython has the infamous Global Interpreter Lock, which prevents several threads from executing Python bytecode in parallel. This makes threading in Python a bad fit for CPU-bound tasks and often forces developers to accept the overhead of multiprocessing.
      </blockquote>
      In Rust - which is a compiled language - there is no GIL, so CPU-bound tasks can be parallelized (with <a href="https://github.com/rayon-rs/rayon">Rayon</a>) with little to no overhead.
    </details>
* **Efficient**: memory efficient
  - works on views of the data (no copies)
  - no intermediate data structures are created
* **Flexible**: works on any type of data
    - supported datatypes are `f16`, `f32`, `f64`, `i16`, `i32`, `i64`, `u16`, `u32`, `u64`
    <details>
      <summary><i>!! 🚀 <code>f16</code> <a href="https://github.com/jvdd/argminmax">argminmax</a> is 200-300x faster than numpy</i></summary>
      In contrast with all other data types above, <code>f16</code> is *not* hardware supported (i.e., no instructions for f16) by most modern CPUs!! <br>
      🐌 Programming languages facilitate support for this datatype by either (i) upcasting to `f32` or (ii) using a software implementation. <br>
      💡 As for argminmax, only comparisons are needed - and thus no arithmetic operations - creating a <ins>symmetrical ordinal mapping from <code>f16</code> to <code>i16</code></ins> is sufficient. This mapping allows to use the hardware supported scalar and SIMD <code>i16</code> instructions - while not producing any memory overhead 🎉 <br>
      <i>More details are described in <a href="https://github.com/jvdd/argminmax/pull/1">argminmax PR #1</a>.</i>
    </details>
* **Easy to use**: simple & flexible API

## Install

> ❗🚨❗ This package is currently under development - no stable release yet ❗🚨❗


```bash
pip install tsdownsample
```

## Usage

```python
from tsdownsample import MinMaxLTTBDownsampler
import pandas as pd; import numpy as np

# Create a time series
y = np.random.randn(10_000_000)
s = pd.Series(y)

# Downsample to 1000 points
s_ds = MinMaxLTTBDownsampler.downsample(s, n_out=1000)
```

---

<p align="center">
👤 <i>Jeroen Van Der Donckt</i>
</p>