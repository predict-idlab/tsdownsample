from setuptools import setup

from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension("tsdownsample._rust.tsdownsample_rs", binding=Binding.PyO3)
    ],
)
