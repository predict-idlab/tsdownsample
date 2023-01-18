# Contributing to tsdownsample

Welcome! We're happy to have you here. Thank you in advance for your contribution to tsdownsample.

## The basics

tsdownsample welcomes contributions in the form of Pull Requests. For small changes (e.g., bug fixes), feel free to submit a PR. For larger changes (e.g., new functionality, major refactoring), consider submitting an [Issue](https://github.com/predict-idlab/tsdownsample/issues) outlining your proposed change.

### Prerequisites

tsdownsample is written in Rust. You'll need to install the [Rust toolchain](https://www.rust-lang.org/tools/install) for development.  

This project uses the nightly version of Rust. You can install it with:

```bash
rustup install nightly
```

and then set it as the default toolchain with:

```bash
rustup default nightly
```

### tsdownsample 

The structure of the tsdownsample project is as follows:

```bash
tsdownsample
├── Cargo.toml
├── README.md
├── src
│   ├── lib.rs     # Python bindings for Rust library
├── tsdownsample   # The Python package
├── downsample_rs  # Rust library containing the actual implementation
├── tests          # Tests for the Python package
```

The Rust library is located in the `downsample_rs` directory. The Python package is located in the `tsdownsample` directory. The `src/lib.rs` file contains the Python bindings for the Rust library.

Under the hood most downsampling algorithms heavily rely on the [argminmax](https://github.com/jvdd/argminmax) - a SIMD accelerated library for finding the index of the minimum and maximum values in an array. If you want to improve the performance of the library, you could also take a look at the `argminmax` library.

### Testing

Changes to the downsample_rs library can be tested with:

```bash
cd downsample_rs
cargo test
```

Changes to the Python package can be tested using the [`Makefile`](Makefile) in the root directory of the project:

*Make sure you have the test dependencies installed:*

```bash
pip install -r test/requirements.txt          # Install test dependencies
pip install -r test/requirements-linting.txt  # Install linting dependencies
```

To run the tests:
```bash
make test
```

To run the tests and linting:
```bash
make lint
```

### Formatting 

We use [black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort) to format the Python code.

To format the code, run the following command (more details in the [Makefile](Makefile)):
```sh
make format
```

*(make sure you have the test linting dependencies installed)*

To format the Rust code, run the following command:
```sh
cargo fmt
```

---

## Improving the performance

When a PR is submitted that improves the performance of the library, we would highly appreciate if the PR also includes a (verifiable) benchmark that shows the improvement.