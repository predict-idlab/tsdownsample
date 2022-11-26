.DEFAULT_GOAL := all
isort = isort tsdownsample tests
black = black tsdownsample tests

install:
	pip install -r requirements.txt
	pip install -e .

.PHONY: format
format:
	$(isort)
	$(black)
	cargo fmt

.PHONY: lint-python
lint-python:
	ruff tsdownsample tests
	$(isort) --check-only --df
	$(black) --check --diff

.PHONY: lint-rust
lint-rust:
	cargo fmt --version
	cargo fmt --all -- --check
	cargo clippy --version
	cargo clippy -- -D warnings -A incomplete_features -W clippy::dbg_macro -W clippy::print_stdout

.PHONY: lint
lint: lint-python lint-rust

.PHONY: mypy
mypy:
	mypy tsdownsample


.PHONY: test
test:
	pytest --cov=tsdownsample --cov-report=term-missing --cov-report=html --cov-report=xml


.PHONY: all
all: lint mypy test

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '*.cpython-*' `
	rm -rf dist
	rm -rf build
	rm -rf target
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -f tsdownsample/*.so