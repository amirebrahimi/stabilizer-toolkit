# Stabilizer Toolkit
[![pypi](https://img.shields.io/pypi/v/stabilizer-toolkit.svg)](https://pypi.org/project/stabilizer-toolkit/)

This package is a toolkit to help with the analysis of stabilizer states, their rank, and related decompositions.
The package was built to support the research for my Master's Thesis at UT Austin: 
[CCZ Magic](https://repositories.lib.utexas.edu/handle)

## Usage

```python
```

## Installation
The package is available on [pypi](https://pypi.org/project/stabilizer-toolkit/) and installable via `pip`:
```shell
pip install stabilizer-toolkit 
```

You may also want to install the optional [stabilizer-states](https://pypi.org/project/stabilizer-states/) package, 
which provides the stabilizer datasets for n <= 6, then you can install with the `states` extras:
```shell
pip install stabilizer-toolkit[states] 
```

## Development
If you'd like to work with a development copy, then clone this repository and install via 
[`poetry`](https://python-poetry.org/docs/#installation):
```shell
poetry install -vvv --with dev -E states
```