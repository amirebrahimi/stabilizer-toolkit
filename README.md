# Stabilizer Toolkit
[![pypi](https://img.shields.io/pypi/v/stabilizer-toolkit.svg)](https://pypi.org/project/stabilizer-toolkit/)

This Python package is a toolkit that can help with the analysis of multi-qubit magic states (e.g. their rank and 
related decompositions). The code originated to support the research for my Master's Thesis at UT Austin: 
[CCZ Magic](ccz_magic.pdf).

## Usage

```python
from stabilizer_states import StabilizerStates
from stabilizer_toolkit.decompositions import rank2, validate_decompositions
from stabilizer_toolkit.magic_states import enumerate_ccz

S3 = StabilizerStates(3)
_, CCZ, _, _ = next(enumerate_ccz(3))
decompositions, coeffs = rank2.search_all_stabilizer_states(CCZ, S3)
validate_decompositions(CCZ, decompositions, coeffs)
```

For more usage take a look at the example [notebooks](notebooks).

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

There are also additional helpers that require the use of additional libraries, so you may also want to install those:
```shell
pip install stabilizer-toolkit[states,helpers] 
```

## Development
If you'd like to work with a development copy, then clone this repository and install via 
[`poetry`](https://python-poetry.org/docs/#installation):
```shell
poetry install -vvv --with dev --all-extras
```