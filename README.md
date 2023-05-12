# Stabilizer Toolkit
[![pypi](https://img.shields.io/pypi/v/stabilizer-toolkit.svg)](https://pypi.org/project/stabilizer-toolkit/)

This Python package is a toolkit that can help with the analysis of multi-qubit magic states (e.g. their rank and 
related decompositions). The code originated to support the research for my Master's Thesis at UT Austin: 
[CCZ Magic](ccz_magic.pdf).

#### Abstract
Universal quantum computation can be achieved with a gate set that includes a generating set of Clifford gates and at least one non-Clifford gate. When it comes to classically simulating quantum computation, Clifford gates generate an important class of circuits known as stabilizer circuits, which are efficiently simulable. Stabilizer simulators can be extended to support universal quantum computation by using magic state injection gadgets and magic state decomposition. The computational scaling directly relates to the exact stabilizer rank, χ, of the magic state, which is the minimum number of stabilizer states in superposition that are necessary to represent the state.

We show that decompositions of Clifford magic states of the form |D⟩ = D|+⟩^{⊗n}, where D is a diagonal circuit composed of `Z`, `CZ`, and `CCZ` gates, have χ = 2 for n ≤ 5. We also establish that all `CCZ` circuits of n ≤ 5 qubits are Clifford equivalent to k ≤ 2 `CCZ` circuits. These results provide a new lower-bound for the complexity of simulating Clifford+`CCZ` circuits in general. We suggest the use of a state-injection gadget of |D⟩ in contrast to the naïve approach of multiple state-injection gadgets of individual |CCZ⟩ states. Since extended stabilizer simulators have a gadget-based method of simulation, this approach can improve classical simulation of quantum computation when using a Clifford+`CCZ` gate set.


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
pip install "stabilizer-toolkit[states]" 
```

There are also additional helpers that require the use of additional libraries, so you may also want to install those:
```shell
pip install "stabilizer-toolkit[states,helpers]" 
```

## Development
If you'd like to work with a development copy, then clone this repository and install via 
[`poetry`](https://python-poetry.org/docs/#installation):
```shell
poetry install -vvv --with dev --all-extras
```