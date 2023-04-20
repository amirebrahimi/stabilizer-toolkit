#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
from functools import reduce

import numpy as np

from stabilizer_toolkit.helpers.unitary import get_tensored_unitary


def enumerate_ccz(num_qubits: int):
    zero = reduce(np.kron, [[1, 0]] * num_qubits)
    D = get_tensored_unitary([["CCZ"], ["H"] * num_qubits])
    state = D @ zero
    yield state
