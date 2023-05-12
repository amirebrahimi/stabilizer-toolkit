#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
from functools import reduce
from itertools import combinations, permutations

import numpy as np

from stabilizer_toolkit.helpers.unitary import get_tensored_unitary
from stabilizer_toolkit.helpers.vector import to_tuple, normalize


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    # From https://stackoverflow.com/a/47521145
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def ccz_circuit_matrix(num_qubits: int):
    # Get all the ways to place a CCZ on n qubits
    ccz_indices = list(combinations(range(num_qubits), 3))
    circuit_matrix = np.zeros((num_qubits, len(ccz_indices)))
    for i, indices in enumerate(ccz_indices):
        circuit_matrix[indices, i] = 1

    return np.array(circuit_matrix, dtype=np.uint8)


def distinct_circuits(num_qubits: int):
    def order_columns(A):
        """Sort columns based on binary value of column (little-endian from the bottom)"""
        diff = 8 - A.shape[0] % 8
        if diff > 0:
            B = np.r_["0", np.zeros((diff, A.shape[1]), int), A]
        order = np.argsort(-np.packbits(B, axis=0)[0])
        return A[:, order]  # sort in descending order

    A = ccz_circuit_matrix(num_qubits)
    num_columns = A.shape[1]

    # Iterate through all the combinations of CCZs
    unique_circuits = set()
    for i in range(2**num_columns):
        x = bin_array(i, num_columns)
        C = A @ np.diag(x)  # this selects the CCZs from the overall array
        C = C[:, ~np.all(C == 0, axis=0)]  # remove all 0 columns

        distinct = True
        # try all possible row swaps
        for p in permutations(range(num_qubits), num_qubits):
            c = to_tuple(
                order_columns(C[list(p)])
            )  # after row swaps the value may have changed, so re-sort in descending order
            if c in unique_circuits:
                distinct = False
                break

        if distinct:
            # check for sums over rows to see if this is a circuit in the subspace of n-1 qubits
            if not np.any(np.sum(C, axis=1) == 0):
                c = to_tuple(C)
                unique_circuits.add(c)

                circuit_index = int(np.array2string(x, separator="")[1:-1], 2)
                yield circuit_index, C


def enumerate_ccz(num_qubits: int):
    plus = normalize(reduce(np.kron, [np.array([1, 1])] * num_qubits))

    for circuit_index, circuit in distinct_circuits(num_qubits):
        cczs = np.ones((2**num_qubits,))

        # Iterate through all possible states and determine phase for that state
        for i in range(2**num_qubits):
            x = bin_array(i, num_qubits)
            num_cczs = np.count_nonzero(x.T @ circuit == 3)
            cczs[i] *= (-1) ** num_cczs

        D = np.diag(cczs)
        state = D @ plus
        yield circuit_index, state, D, circuit
