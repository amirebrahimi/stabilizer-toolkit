#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
import sys

import numpy as np


def validate_decomposition(psi, decomposition, coefficients, show=True):
    """
    Validate that a decomposition matches the state.

    :param psi: The state that the decomposition should match.
    :param decomposition: The stabilizer state decomposition.
    :param coefficients: The coefficients to apply to the stabilizer states.
    :return: True if the decomposition is valid; Otherwise, False.
    """
    if len(decomposition) != len(coefficients):
        raise ValueError("Count of decompositions and coefficients do not match.")

    if np.all(np.isreal(psi)):
        psi = np.real(psi)
        decomposition = np.real(decomposition)
        if np.all(np.isreal(coefficients)):
            coefficients = np.real(coefficients)

    # valid = np.all(np.isclose(psi, np.sum(decomposition * coefficients[:, None], axis=0)))
    valid = np.all(np.isclose(psi, decomposition.T @ coefficients))

    with np.printoptions(precision=3, linewidth=sys.maxsize, suppress=True):
        if show:
            print(f"{'✅' if valid else '❌'} |ψ〉\t= {psi}")
            states = [f"{c[None]} * {d}\n" for d, c in zip(decomposition, coefficients)]
            tab = "\t"
            print(f"\t= {(tab + '+ ').join(states)}")

    return valid
