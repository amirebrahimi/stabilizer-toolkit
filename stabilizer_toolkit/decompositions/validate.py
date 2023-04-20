#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
import sys

import numpy as np

from stabilizer_toolkit.helpers.vector import normalize


def get_decomposition(psi, *states, normalize_states=False, debug=False):
    if normalize_states:
        s = list(states)
        for i in range(len(s)):
            s[i] = normalize(s[i])
    else:
        s = states

    A = np.vstack(s).T
    x, _, _, _ = np.linalg.lstsq(A, psi, rcond=-1)
    if np.all(np.isclose(A @ x, psi)):
        if debug:
            validate_decomposition(psi, A.T, x, show=True)

        return A.T, x

    return None, None


def validate_decompositions(psi, decompositions, coefficients, show=True) -> bool:
    if show:
        if np.all(np.isreal(psi)):
            psi = np.real(psi)

        print(f"{len(decompositions)} decompositions")
        print(f"|ψ〉\t= {psi}\n")

    return all(
        validate_decomposition(psi, d, c, show=show, print_psi=False) for d, c in zip(decompositions, coefficients)
    )


def validate_decomposition(psi, decomposition, coefficients, show=True, print_psi=True) -> bool:
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

    if show:
        with np.printoptions(precision=3, linewidth=sys.maxsize, suppress=True):
            if print_psi:
                print(f" |ψ〉\t= {psi}")
            states = [f"{c[None]} * {d}\n" for d, c in zip(decomposition, coefficients)]
            tab = "\t"
            status = "✅" if valid else "❌"
            print(f"{status}\t= {(tab + '+ ').join(states)}")

    return valid
