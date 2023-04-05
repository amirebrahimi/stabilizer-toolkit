#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
from typing import Union

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

try:
    from stabilizer_states import StabilizerStates
except ImportError:
    # we just replace the type with an exception, so that it can be used like a type (e.g. isinstance)
    StabilizerStates = TypeError


def search_all_stabilizer_states(
    psi, stabilizer_states: Union[StabilizerStates, npt.NDArray], show_progress=True, debug=False
):
    """
    Use brute-force search to find rank-2 decompositions among all stabilizer states

    :param psi: The state to search for a decomposition among the stabilizer states.
    :param stabilizer_states: The stabilizer states dataset to use.
    :param show_progress: Show progress of the search (Optional).
    :param debug: Print additional debug information (Optional).

    :return: A tuple of the decompositions found and corresponding coefficients.
    """
    if debug:
        print(f"|ψ〉= {psi}")

    if isinstance(stabilizer_states, StabilizerStates):
        states = stabilizer_states._states
    else:
        states = stabilizer_states

    num_states = len(states)
    if debug:
        print(f"Number of stabilizer states: {num_states}")

    decompositions = []
    coefficients = []
    for i, state_0 in enumerate(tqdm(states, disable=(not show_progress))):
        for j, state_1 in enumerate(states[i + 1 :]):
            A = np.vstack([state_0, state_1]).T
            x, _, _, _ = np.linalg.lstsq(A, psi.T, rcond=-1)
            if np.all(np.isclose(A @ x, psi)):
                if debug:
                    print(f"Found decomposition:\n  {state_0}\n  {state_1}")

                decompositions.append(A.T)
                coefficients.append(x)

    return decompositions, coefficients
