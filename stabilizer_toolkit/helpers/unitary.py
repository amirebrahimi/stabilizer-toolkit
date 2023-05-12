#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
import warnings
from functools import reduce

import numpy as np
import numpy.typing as npt

try:
    import cirq
except ImportError:
    cirq = None


def get_tensored_unitary(*gates) -> npt.NDArray:
    if cirq:
        # print(gates)
        return reduce(
            np.kron,
            [
                reduce(np.matmul, [get_tensored_unitary(*o) for o in g])
                if isinstance(g, list)
                else eval(f"cirq.{g}._unitary_()")
                for g in gates
            ],
        )
    else:
        warnings.warn("Missing optional dependency for cirq; Install via [helpers] extra")
