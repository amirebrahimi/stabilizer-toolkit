#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
