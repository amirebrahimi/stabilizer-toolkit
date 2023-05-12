#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
from typing import Union

import numpy as np
import numpy.typing as npt
import ray
from psutil import cpu_count
from ray import remote
from tqdm import tqdm

from stabilizer_toolkit.decompositions.validate import get_decomposition
from stabilizer_toolkit.helpers.ray import initialize_ray

try:
    from stabilizer_states import StabilizerStates
except ImportError:
    # we just replace the type with an exception, so that it can be used like a type (e.g. isinstance)
    StabilizerStates = TypeError


def search_all_stabilizer_states(
    psi,
    stabilizer_states: Union[StabilizerStates, npt.NDArray],
    show_progress: bool = True,
    debug: bool = False,
    num_cpus: int = -1,
):
    """
    Use brute-force search to find rank-2 decompositions among all stabilizer states

    :param psi: The state to search for a decomposition among the stabilizer states.
    :param stabilizer_states: The stabilizer states dataset to use.
    :param show_progress: Show progress of the search (Optional).
    :param debug: Print additional debug information (Optional).
    :param num_cpus: The number of CPUs to use for Ray where negative values exclude CPUs, 0 uses all CPUs, and positive
        values reserve that specific number of CPUs (Optional; Default=-1).

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

    ray_implicit_init, num_cpus = initialize_ray(num_cpus)

    psi_ref = ray.put(psi)
    states_ref = ray.put(states)

    @remote
    def check_decomposition(psi, states, i, start_j, range_j):
        decompositions = []
        coefficients = []
        count = 0
        a = states[i]
        for j in range(start_j, start_j + range_j):
            count += 1
            b = states[j]
            decomposition, coefficient = get_decomposition(psi, a, b, normalize_states=True, debug=debug)
            if decomposition is not None and coefficient is not None:
                decompositions.append(decomposition)
                coefficients.append(coefficient)

        return decompositions, coefficients, count

    max_num_pending_tasks = 2 * num_cpus
    futures = []
    num_states = len(states)
    num_tasks = int(num_states * (num_states - 1) / 2)
    task_size = 4096
    try:
        decompositions = []
        coefficients = []
        progress = tqdm(total=num_tasks, disable=(not show_progress))
        for i in range(num_states):
            for j in range(i + 1, num_states, task_size):
                if len(futures) > max_num_pending_tasks:
                    completed, futures = ray.wait(futures, num_returns=1)
                    for output in ray.get(completed):
                        if output is not None:
                            d, c, count = output
                            decompositions.extend(d)
                            coefficients.extend(c)
                            progress.update(n=count)

                futures.append(check_decomposition.remote(psi_ref, states_ref, i, j, min(task_size, num_states - j)))

        # Get all remaining tasks
        for output in ray.get(futures):
            if output is not None:
                d, c, count = output
                decompositions.extend(d)
                coefficients.extend(c)
                progress.update(n=count)
    except Exception as ex:
        [ray.cancel(f) for f in futures]
        raise ex
    finally:
        progress.close()

        if ray_implicit_init:
            ray.shutdown()

    return decompositions, coefficients
