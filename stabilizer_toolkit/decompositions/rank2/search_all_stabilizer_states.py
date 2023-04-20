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

    ray_implicit_init = not ray.is_initialized()
    if ray_implicit_init:
        if num_cpus == 0:
            num_cpus = cpu_count(logical=False)
        elif num_cpus < 0:
            num_cpus = cpu_count(logical=False) - num_cpus

        ray.init(
            num_cpus=num_cpus,
            _system_config={
                # Allow spilling until the local disk is 99% utilized.
                # This only affects spilling to the local file system.
                "local_fs_capacity_threshold": 0.99,
            },
        )

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
            A = np.vstack([a, b]).T
            x, _, _, _ = np.linalg.lstsq(A, psi, rcond=-1)
            if np.all(np.isclose(A @ x, psi)):
                if debug:
                    print(f"Found decomposition:\n  {a}\n  {b}")

                # return A.T, x
                decompositions.append(A.T)
                coefficients.append(x)

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
