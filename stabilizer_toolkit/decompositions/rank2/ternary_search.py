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


def ternary_search(
    psi,
    stabilizer_states: Union[StabilizerStates, npt.NDArray],
    show_progress: bool = True,
    debug: bool = False,
    steps=[1, 2],
    num_cpus: int = -1,
):
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
    # print(num_cpus)

    # num_qubits = int(np.log2(len(psi)))
    hilbert_space_size = len(psi)
    # print(hilbert_space_size)

    # filter to only use reals first
    states_real = np.real(states[np.all(np.isreal(states), axis=1)])

    # create a ternary version of these, too
    states_ternary = np.sign(states_real).astype(np.int8)

    filtered_states = states_ternary
    # print(filtered_states.shape)

    psi_ternary = np.sign(np.real(psi)).astype(np.int8)
    # print(psi_ternary)

    # Limit max memory to remove memory pressure
    max_memory = 16 // 4 * 1024**3
    # print(max_memory)
    # print(max_states)

    # start with checking decompositions that are sums of complementary states
    filtered_states = states_ternary[np.count_nonzero(states_ternary, axis=1) == hilbert_space_size // 2]
    # print(filtered_states.shape)

    max_states = int(2 ** np.floor(np.log2(np.sqrt(max_memory / hilbert_space_size) // num_cpus)))
    # print(max_states)

    total_states_to_process = len(states_ternary)
    if 1 not in steps:
        total_states_to_process -= len(filtered_states)
    elif 2 not in steps:
        total_states_to_process = len(filtered_states)

    max_num_pending_tasks = 2 * num_cpus
    futures = []
    # num_states = len(states)
    # num_tasks = int(num_states * (num_states - 1) / 2)
    # task_size = 4096
    try:
        decompositions = []
        coefficients = []

        decomposition_tuples = []

        def add_to_decompositions(s0, s1):
            if (
                tuple(s0.tolist()),
                tuple(s1.tolist()),
            ) not in decomposition_tuples and (
                tuple(s1.tolist()),
                tuple(s0.tolist()),
            ) not in decomposition_tuples:
                decomposition, coefficient = get_decomposition(psi, s0, s1, normalize_states=True, debug=debug)
                if decomposition is not None and coefficient is not None:
                    decompositions.append(decomposition)
                    coefficients.append(coefficient)

                    decomposition_tuples.append((tuple(s0.tolist()), tuple(s1.tolist())))

                    return True

            return False

        # FIXME: Not currently calculating the actual number of tasks
        num_tasks = total_states_to_process

        progress = tqdm(total=num_tasks, disable=(not show_progress))

        if 1 in steps:
            filtered_states_ref = ray.put(filtered_states)
            start = 0
            while len(filtered_states[start:]) > 0:
                # max_states = 2 ** floor(log2(max_memory / (filtered_states.shape[0] * hilbert_space_size)))
                # print(max_states)
                clip = min(max_states, filtered_states[start:].shape[0])
                # print(subset.shape, filtered_states.shape)

                num_blocks = int(np.ceil(filtered_states[start:].shape[0] / max_states))

                # print("step 1: num blocks", num_blocks)
                @ray.remote
                def block_check(filtered_states, offset):
                    subset = filtered_states[start : start + clip]
                    block = filtered_states[start + offset : start + offset + max_states]

                    global first_checks

                    def first_checks():
                        yield subset[:, np.newaxis] + block
                        yield subset[:, np.newaxis] - block
                        # neg_subset = -subset
                        # yield neg_subset[:, np.newaxis] + block
                        # yield neg_subset[:, np.newaxis] - block

                    pairs = []
                    for summed in first_checks():
                        # found_states = np.where(np.all(summed == psi_ternary, axis=2))
                        # print(1, found_states)
                        # found_states = np.where(np.all(summed == -psi_ternary, axis=2))
                        # print(2, found_states)
                        found_states = np.where(
                            np.logical_or(np.all(summed == psi_ternary, axis=2), np.all(summed == -psi_ternary, axis=2))
                        )
                        # print(3, found_states)

                        for pair in zip(*found_states):
                            state_0 = subset[pair[0]]
                            state_1 = block[pair[1]]
                            pairs.append((state_0, state_1))

                    return pairs

                futures = []
                for i in range(num_blocks):
                    offset = i * max_states
                    # if len(futures) > max_num_pending_tasks:
                    #     completed, futures = ray.wait(futures, num_returns=1)
                    #     for output in ray.get(completed):
                    #         for pairs in output:
                    #             for state_0, state_1 in pairs:
                    #                 if not add_to_decompositions(state_0, state_1) and debug:
                    #                     print(f"{state_0} + {state_1} already in decompositions")

                    # progress.update(n=count)

                    futures.append(block_check.remote(filtered_states_ref, offset))

                while futures:
                    completed, futures = ray.wait(futures, num_returns=1)
                    for output in ray.get(completed):
                        for state_0, state_1 in output:
                            if not add_to_decompositions(state_0, state_1) and debug:
                                print(f"{state_0} + {state_1} already in decompositions")

                # filtered_states = np.delete(filtered_states, range(clip), axis=0)
                start += clip
                progress.update(clip)

        # Get all remaining tasks
        # completed, futures = ray.wait(futures, num_returns=1)
        # for output in ray.get(completed):
        #     for pairs in output:
        #         for state_0, state_1 in pairs:
        #             if not add_to_decompositions(state_0, state_1) and debug:
        #                 print(f"{state_0} + {state_1} already in decompositions")

        if 2 in steps:
            # all other states
            filtered_states = states_ternary[np.count_nonzero(states_ternary, axis=1) != hilbert_space_size // 2]
            filtered_states_ref = ray.put(filtered_states)
            start = 0
            while len(filtered_states[start:]) > 0:
                # max_states = 2 ** floor(log2(max_memory / (filtered_states.shape[0] * hilbert_space_size)))
                clip = min(max_states, filtered_states[start:].shape[0])

                num_blocks = int(np.ceil(filtered_states[start:].shape[0] / max_states))

                # print("step 2: num blocks", num_blocks)
                @ray.remote
                def block_check(filtered_states, offset):
                    subset = filtered_states[start : start + clip]
                    block = filtered_states[start + offset : start + offset + max_states]

                    global second_checks

                    def second_checks():
                        doubled_states = 2 * block
                        yield subset[:, np.newaxis] + doubled_states
                        yield subset[:, np.newaxis] - doubled_states
                        # neg_subset = -subset
                        # yield neg_subset[:,np.newaxis] + doubled_states
                        # yield neg_subset[:,np.newaxis] - doubled_states
                        doubled_subset = 2 * subset
                        yield doubled_subset[:, np.newaxis] + block
                        yield doubled_subset[:, np.newaxis] - block
                        # neg_doubled_subset = -doubled_subset
                        # yield neg_doubled_subset[:,np.newaxis] + block
                        # yield neg_doubled_subset[:,np.newaxis] - block

                    pairs = []
                    for summed in second_checks():
                        # found_states = np.where(np.all(summed == psi_ternary, axis=2))
                        # print(1, found_states)
                        # found_states = np.where(np.all(summed == -psi_ternary, axis=2))
                        # print(2, found_states)
                        found_states = np.where(
                            np.logical_or(np.all(summed == psi_ternary, axis=2), np.all(summed == -psi_ternary, axis=2))
                        )
                        # print(3, found_states)

                        for pair in zip(*found_states):
                            state_0 = subset[pair[0]]
                            state_1 = block[pair[1]]
                            pairs.append((state_0, state_1))

                    return pairs

                futures = []
                for i in range(num_blocks):
                    offset = i * max_states
                    futures.append(block_check.remote(filtered_states_ref, offset))

                while futures:
                    completed, futures = ray.wait(futures, num_returns=1)
                    for output in ray.get(completed):
                        for state_0, state_1 in output:
                            if not add_to_decompositions(state_0, state_1) and debug:
                                print(f"{state_0} + {state_1} already in decompositions")

                # filtered_states = np.delete(filtered_states, range(clip), axis=0)
                start += clip
                progress.update(clip)
    except Exception as ex:
        [ray.cancel(f) for f in futures]
        raise ex
    finally:
        progress.close()

        if ray_implicit_init:
            ray.shutdown()

    return decompositions, coefficients
