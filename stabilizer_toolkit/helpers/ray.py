#
# Copyright 2023, Amir Ebrahimi. All Rights Reserved.
#
import ray
from psutil import cpu_count


def initialize_ray(num_cpus: int):
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

    return ray_implicit_init, num_cpus
