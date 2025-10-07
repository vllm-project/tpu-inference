# SPDX-License-Identifier: Apache-2.0
"""utilities for collective kernels."""

import functools

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def local_barrier(left_neighbor, right_neighbor, double_barrier=True):
    """Performs a barrier with neighbors on the global barrier semaphore.

  Optionally performs a second barrier, which prevents a potential race
  when reusing the same collective_id across kernel invocations.

  Args:
    left_neighbor: Left neighbor device id.
    right_neighbor: Right neighbor device id.
    double_barrier: Whether to perform a second barrier.
  """
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor, ),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pltpu.semaphore_wait(barrier_sem, 2)
    if double_barrier:
        # The double-barrier prevents a race condition where one neighbor can
        # re-enter the kernel again on a subsequent call and increment the
        # barrier semaphore a second time. This would unblock the current device
        # even if the other neighbor is not ready yet.
        # To implement a double-barrier, we stack-allocate a second REGULAR
        # semaphore using run_scoped.
        @functools.partial(pl.run_scoped,
                           second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for neighbor in [left_neighbor, right_neighbor]:
                pltpu.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=(neighbor, ),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(second_barrier, 2)
