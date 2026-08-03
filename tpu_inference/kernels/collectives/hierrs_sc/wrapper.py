# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Top-level dispatcher for Reduce-Scatter."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from tpu_inference.kernels.collectives.hierrs_sc.config import Config
from tpu_inference.kernels.collectives.hierrs_sc.kernel import (scs_kernel,
                                                                tec_kernel)


def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int | None = None,
    axis_name: str | tuple[str, ...] = "x",
) -> jax.Array:
    """Performs hierarchical Reduce-Scatter on SparseCore.

  This performs hierarchical recursive halving algorithm to perform Reduce
  scatter operation on SparseCore. It uses a 2-stage pipelined execution to
  overlap Die-to-Die ICI, Chip-to-Chip ICI, and local compute:

  Args:
    local_x: Local input shard of shape `[num_tokens, hidden_dim]`, representing
      the partial sum that the current device owns.
    num_devices: Total number of devices forming the reduction ring. Devices
      must be ordered by physical topology coordinates, with the chiplet
      dimension positioned as the last coordinate.
    num_micro_batches: Number of micro-batches to split the hidden dimension
      into for pipelining. If None, determined by heuristic.
    axis_name: Mesh axis name mapped explicitly out to the enclosing
      `jax.lax.shard_map` context.

  Returns:
    The reduced output array shard of shape `[num_tokens // num_devices,
    hidden_dim]`.
  """
    num_tokens, hidden_dim_size = local_x.shape
    chunk_size_orig = num_tokens // num_devices
    # Pad the local input to the minimum chunk size.
    min_chunk_size = pltpu.get_tpu_info().get_sublane_tiling(local_x.dtype)
    pad_amount = max(0, min_chunk_size - chunk_size_orig)
    reshaped_x = local_x.reshape(num_devices, -1, hidden_dim_size)
    padded_x = jnp.pad(
        reshaped_x,
        ((0, 0), (0, pad_amount), (0, 0)),
    )
    local_x = padded_x.reshape(-1, hidden_dim_size)
    num_tokens = local_x.shape[0]

    chunk_size = num_tokens // num_devices

    config = Config(
        num_devices=num_devices,
        hidden_dim_size=hidden_dim_size,
        chunk_size=chunk_size,
        num_tokens=num_tokens,
        dtype=local_x.dtype,
        _num_micro_batches=num_micro_batches,
    )

    scs_mesh = plsc.ScalarSubcoreMesh(axis_name="core",
                                      num_cores=config.num_cores)
    tec_mesh = plsc.VectorSubcoreMesh(
        core_axis_name="core",
        subcore_axis_name="subcore",
        num_cores=config.num_cores,
        num_subcores=config.num_subcores,
    )

    out, _, _ = pl.kernel(
        interpret=False,
        body=[
            # SCS (SparseCore Sequencer) exclusively manages async D2D and C2C ICI
            # operations. TEC (Tile Core) strictly executes vector ALU
            # instructions for the accumulation math. They run concurrently as
            # decoupled cores, maintaining pipeline synchronization dynamically
            # utilizing hardware semaphores to gracefully hand-off buffers between
            # copies and compute.
            functools.partial(scs_kernel, config=config, axis_name=axis_name),
            functools.partial(tec_kernel, config=config, axis_name=axis_name),
        ],
        mesh=[scs_mesh, tec_mesh],
        out_type=(
            # output_ref
            jax.ShapeDtypeStruct((config.chunk_size, hidden_dim_size),
                                 local_x.dtype),
            # running_sum_ref[i, ...]: The accumulated result at each step (i=0 is reserved for Phase 1)
            jax.ShapeDtypeStruct((config.num_hcube_dims, *local_x.shape),
                                 local_x.dtype),
            # recv_buf_ref[i, ...]: The received data from peer at each step (i=0 is reserved for Phase 1)
            jax.ShapeDtypeStruct((config.num_hcube_dims + 1, *local_x.shape),
                                 local_x.dtype),
        ),
        scratch_types=dict(
            scs_to_tec=pltpu.SemaphoreType.REGULAR(
                (2, config.num_hcube_dims + 1)) @ tec_mesh,
            tec_to_scs=pltpu.SemaphoreType.REGULAR(
                (2, config.num_hcube_dims + 1)) @ scs_mesh,
            p1_send_sem=pltpu.SemaphoreType.DMA(
                (2, config.num_chips)) @ scs_mesh,
            p2_send_sem=pltpu.SemaphoreType.DMA((
                2,
                config.num_hcube_dims,
                config.num_chips // config.cores_per_chip,
                config.num_hcube_dims,
            )) @ scs_mesh,
            p1_recv_sem=pltpu.SemaphoreType.DMA(
                (2, config.num_chips)) @ scs_mesh,
            p2_recv_sem=pltpu.SemaphoreType.DMA((
                2,
                config.num_hcube_dims,
                config.num_chips // config.cores_per_chip,
                config.num_hcube_dims,
            )) @ scs_mesh,
        ),
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True, ),
    )(local_x)

    out = out[:chunk_size_orig, :]
    return out
