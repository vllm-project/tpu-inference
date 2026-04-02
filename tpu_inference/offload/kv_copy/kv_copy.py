# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for array copying on TPU."""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

Future = Any
shard_map = jax.shard_map
P = jax.sharding.PartitionSpec
Mesh = jax.sharding.Mesh

_NUM_CHUNKS_CACHE = {}


def _simple_copy_kernel(src_ref, host_ref_in, host_ref_out):
    del host_ref_in

    def body(sem):
        pltpu.async_copy(src_ref, host_ref_out, sem).wait()

    pl.run_scoped(body, pltpu.SemaphoreType.DMA)


@functools.lru_cache(None)
def _get_copy_to_dest_fn(mesh, sharding_spec, out_sharding, dtype, memory_kind):
    """Returns a cached JIT-compiled copy function."""
    memory_space = pltpu.HOST if memory_kind == 'pinned_host' else pltpu.HBM

    def _copy_wrapped(src, dest):
        return pl.pallas_call(
            _simple_copy_kernel,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=memory_space),
            ],
            out_specs=pl.BlockSpec(memory_space=memory_space),
            input_output_aliases={1: 0},
            out_shape=memory_space(shape=dest.shape, dtype=dtype),
        )(src, dest)

    @functools.partial(
        jax.jit, 
        out_shardings=out_sharding, 
        donate_argnames=('dest',)
    )
    def _copy(src, dest):
        result = []
        for x, y in zip(src, dest):
            result.append(jax.shard_map(
                _copy_wrapped,
                mesh=mesh,
                in_specs=(sharding_spec, sharding_spec),
                out_specs=sharding_spec,
                check_vma=False,
            )(x, y))
        return result

    # def _copy(src, dest):
    #     return jax.shard_map(
    #         _copy_wrapped,
    #         mesh=mesh,
    #         in_specs=(sharding_spec, sharding_spec),
    #         out_specs=sharding_spec,
    #         check_vma=False,
    #     )(src, dest)

    return _copy


def copy_to_dest(src, dest, mesh, sharding_spec, out_sharding, out_memory_kind, dtype):
    """"Copies from src to dest."""
    _copy = _get_copy_to_dest_fn(
        mesh, sharding_spec, out_sharding, dtype, out_memory_kind
    )
    return _copy(src, dest)
    # return jax.tree.map(_copy, src, dest)