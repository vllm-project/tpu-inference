# Copyright 2025 Google LLC
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

import os
from contextlib import nullcontext

import jax
import jax.numpy as jnp
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding, Sharding

from tpu_inference import envs

# Lazy initialized, since device might not be ready at import time.
_cpu_mesh = None


def reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array,
                                             split_sizes: list[int],
                                             n_shards: int, dim: int):
    """
    Reorder a replicated concatenated tensor such that when sharded on multiple chips, each shard is a concatenation of the shards of the individual tensors.
    For example, let the concatenated_tensor be:
        AAAAAAAAAAAABBBBBBBBCCCC
            12 As     8 Bs  4 Cs
    and let the split_sizes = [12, 8, 4] and n_shards = 4.
    The output is:
        AAABBCAAABBCAAABBCAAABBC
    In other words, it reorders the input tensor into 4 segements, with each segment corresponding to a shard and being AAABBC.
    Args:
        concatenated_tensor: the tensor, concatenated on the dimension specified by `dim`.
        split_sizes: each individual tensor's size on the dimension specified by `dim`.
        n_shards: num of shards.
        dim: the dimension on which the concatenated_tensor is concatenated.
    """
    # Split the concatenated tensor into individual tensors.
    if dim < 0:
        dim += concatenated_tensor.ndim
    split_tensors = []
    start_offset = 0
    old_shape = concatenated_tensor.shape
    # New shape ensures each split_tensor[i] maps to a tensor in ith shards
    new_shape = old_shape[:dim] + (n_shards, -1) + old_shape[dim + 1:]
    for split_size in split_sizes:
        split_tensor = jax.lax.slice_in_dim(concatenated_tensor,
                                            start_offset,
                                            start_offset + split_size,
                                            axis=dim)
        split_tensors.append(split_tensor.reshape(new_shape))
        start_offset += split_size
    # While maintaining 0th dim as a shard dim, we concatenate along 1th dim to
    # to create concatenated tnensor where 0th dim maps to shard dim.
    reordered_tensor = jnp.concatenate(split_tensors, axis=dim + 1)
    return reordered_tensor.reshape(old_shape)


def slice_sharded_tensor_for_concatenation(sharded_tensor: jax.Array,
                                           split_sizes: list[int],
                                           n_shards: int):
    """
    Slice the input tensor which is sharded on multiple chips (on the last dim) into individual tensors with the same sharding.
    For example, let the sharded_tensor be:
        AAABBC | AAABBC | AAABBC | AAABBC
        Shard0   Shard1   Shard2   Shard3
    and let the split_sizes = [12, 8, 4] and n_shards = 4.
    The output is a list of 3 tensors:
         AAA   |  AAA   |  AAA   |  AAA
          BB   |   BB   |   BB   |   BB
           C   |    C   |    C   |    C
        Shard0   Shard1   Shard2   Shard3
    In other words, each individual tensor is a slice of the input tensor with the same sharding.
    Args:
        sharded_tensor: the input tensor, sharded on the last dim.
        split_sizes: each individual tensor's size on the last dim.
        n_shards: num of shards.
    """
    new_shape = sharded_tensor.shape[:-1] + (n_shards, -1)
    # New shape ensures each sharded_tensor[:, i] maps to a tensor in ith shards
    sharded_tensor = sharded_tensor.reshape(new_shape)

    split_tensors = []
    start_offset = 0
    for split_size in split_sizes:
        assert split_size % n_shards == 0
        sz = split_size // n_shards  # size of this split tensor per shard
        end_offset = start_offset + sz
        # Because we are slicing over last dim, sharding dim remains intact.
        # Therefore, splitting happens locally.
        split_tensor = sharded_tensor[..., start_offset:end_offset]
        split_tensors.append(split_tensor.reshape(new_shape[:-2] + (-1, )))
        start_offset = end_offset

    return split_tensors


def general_device_put(tensor: jax.Array,
                       sharding: Sharding,
                       *,
                       layout: Layout | None = None,
                       source_mesh: Mesh | None = None,
                       global_shape: tuple[int, ...] | None = None) -> jax.Array:
    """
    Put a tensor onto devices with the given sharding.
    This method handles both single-host and multi-host cases.

    `source_mesh` specifies the mesh on which the input tensor is currently located.

    `global_shape` is used in multi-host EP mode where each host holds a
    local shard of the weights.  When provided, we use
    `jax.make_array_from_process_local_data` to construct the correct
    global distributed array without triggering cross-host communication.
    """
    from jax.tree_util import tree_map

    # TP-selective loader: if the per-leaf tensor was pre-sliced by the
    # multi-host loader and its original full shape registered, pick that up
    # here so `make_array_from_process_local_data` sees the right global
    # shape without every caller having to plumb it.  No-op when the
    # registry is empty (env flag off) or the tensor has no torch data_ptr
    # (already converted to a JAX array upstream).
    try:
        from tpu_inference.models.vllm.tp_selective_loader import \
            lookup_tp_full_shape
    except Exception:  # noqa: BLE001
        lookup_tp_full_shape = lambda _t: None  # type: ignore

    def _put(t):
        # Consult TP-selective registry only when no explicit global_shape was
        # supplied by the caller.  EP/MoE paths set global_shape directly and
        # take precedence.
        effective_global = global_shape
        if effective_global is None:
            effective_global = lookup_tp_full_shape(t)

        multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "")
        # Single-host or non-Ray: use jax.device_put directly.
        if multihost_backend != "ray" or (isinstance(t, jax.Array)
                                          and not t.is_fully_addressable):
            if layout is not None:
                return jax.device_put(t, Format(layout, sharding))
            else:
                return jax.device_put(t, sharding)

        # Multi-host Ray mode.
        # When a global_shape is provided (either by the caller for EP/MoE or
        # by the TP-selective registry lookup above), each host holds only
        # its local shard.  Use make_array_from_process_local_data to
        # construct the correct global distributed array.
        if effective_global is not None:
            return jax.make_array_from_process_local_data(
                sharding, t, effective_global)

        # Replicated/TP-sharded weights (no global_shape): all hosts have
        # identical data.  Use make_array_from_process_local_data instead of
        # device_put: the latter triggers process_allgather in multi-host
        # mode to verify data consistency across hosts.  That allgather
        # collects all hosts' data onto every chip, requiring
        # local_data × num_hosts HBM which causes OOM (e.g., 14.18 GB).
        # make_array_from_process_local_data skips this verification and
        # places data directly on devices.
        # CRITICAL: Must pass global_shape=t.shape explicitly. Without it,
        # JAX infers global_dim = local_dim × (total_mesh / local_mesh) which
        # is wrong for replicated weights where all hosts have identical data.
        # For example: local_dim=16384, total_mesh=32, local_mesh=4 →
        # inferred global = 16384 × 8 = 131072, per-device = 4096 (should be 512).
        arr = jax.make_array_from_process_local_data(sharding, t, t.shape)

        # If a Layout is needed, apply it on top of the distributed array.
        # Since the array is already distributed (is_fully_addressable=False),
        # this only transforms the local shard without triggering allgather.
        if layout is not None:
            return jax.device_put(arr, Format(layout, sharding))
        return arr

    return tree_map(_put, tensor)


def cpu_mesh() -> Mesh:
    global _cpu_mesh
    if _cpu_mesh is None:
        _cpu_mesh = Mesh(jax.devices("cpu")[:1], ("cpu", ))
    return _cpu_mesh


def cpu_mesh_context():
    """A context to enforce using CPU mesh, used for loading weights on CPU."""
    return jax.set_mesh(cpu_mesh())
