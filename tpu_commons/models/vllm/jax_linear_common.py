from typing import Optional, Union

import jax
import jax.numpy as jnp
import torch
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torchax.interop import torch_view
from torchax.ops.mappings import t2j

from tpu_commons.kernels.quantized_matmul.kernel import quantized_matmul_kernel


def sharded_quantized_matmul(x: jax.Array, w_q: jax.Array, w_s: jax.Array,
                             mesh: Mesh, weight_sharding: P):
    out_axis, in_axis = weight_sharding
    x_sharding = P(None, in_axis)
    scale_sharding = P(out_axis, )
    out_sharding = P(None, out_axis)

    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, x_sharding))

    def wrapper(x, w_q, w_s):
        output = quantized_matmul_kernel(x, w_q, w_s, x_q_dtype=w_q.dtype)
        if in_axis:
            output = jax.lax.psum(output, axis_name=in_axis)
        return output

    return shard_map(wrapper,
                     mesh=mesh,
                     in_specs=(x_sharding, weight_sharding, scale_sharding),
                     out_specs=(out_sharding),
                     check_rep=False)(x, w_q, w_s)


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


def torch_to_jax_param(
    tensor: torch.Tensor,
    sharding: NamedSharding,
    output_sizes: Optional[int],
    n_shards: int,
    fused: bool,
    dim: int = 0,
    jax_dtype: Optional[jnp.dtype] = None,
) -> Union[torch.nn.Parameter, torch.nn.ParameterList]:
    if output_sizes is None:
        output_sizes = [tensor.shape[0]]

    tensor = t2j(tensor, use_dlpack=False)
    if jax_dtype:
        tensor = tensor.astype(jax_dtype)

    if fused:
        tensor = reorder_concatenated_tensor_for_sharding(
            tensor, output_sizes, n_shards, dim)
        tensor = jax.device_put(tensor, sharding)
        param = torch.nn.Parameter(torch_view(tensor), requires_grad=False)
    else:
        tensors = []
        start_offset = 0
        for size in output_sizes:
            end_offset = start_offset + size

            tensor_split = jax.lax.slice_in_dim(tensor,
                                                start_offset,
                                                end_offset,
                                                axis=dim)
            tensor_split = jax.device_put(tensor_split, sharding)
            tensor_split = torch.nn.Parameter(torch_view(tensor_split),
                                              requires_grad=False)
            tensors.append(tensor_split)

            start_offset = end_offset
        param = torch.nn.ParameterList(tensors)
    return param
