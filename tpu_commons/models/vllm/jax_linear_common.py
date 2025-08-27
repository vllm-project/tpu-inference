import functools

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.kernels.quantized_matmul.kernel import quantized_matmul_kernel

_quantized_matmul_kernel = functools.partial(quantized_matmul_kernel,
                                             quantize_activation=True)


def forward_unqunatized(x: jax.Array, w: jax.Array, b: jax.Array):
    output = jnp.einsum('mn,pn->mp', x, w)
    if b is not None:
        output = output + b
    return output


def forward_w8a8_int8_col_parallel(x: jax.Array, w: jax.Array, b: jax.Array,
                                   w_s: jax.Array, mesh: Mesh):
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
    output = shard_map(_quantized_matmul_kernel,
                       mesh=mesh,
                       in_specs=(P(), P('model', None), P('model')),
                       out_specs=(P(None, 'model')),
                       check_rep=False)(x, w, w_s)
    if b is not None:
        output = output + b
    return output


def forward_w8a8_int8_row_parallel(x: jax.Array, w: jax.Array, b: jax.Array,
                                   w_s: jax.Array, mesh: Mesh,
                                   reduce_results: bool):
    x = jax.lax.with_sharding_constraint(x,
                                         NamedSharding(mesh, P(None, 'model')))
    output = shard_map(_quantized_matmul_kernel,
                       mesh=mesh,
                       in_specs=(P(None, 'model'), P(None, 'model'), P()),
                       out_specs=(P(None, 'model')),
                       check_rep=False)(x, w, w_s)
    if reduce_results:
        output = shard_map(lambda x: jax.lax.psum(x, axis_name='model'),
                           mesh=mesh,
                           in_specs=P(None, 'model'),
                           out_specs=P(),
                           check_rep=False)(output)
    if b is not None:
        output = output + b
    return output


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
