import functools
from enum import Enum

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.kernels.quantized_matmul.kernel import quantized_matmul_kernel


class ParallelType(Enum):
    COL_PARALLEL = 1
    ROW_PARALLEL = 2


def forward_unqunatized(x: jax.Array, w: jax.Array, b: jax.Array):
    output = jnp.einsum('mn,pn->mp', x, w)
    if b is not None:
        output = output + b
    return output


def forward_w8a8_int8(x: jax.Array, w: jax.Array, b: jax.Array, w_s: jax.Array,
                      mesh: Mesh, collective_combine_results: bool,
                      parallel_type: ParallelType):

    _quantized_matmul_kernel = functools.partial(quantized_matmul_kernel,
                                                 quantize_activation=True)

    def _w8a8_int8_quant_matmul_col(x, w, w_s):
        x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
        output = shard_map(_quantized_matmul_kernel,
                           mesh=mesh,
                           in_specs=(P(), P('model', None), P('model')),
                           out_specs=(P(None, 'model')),
                           check_rep=False)(x, w, w_s)
        if collective_combine_results:  # Gather results
            output = jax.lax.with_sharding_constraint(output,
                                                      NamedSharding(mesh, P()))
        return output

    def _w8a8_int8_quant_matmul_row(x, w, w_s):
        x = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P(None, 'model')))
        output = shard_map(_quantized_matmul_kernel,
                           mesh=mesh,
                           in_specs=(P(None, 'model'), P(None, 'model'), P()),
                           out_specs=(P(None, 'model')),
                           check_rep=False)(x, w, w_s)
        if collective_combine_results:  # Reduce results
            output = shard_map(lambda x: jax.lax.psum(x, axis_name='model'),
                               mesh=mesh,
                               in_specs=P(None, 'model'),
                               out_specs=P(),
                               check_rep=False)(output)
        return output

    if parallel_type == ParallelType.COL_PARALLEL:
        output = _w8a8_int8_quant_matmul_col(x, w, w_s)
    elif parallel_type == ParallelType.ROW_PARALLEL:
        output = _w8a8_int8_quant_matmul_row(x, w, w_s)

    if b is not None:
        output = output + b
    return output


def reorder_concatenated_tensor_for_sharding(concatenated_tensor: jax.Array,
                                             split_sizes: list[int],
                                             n_shards: int):
    """
    Reorder a replicated concatenated tensor such that when sharded on multiple chips, each shard is a concatenation of the shards of the individual tensors.

    Args:
        concatenated_tensor: the tensor, concatenated on the 0-th dim.
        split_sizes: each individual tensor's size on the 0-th dim.
        n_shards: num of shards.
    """

    # Split the concatenated tensor into individual tensors.
    split_tensors = []
    start_offset = 0
    for i, split_size in enumerate(split_sizes):
        split_tensor = concatenated_tensor[start_offset:start_offset +
                                           split_size]
        split_tensors.append(split_tensor)
        start_offset += split_size

    # For each shard, collect the portion of each split tensor that should be on this shard.
    reordered_tensor_shards = []
    for j in range(n_shards):
        split_tensors_for_this_shard = []
        for i, split_size in enumerate(split_sizes):
            sz = split_size // n_shards  # size of this split tensor per shard
            split_tensor = split_tensors[i]
            t = split_tensor[j * sz:(j + 1) * sz]
            split_tensors_for_this_shard.append(t)
        tensor_shard = jnp.concatenate(split_tensors_for_this_shard)
        reordered_tensor_shards.append(tensor_shard)

    reordered_tensor = jnp.concatenate(reordered_tensor_shards)
    return reordered_tensor


def slice_sharded_tensor_for_concatenation(sharded_tensor: jax.Array,
                                           split_sizes: list[int],
                                           n_shards: int, mesh: Mesh):
    """
    Slice the input tensor which is sharded on multiple chips into individual tensors with the same sharding.

    Args:
        sharded_tensor: the input tensor, sharded on the last dim.
        split_sizes: each individual tensor's size on the last dim.
        n_shards: num of shards.
    """

    def _get_slice_on_shard(tensor_shard, start_offset, end_offset):
        return tensor_shard[..., start_offset:end_offset]

    sharding_spec = P(*((None, ) * (sharded_tensor.ndim - 1) +
                        ('model', )))  # Shard on the last dim.

    split_tensors = []
    start_offset = 0
    for i, split_size in enumerate(split_sizes):
        sz = split_size // n_shards  # size of this split tensor per shard
        end_offset = start_offset + sz
        _get_slice_on_shard_bound = functools.partial(
            _get_slice_on_shard,
            start_offset=start_offset,
            end_offset=end_offset)
        start_offset = end_offset
        split_tensor = shard_map(_get_slice_on_shard_bound,
                                 mesh=mesh,
                                 in_specs=(sharding_spec),
                                 out_specs=(sharding_spec),
                                 check_rep=False)(sharded_tensor)
        split_tensors.append(split_tensor)

    return split_tensors
