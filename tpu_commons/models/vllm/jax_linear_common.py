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
