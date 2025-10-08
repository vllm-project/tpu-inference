from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm
from jax.experimental.shard_map import shard_map
import jax 
import jax.numpy as jnp
import numpy as np
from tpu_inference.layers.jax.sharding import EXPERT_AXIS_NAME
from tpu_inference.models.vllm.jax_fused_moe import _get_tiling_size_for_gmm_kernel
from jax.sharding import Mesh, NamedSharding, PartitionSpec

P = PartitionSpec

mesh = jax.sharding.Mesh( np.array(jax.devices()).reshape(1, 1, 4, 2), ['data', 'expert', 'model', 'attn_dp'])


lhs = jax.numpy.ones((256, 2048), dtype=jnp.float32)
rhs = jax.numpy.ones((128, 1536, 2048), dtype=jnp.float32)
group_sizes = jax.numpy.array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 32,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0, 32,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       32,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 32, 32,  0,  0,
        0,  0,  0,  0,  0, 32,  0,  0,  0,  0, 32,  0,  0,  0,  0,  0,  0,
        0, 32,  0,  0,  0,  0,  0,  0,  0], dtype=jnp.int32)
group_offset = jax.numpy.array([  0,  16,  32,  48,  64,  80,  96, 112], dtype=jnp.int32)
group_offset = jax.lax.with_sharding_constraint(
    group_offset, NamedSharding(mesh, P(EXPERT_AXIS_NAME)))

transpose_rhs = True

m, k, g = lhs.shape[0], lhs.shape[1], rhs.shape[0]
n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)


def _gmm(lhs, rhs, group_sizes, group_offset):
    # Group offset for this shard. `group_offset` is sharded, and in this sharded
    # function, it has only 1 element and `group_offset.shape` is (1,) but gmm kernel requires
    # the group_offset to be a ()-shaped array, so we group_offset[0].
    group_offset_of_shard = group_offset[0]
    return gmm(lhs=lhs,
                rhs=rhs,
                group_sizes=group_sizes,
                preferred_element_type=lhs.dtype,
                tiling=(tm, tk, tn),
                transpose_rhs=transpose_rhs,
                group_offset=group_offset_of_shard)

# breakpoint()
gmm_res = shard_map(
    _gmm,
    mesh=mesh,
    in_specs=(P(), P(EXPERT_AXIS_NAME, None, None), P(), P(EXPERT_AXIS_NAME)),
    out_specs=(P(EXPERT_AXIS_NAME, None)), 
    check_rep=False,
)(lhs, rhs, group_sizes, group_offset)
# breakpoint()
# print("gmm_res", gmm_res)