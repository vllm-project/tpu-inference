import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


class ShardingAxisName:
    SEQUENCE = ('data', 'attn_dp')
    ATTN_DATA = ('data', 'attn_dp')
    MLP_DATA = 'data'
    ATTN_HEAD = 'model'
    ATTN_TENSOR = None
    MLP_TENSOR = ('attn_dp', 'model', 'expert')
    MOE_TENSOR = ('attn_dp', 'model')
    EXPERT = ('attn_dp', 'expert', 'model')
    VOCAB = ('expert', 'model')


num_devices = 8
devices = mesh_utils.create_device_mesh((1, 1, 2, 4))
mesh = Mesh(devices, axis_names=('data', 'expert', 'attn_dp', 'model'))


@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(
        P(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR),  # x (input)
        P(ShardingAxisName.MLP_TENSOR, None)  # kernel
    ),
    out_specs=P(ShardingAxisName.ATTN_DATA, None),  # output
    check_vma=False,
)
def down_proj_sharded(x_shard, kernel_shard):
    data_idx = jax.lax.axis_index('attn_dp')
    partial_result = jnp.dot(x_shard, kernel_shard)  #[512, 2048]
    partial_result = partial_result[data_idx * 256:(data_idx + 1) * 256]
    jax.debug.print("partial_result output shape: {}", partial_result.shape)
    # Reduce-scatter in one operation
    result = jax.lax.psum(
        partial_result,
        axis_name=ShardingAxisName.MLP_TENSOR,
    )
    jax.debug.print("down_proj_sharded output shape: {}", result.shape)
    return result


# Create input tensors
# x = jnp.ones((512, 8096))
# x = jax.device_put(x, jax.sharding.NamedSharding(mesh, P(ShardingAxisName.MLP_DATA, ShardingAxisName.MLP_TENSOR)))

# kernel = jnp.ones((8096, 2048))
# kernel = jax.device_put(kernel, jax.sharding.NamedSharding(mesh, P(ShardingAxisName.MLP_TENSOR, None)))

# output = down_proj_sharded(x, kernel)

# print("Output shape:", output.shape)

# time for putting input to 8 chips. replicated
x1 = jnp.ones((512, 8096))
start_time = time.time()
x1 = jax.device_put(x1, jax.sharding.NamedSharding(mesh, P(None, None)))
jax.block_until_ready(x1)
end_time = time.time()
print("Time for replicated sharding:", end_time - start_time)

# time for putting input to 8 chips. DP sharded.
x2 = jnp.ones((1024, 8096))
start_time = time.time()
x2 = jax.device_put(
    x2, jax.sharding.NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))
jax.block_until_ready(x2)
end_time = time.time()
print("Time for DP sharding:", end_time - start_time)
