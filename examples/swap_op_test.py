import functools

import jax
import jax.numpy as jnp
import numpy as np
from einshape import jax_einshape
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference.distributed.cache_util import TokenProcessor, swap_ops
from tpu_inference.distributed.local_cpu_backend import LocalCPUBackend
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def create_default_mesh(axis_shapes, axis_names):
    """Creates a JAX device mesh with the default device order."""
    try:
        devices = jax.devices()
        print(f"Found {len(devices)} devices. Using default device order.")
        device_array = np.asarray(devices).reshape(axis_shapes)
        return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
        print("No TPU devices found. This script must be run on a TPU node.")
        return None


def test_d2h_h2d_roundtrip():
    # Define cache properties
    num_layers = 10
    num_blocks = 64
    block_size = 32
    num_heads = 8
    head_size = 128
    cache_shape = (num_blocks, block_size, num_heads, 2, head_size)
    cache_dtype = jnp.bfloat16

    cpu_backend = LocalCPUBackend()
    token_processor = TokenProcessor(model_name="debug",
                                     chunk_size=block_size // 2)

    axis_shapes = (1, 8)
    axis_names = ("data", "model")
    mesh = create_default_mesh(axis_shapes, axis_names)

    # Define shardings
    partition_spec = PartitionSpec(None, None, "model")
    device_sharding = NamedSharding(mesh, partition_spec)
    host_sharding = NamedSharding(mesh,
                                  partition_spec,
                                  memory_kind="pinned_host")

    # Create mock KV caches on TPU
    @functools.partial(jax.jit, out_shardings=device_sharding)
    def create_on_device(key):
        return jax.random.uniform(key, shape=cache_shape, dtype=cache_dtype)

    src_kv_cache = [
        create_on_device(jax.random.PRNGKey(0)) for i in range(num_layers)
    ]
    dst_kv_cache = [
        jax.device_put(jnp.zeros(cache_shape, dtype=cache_dtype),
                       device_sharding) for i in range(num_layers)
    ]
    jax.block_until_ready(src_kv_cache)
    jax.block_until_ready(dst_kv_cache)

    # target block_ids:
    target_block_ids = [2, 3]
    num_target_blocks = len(target_block_ids)
    target_tokens = [i for i in range(num_target_blocks * block_size)]
    fn_type = "pallas"

    # d2h
    # d2h: 0.swap
    extracted_blocks_tpu = [
        layer_cache_tpu[target_block_ids, ...]
        for layer_cache_tpu in src_kv_cache
    ]
    blocks_cpu = [
        swap_ops(layer_tpu, host_sharding, "d2h", fn_type)
        for layer_tpu in extracted_blocks_tpu
    ]
    jax.block_until_ready(blocks_cpu)
    print(
        f"blocks_cpu[0]: shape:{blocks_cpu[0].shape}, shardng:{blocks_cpu[0].sharding}"
    )

    # d2h: 1.flat
    flat_blocks_cpu = [
        jax_einshape("ab...->(ab)...", layer_cache)
        for layer_cache in blocks_cpu
    ]
    jax.block_until_ready(flat_blocks_cpu)
    print(
        f"flat_blocks_cpu[0]: shape:{flat_blocks_cpu[0].shape}, shardng:{flat_blocks_cpu[0].sharding}"
    )

    # d2h: 2. generate keys and store to cpu backend
    keys_generator = token_processor.process_tokens(target_tokens)
    keys = list(keys_generator)

    for i, obj in enumerate(keys):
        sidx, eidx, key = obj
        cache = [
            jax.lax.slice_in_dim(flat_layer, sidx, eidx, axis=0)
            for flat_layer in flat_blocks_cpu
        ]
        jax.block_until_ready(cache)
        print(
            f"({sidx}, {eidx}, {key}) cache: {cache[0].shape}, {cache[0].sharding}"
        )
        cpu_backend.add(key, cache)

    # h2d
    # h2d: 0. get key and fetch data blocks
    num_matched_tokens = 48
    retrieve_tokens = target_tokens[:num_matched_tokens]
    # retrieve_tokens = target_tokens
    fetch_keys_generator = token_processor.process_tokens(retrieve_tokens)
    fetch_keys = list(fetch_keys_generator)

    assemble_blocks_on_cpu = [[] for _ in range(num_layers)]
    for i, obj in enumerate(fetch_keys):
        sidx, eidx, key = obj
        cache = cpu_backend.get(key)
        if cache:
            for i in range(num_layers):
                assemble_blocks_on_cpu[i].append(cache[i])
        logger.info(
            f"get ({sidx}, {eidx}, {key}) cache: {cache[0].shape}, {cache[0].sharding}"
        )

    # h2d: 1. concat
    final_kv_on_cpu = []
    for block_list in assemble_blocks_on_cpu:
        final_kv_on_cpu.append(jnp.concatenate(block_list, axis=0))

    logger.info(
        f"final_kv_on_cpu[0]: {final_kv_on_cpu[0].shape}, {final_kv_on_cpu[0].sharding}"
    )

    # h2d: TODO: 2. padding
    padding_size = num_target_blocks * block_size - final_kv_on_cpu[0].shape[0]
    if padding_size > 0:
        padding_config = [(0, padding_size, 0)
                          ] + [(0, 0, 0)] * (len(final_kv_on_cpu[0].shape) - 1)
        pad_value = jnp.array(0, dtype=final_kv_on_cpu[0].dtype)
        padded_final_kv_on_cpu = [
            jax.lax.pad(layer_cpu, pad_value, padding_config)
            for layer_cpu in final_kv_on_cpu
        ]
        jax.block_until_ready(padded_final_kv_on_cpu)
        logger.info(
            f"padded_final_kv_on_cpu[0]: {padded_final_kv_on_cpu[0].shape}, {padded_final_kv_on_cpu[0].sharding}"
        )
    else:
        padded_final_kv_on_cpu = final_kv_on_cpu

    # h2d: 3. reshape
    # num_load_blocks = num_target_blocks
    blocked_kv_on_cpu = [
        # layer_cache.reshape(num_load_blocks, block_size, num_heads, 2,
        #                     head_size) for layer_cache in padded_final_kv_on_cpu
        jax_einshape("(nb)...->nb...", layer_cache, b=block_size)
        for layer_cache in padded_final_kv_on_cpu
    ]
    jax.block_until_ready(blocked_kv_on_cpu)
    logger.info(
        f"blocked_kv_on_cpu[0]: {blocked_kv_on_cpu[0].shape}, {blocked_kv_on_cpu[0].sharding}"
    )

    # h2d: 4. swap_in
    loaded_kv_on_tpu = [
        swap_ops(layer_cpu, device_sharding, "h2d", fn_type)
        for layer_cpu in blocked_kv_on_cpu
    ]
    jax.block_until_ready(loaded_kv_on_tpu)
    logger.info(
        f"loaded_kv_on_tpu[0]: {loaded_kv_on_tpu[0].shape}, {loaded_kv_on_tpu[0].sharding}"
    )

    # h2d: 5. split_back
    for i in range(num_layers):
        dst_kv_cache[i] = dst_kv_cache[i].at[target_block_ids,
                                             ...].set(loaded_kv_on_tpu[i])
    jax.block_until_ready(dst_kv_cache)

    remaining_tokens = num_matched_tokens
    for block_id in target_block_ids:
        print(f"testing: {block_id}")
        if remaining_tokens > block_size:
            num_test_tokens = block_size
        else:
            num_test_tokens = remaining_tokens
        np.testing.assert_array_equal(
            np.array(src_kv_cache[0][block_id][:num_test_tokens]),
            np.array(dst_kv_cache[0][block_id][:num_test_tokens]),
        )
        src_data_sample = src_kv_cache[0][block_id].flatten(
        )[:num_test_tokens // 2]
        dst_data_sample = dst_kv_cache[0][block_id].flatten(
        )[:num_test_tokens // 2]
        logger.info(
            f"---checking block {block_id}---: \n{src_data_sample}\n{dst_data_sample}"
        )
        remaining_tokens -= block_size


def test_host_input_host_to_hbm_dma():

    def kernel(x_host_ref, y_hbm_ref):

        def body(sem):
            pltpu.async_copy(x_host_ref, y_hbm_ref, sem).wait()

        pl.run_scoped(body, pltpu.SemaphoreType.DMA)

    x = jnp.arange(8 * 128.0).reshape((8, 128))
    # Move input to the host.
    x = jax.device_put(
        x,
        jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), 'x'),
            jax.sharding.PartitionSpec(),
            memory_kind='pinned_host',
        ),
    )
    y = pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pl.HOST),
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(x)
    jax.block_until_ready(y)
    print(y.sharding)
    np.testing.assert_array_equal(y, x)


def test_sharded_host_to_hbm_dma():
    from tpu_inference.kernels.dma.host_dma import h2d_dma

    axis_shapes = (1, 8)
    axis_names = ("data", "model")
    mesh = create_default_mesh(axis_shapes, axis_names)
    partition_spec = PartitionSpec(None, None, "model")
    host_sharding = NamedSharding(mesh,
                                  partition_spec,
                                  memory_kind="pinned_host")
    device_sharding = NamedSharding(mesh, partition_spec)

    data_shape = (2, 16, 8, 2, 128)
    data = jax.device_put(
        jax.random.uniform(jax.random.PRNGKey(1),
                           shape=data_shape,
                           dtype=jnp.bfloat16), host_sharding)
    jax.block_until_ready(data)
    data_tpu = h2d_dma(data, device_sharding)
    jax.block_until_ready(data_tpu)
    print(f"data_tpu: {data_tpu.shape}, {data_tpu.sharding}")
    np.testing.assert_array_equal(data, data_tpu)


if __name__ == "__main__":
    from jax._src import test_util as jtu
    if not jtu.if_cloud_tpu_at_least(2025, 8, 14):
        raise RuntimeError("libtpu version does not support DMA host-hbm")
    test_d2h_h2d_roundtrip()
