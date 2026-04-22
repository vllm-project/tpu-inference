print("Python script started")
import faulthandler
faulthandler.enable()
import time
import jax
import jax.numpy as jnp
import numpy as np
import raw_transfer

SUPPORTED_DTYPES = {
    jnp.bfloat16: "bf16",
    jnp.float32: "fp32",
    jnp.float8_e4m3fn: "fp8",
    jnp.int32: "int32",
}

def create_mesh(axis_shapes, axis_names, devices=None):
    try:
        num_required_devices = np.prod(axis_shapes)
        if devices is None:
            devices = jax.devices()
        devices = np.array(devices)
        if len(devices) < num_required_devices:
            print(f"Not enough devices to create mesh of shape {axis_shapes}. Have {len(devices)}, need {num_required_devices}.")
            return None
        device_array = devices[:num_required_devices].reshape(axis_shapes)
        return jax.sharding.Mesh(device_array, axis_names)
    except RuntimeError:
        print("Cannot create mesh.")
        return None

def run_test_kv_cache_perf_compare(dtype, num_layers, num_blocks):
    if dtype not in SUPPORTED_DTYPES:
        print(f"Unsupported dtype: {dtype}")
        return
    dtype_str = SUPPORTED_DTYPES[dtype]

    try:
        print("Getting TPU devices...")
        devices = jax.devices("tpu")
    except RuntimeError:
        print("No TPU devices found")
        return

    if not devices:
        print("No TPU devices found")
        return

    num_devices = len(devices)
    print(f"Found {len(devices)} TPU devices for dtype {dtype_str}")

    axis_shapes = (1, num_devices)
    axis_names = ("data", "model")
    mesh = create_mesh(axis_shapes, axis_names)
    if mesh is None:
        return

    spec = jax.sharding.PartitionSpec(None, None, "model")
    shape = (num_blocks, 128, 8, 2, 128)

    tpu_sharding = jax.sharding.NamedSharding(mesh, spec)
    key = jax.random.key(0)
    src_arrs = []
    print(f"Creating {num_layers} source arrays on TPU...")
    for i in range(num_layers):
        if dtype == jnp.int32:
            arr = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(shape)
        else:
            arr = jax.random.uniform(key, shape, dtype=dtype)
        src_arrs.append(jax.device_put(arr, tpu_sharding))
    print("Waiting for source arrays to be ready...")
    jax.block_until_ready(src_arrs)
    print("Source arrays ready.")

    pinned_host_sharding = jax.sharding.NamedSharding(mesh, spec, memory_kind="pinned_host")

    def _create_zeros():
        return jnp.zeros(shape, dtype=dtype)

    alloc_zeros = jax.jit(_create_zeros, out_shardings=pinned_host_sharding)

    dst_arrs = []
    print(f"Creating {num_layers} destination arrays on pinned host...")
    for i in range(num_layers):
        dst_arrs.append(alloc_zeros())
    print("Waiting for destination arrays to be ready...")
    jax.block_until_ready(dst_arrs)
    print("Destination arrays ready.")

    tpu_dst_arrs = []
    print(f"Creating {num_layers} destination arrays on TPU...")
    for i in range(num_layers):
        tpu_dst_arrs.append(jax.device_put(jnp.empty(shape, dtype=dtype), tpu_sharding))
    print("Waiting for TPU destination arrays to be ready...")
    jax.block_until_ready(tpu_dst_arrs)
    print("TPU destination arrays ready.")

    start = time.time()
    print("Starting transfer_d2h_batch_async...")
    futures = raw_transfer.transfer_d2h_batch_async(src_arrs, dst_arrs)
    print("Waiting for futures...")
    futures.Await()
    print(f"[{dtype_str}] D2H completed in {time.time() - start:.4f} s")

if __name__ == "__main__":
    print("Starting fully migrated standard python benchmarks...")
    run_test_kv_cache_perf_compare(jnp.bfloat16, 64, 16)
    print("All standard benchmarks completed successfully!")


