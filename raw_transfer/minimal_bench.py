import time
import jax
import jax.numpy as jnp
import numpy as np
import raw_transfer

def run_minimal_benchmark():
    print("Starting minimal benchmark...")
    shape = (16, 128, 8, 2, 128)
    devices = jax.devices("tpu")
    mesh = jax.sharding.Mesh(np.array(devices), ("data", "model"))
    spec = jax.sharding.PartitionSpec(None, None, "model")
    sharding = jax.sharding.NamedSharding(mesh, spec)
    
    arr = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(shape)
    src_arr = jax.device_put(arr, sharding)
    jax.block_until_ready(src_arr)
    
    pinned_sharding = jax.sharding.NamedSharding(mesh, spec, memory_kind="pinned_host")
    dst_arr = jax.device_put(jnp.zeros(shape, dtype=jnp.int32), pinned_sharding)
    jax.block_until_ready(dst_arr)
    
    start = time.time()
    futures = raw_transfer.transfer_d2h_async(src_arr, dst_arr)
    futures.Await()
    print(f"D2H Transfer completed in {time.time() - start:.4f} seconds!")

if __name__ == "__main__":
    run_minimal_benchmark()
